from __future__ import print_function

import argparse
import os
import json
import queue
import torch
import yaml
import threading
import struct
import time
import torchaudio
import datetime
import builtins
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copy import deepcopy
from threading import Timer
from flask import Flask, render_template, request
from flask_socketio import SocketIO, disconnect, emit
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from web.parms import GlobalParams
from web.pool import pipelineObjectPool
from web.pem import generate_self_signed_cert
from web.queue import PCMQueue, ProcPCMQueue, ThreadSafeQueue
from periphrals.PureVAD import PureVAD
from periphrals.AudioFeatureGating import AudioFeatureGating


def get_args():
    parser = argparse.ArgumentParser(description='Freeze-Omni Dialog State Server')
    # Keep essential paths as arguments, but most settings will move to YAML
    parser.add_argument('--config', type=str, default='configs/server_config.yaml', help='Path to the server configuration YAML file.')
    # parser.add_argument('--model_path', required=False, help='model_path to load (overrides config file)')
    # parser.add_argument('--llm_path', required=False, help='llm_path to load (overrides config file)')

    # parser.add_argument('--model_path', required=True, help='model_path to load')
    # parser.add_argument('--llm_path', required=True, help='llm_path to load')
    # parser.add_argument('--top_k', type=int, default=5)
    # parser.add_argument('--top_p', type=float, default=0.8)
    # parser.add_argument('--temperature', type=float, default=0.7)
    # parser.add_argument('--ip', required=True, help='ip of server')
    # parser.add_argument('--port', required=True, help='port of server')
    # parser.add_argument('--max_users', type=int, default=5)
    # parser.add_argument('--llm_exec_nums', type=int, default=1)
    # parser.add_argument('--timeout', type=int, default=600)
    # parser.add_argument('--use_standalone_vad', action='store_true', help='Initiate the standalone VAD thread. If not set, relies on external VAD input.')

    args = parser.parse_args()

    # Load config from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # # Command-line arguments override YAML settings
    # if args.model_path:
    #     # This assumes you will add model_path to your YAML, which is a good practice.
    #     # For now, we add it to the config dictionary if provided via command line.
    #     config['model_path'] = args.model_path
    # if args.llm_path:
    #     config['llm_path'] = args.llm_path

    print("Configuration loaded:", json.dumps(config, indent=2))
    return config


def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

# init parms
configs = get_args()
# Assumed audio sampling rate
SAMPLING_RATE = configs['audio']['expected_sampling_rate']
# max users to connect
MAX_USERS = configs['connection']['max_users']
# number of inference pipelines to use
PIPELINE_NUMS = configs['connection']['llm_exec_nums']
# timeout to each user
TIMEOUT = configs['connection']['session_timeout']

# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print

# init inference pipelines pool (no TTS needed)
pipeline_pool = pipelineObjectPool(size=PIPELINE_NUMS, configs=configs)

# init flask app
app = Flask(__name__, template_folder='../web/resources')
socketio = SocketIO(app)
# init connected users
connected_users = {}

class DialogStateParams:
    """Simplified version focusing only on dialog state prediction without VAD"""
    def __init__(self, pipeline_pool, server_configs):
        try:
            self.server_configs = server_configs
            self.pipeline_pool = pipeline_pool
            self.pipeline_obj = self.pipeline_pool.acquire()
            if self.pipeline_obj is None:
                raise Exception("Failed to get pipeline object from pool")
                
            # init default prompt
            init_outputs = self.pipeline_obj.pipeline_proc.speech_dialogue(None, stat='pre', 
                                                                        role=server_configs['inference_control']['default_prompt'])
            self.system_role = deepcopy(init_outputs)

            
            # Dialog state prediction context
            self.generate_outputs = None
            self.dialog_state_callback = None
            
            # Timeout tracking
            self.last_activity_time = time.time()


            self.feature_gater = AudioFeatureGating(
                sample_rate=server_configs['audio']['expected_sampling_rate'],
                cache_history_size=server_configs['audio_feature_gating']['feature_gating_history_size'],
                onset_input_chunk_cache_size=server_configs['audio_feature_gating']['onset_input_chunk_cache_size'],
                fbank_config=server_configs['audio_feature_gating']['fbank']
            )
            if server_configs['vad']['use_standalone_vad']:
                # Standalone VAD component
                self.standalone_vad = PureVAD(
                    min_silent_duration_second=server_configs['vad']['min_silent_duration_second'],
                    speech_pad_second=server_configs['vad']['speech_pad_second'],
                    vad_threshold=server_configs['vad']['vad_threshold'],
                    cache_history_size=server_configs['vad']['vad_history_cache_chunk_cnt'],
                    audio_chunk_size=self.feature_gater.expected_frames_per_audio_chunk,
                )
                
            # Control flags
            self.stop_all_threads = False
            

            # Thread references
            self.pcm_thread = None
            self.vad_thread = None
            self.timeout_thread = None

            # Initialize context with system role
            self.reset_context()
            
        except Exception as e:
            print(f"Error initializing DialogStateParams: {e}")
            if hasattr(self, 'pipeline_obj') and self.pipeline_obj:
                self.pipeline_obj.release()
            raise
    
    def reset_context(self):
        """Reset the conversation context"""
        try:

            # Audio queues
            self.raw_pcm_queue = PCMQueue()  # Raw audio chunk annotated with IPU status
            self.annotated_audio_queue = ProcPCMQueue()  # Annotated audio from VAD
            self.processed_pcm_queue = ProcPCMQueue()  # Gated audio features


            # Reset feature gater state
            if hasattr(self, 'feature_gater'):
                self.feature_gater.reset()

            if hasattr(self, 'standalone_vad'):
                self.standalone_vad.reset()
                
            self.generate_outputs = deepcopy(self.system_role)#Initially or upon reset, only contains a system prompt

        except Exception as e:
            print(f"Error resetting context: {e}")
            raise
    
    def set_dialog_callback(self, callback):
        """Set callback function to be called when dialog_ss is predicted"""
        self.dialog_state_callback = callback
    
    def set_prompt(self, prompt):
        """Set system prompt and reset context"""
        self.system_role = self.pipeline_obj.pipeline_proc.speech_dialogue(None, stat='pre', role=prompt)

    def release(self):
        """Release resources"""
        try:
            self.stop_all_threads = True
            if self.pipeline_obj:
                self.pipeline_pool.release(self.pipeline_obj)
        except Exception as e:
            print(f"Error releasing resources: {e}")

def standalone_vad_thread(sid):
    """
    Standalone VAD thread that consumes raw audio, annotates it, and puts it
    into the annotated_audio_queue.
    """
    user = connected_users[sid]
    chunk_size = user.standalone_vad.get_chunk_size()
    print(f"Sid: {sid} Starting standalone VAD thread with chunk size: {chunk_size}")
    
    while not user.stop_all_threads:

        time.sleep(0.01)

        audio_chunk = user.raw_pcm_queue.get(chunk_size)
        if audio_chunk is None:
            continue

        # print(f"Sid: {sid} Received raw audio chunk of size: {len(audio_chunk)}")
            
        # Run VAD prediction to get annotated audio
        annotated_audio = user.standalone_vad.predict(audio_chunk)
        
        # Emit VAD events for monitoring GUI
        status = annotated_audio['status']
        if status == 'ipu_sl':
            emit_vad_event(sid, status)
            emit_state_update(sid, vad_state=True)
        elif status == 'ipu_el':
            emit_vad_event(sid, status)
            emit_state_update(sid, vad_state=False)
        elif status == 'ipu_cl':
            emit_state_update(sid, vad_state=True)

        # Put annotated audio into the queue for the feature gating thread
        user.annotated_audio_queue.put(annotated_audio)

    print(f"Sid: {sid} Stopping standalone VAD thread")

def feature_gating_thread(sid):
    """
    This thread gets annotated audio, computes fbank features for all of it
    to maintain state, and gates the features for IPU-related chunks to the
    main processing thread.
    """
    user = connected_users[sid]
    print(f"Sid: {sid} Starting feature gating thread.")

    while not user.stop_all_threads:

        time.sleep(0.01)
        
        # Get annotated audio chunk
        annotated_audio = user.annotated_audio_queue.get()
        

        if(annotated_audio is None):
            continue

        # Update last activity time upon receiving any annotated audio
        user.last_activity_time = time.time()

        # print(f"Sid: {sid} Received annotated audio chunk with status: {annotated_audio['status']}, size: {len(annotated_audio['audio'])}")


        # Process chunk to extract and gate features
        gated_feature_data = user.feature_gater.process_and_gate(annotated_audio)
        
        # If the feature gater returns data, put it in the processed queue for the LLM to process.
        if gated_feature_data:
            
            # print(f"Sid: {sid} Approved annotated audio chunk with status: {annotated_audio['status']}, size: {len(annotated_audio['audio'])}")

            # For speech start, the original logic sends cached chunks separately.
            if gated_feature_data['status'] == 'ipu_sl':
                # The 'feature_last_chunk' contains the features of chunks not belonging to the current IPU
                # We need to send them one by one.
                for i, feature in enumerate(gated_feature_data['feature_last_chunk']):
                    feature_item = {
                        'feature': feature,
                        'status': 'ipu_sl' if i == 0 else 'ipu_cl'##Treat the first input chunk in the history as the start of the IPU, despite the fact that the IPU is detected at the current chunk.
                    }
                    user.processed_pcm_queue.put(feature_item)
                
                # The current chunk becomes 'ipu_cl' as it follows the cached start.
                feature_item = {
                    'feature': gated_feature_data['feature'],
                    'status': 'ipu_cl' if len(gated_feature_data['feature_last_chunk']) > 0 else 'ipu_sl'##Treat the current chunk as the continuation of the IPU, although the VAD only detects it here, if we are sending context input chunks
                }
                user.processed_pcm_queue.put(feature_item)
            else:
                # For 'ipu_cl' and 'ipu_el', just forward the data.
                user.processed_pcm_queue.put(gated_feature_data)


    print(f"Sid: {sid} Stopping feature gating thread.")

def predict_dialog_state(sid):
    """
    Main loop for processing gated audio and predicting dialog states.
    Uses VAD-provided labels to determine IPU boundaries.
    
    Parameters:
    - sid (str): Session ID
    """
    print(f"Sid: {sid} Starting dialog state prediction thread")
    
    # Local outputs variable for current IPU processing
    outputs = None
    
    while True:

        time.sleep(0.01)
        
        if connected_users[sid].stop_all_threads:
            print(f"Sid: {sid} Stopping dialog state prediction thread")
            break
                
        # Get processed audio from the gated queue
        feature_data = connected_users[sid].processed_pcm_queue.get()  # Get one item, i.e., one chunk
        if feature_data is None:
            continue
            
        print(f"Sid: {sid} Processing approved audio for dialog state prediction, status: {feature_data['status']}")
        
        # Handle IPU start - create fresh context snapshot
        if feature_data['status'] == 'ipu_sl':
            print(f"Sid: {sid} IPU start detected - creating context snapshot")
            
            # Create snapshot of current context from shared memory (just like original server)
            outputs = deepcopy(connected_users[sid].generate_outputs)
            outputs['adapter_cache'] = None
            outputs['encoder_cache'] = None
            outputs['pe_index'] = 0
            outputs['stat'] = 'dialog_sl'
            outputs['last_id'] = None
            
            # Clean up any existing text/hidden state
            if 'text' in outputs:
                del outputs['text']
            if 'hidden_state' in outputs:
                del outputs['hidden_state']
            
            # Process the first chunk
            outputs = llm_prefill(feature_data, outputs, sid, is_first_pack=True)
        
        # Handle continuing or ending chunks
        elif feature_data['status'] in ['ipu_cl', 'ipu_el']:
            if outputs is not None:  # Only process if we have an active IPU context
                outputs = llm_prefill(feature_data, outputs, sid)
            else:
                print(f"Sid: {sid} Warning: Received {feature_data['status']} without active IPU context")

def timeout_monitor_thread(sid, timeout_seconds):
    """
    Monitors user activity and disconnects if the timeout is exceeded.
    """
    user = connected_users.get(sid)
    if not user:
        return

    print(f"Sid: {sid} Starting timeout monitor with a {timeout_seconds}s timeout.")
    while not user.stop_all_threads:
        time.sleep(5) # Check every 5 seconds
        if time.time() - user.last_activity_time > timeout_seconds:
            print(f"Sid: {sid} has been inactive for more than {timeout_seconds} seconds. Disconnecting.")
            disconnect_user(sid)
            break
    print(f"Sid: {sid} Stopping timeout monitor thread.")


def llm_prefill(data, outputs, sid, is_first_pack=False):
    """
    Simplified LLM prefill for dialog state prediction only.
    Uses VAD-provided status labels to handle different chunk types.
    
    Parameters:
    - data (dict): Audio chunk data with features and VAD status
    - outputs (dict): Current conversation context
    - sid (str): Session ID
    - is_first_pack (bool): Whether this is the first pack in an IPU
    
    Returns:
    - outputs (dict): Updated conversation context
    """
    
    print(f"Sid: {sid} Processing audio chunk with status: {data['status']}")
    
    # Handle different VAD statuses
    if data['status'] == 'ipu_sl':
        # Stage1: start listen
        print(f"Sid: {sid} Start listening to new IPU.")
        outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **outputs)
    
    elif data['status'] == 'ipu_el':
        # VAD detected speech end - process final chunk
        print(f"Sid: {sid} VAD end detected - processing final chunk.")
        outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **outputs)
    
    elif data['status'] == 'ipu_cl':
        # Continue listening
        print(f"Sid: {sid} within IPU - processing chunks.")
        outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
                torch.tensor(data['feature']), **outputs)
        
        # Force dialog_cl state for first pack (like original server)
        if is_first_pack:
            outputs['stat'] = 'dialog_cl'
    
    # Emit state update
    emit_state_update(sid, dialog_state=outputs.get('stat'))
    
    # Handle state transitions (no VAD state changes needed)
    if outputs['stat'] == 'dialog_cl':
        print(f"Sid: {sid} Dialog state: continue listening")
        
    elif outputs['stat'] == 'dialog_el':
        print(f"Sid: {sid} Dialog state: end listening (no response)")
        # Update shared memory to preserve this interaction
        connected_users[sid].generate_outputs = deepcopy(outputs)
        
    elif outputs['stat'] == 'dialog_ss':
        print(f"Sid: {sid} Dialog state: start preparing response")
        # Update shared memory with current context
        connected_users[sid].generate_outputs = deepcopy(outputs)
        
        # Trigger external callback if set
        if connected_users[sid].dialog_state_callback:
            connected_users[sid].dialog_state_callback(sid, deepcopy(outputs))
    
        # Reset to the sl state -- otherwise the LLM will start to decode utterances.
        outputs['stat'] = 'dialog_sl'

    return outputs

def disconnect_user(sid):
    """Disconnect user and cleanup resources"""
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to timeout")
        socketio.emit('out_time', to=sid)
        connected_users[sid].release()
        time.sleep(3)
        del connected_users[sid]

def dialog_ss_callback(sid, context):
    """
    Callback function called when dialog_ss state is predicted.
    
    Parameters:
    - sid (str): Session ID
    - context (dict): Current conversation context
    """
    print(f"Dialog SS callback triggered for user {sid}")
    print(f"Context keys: {context.keys()}")
    
    # Prepare context info for the GUI
    context_info = "External LLM integration point"
    if 'past_tokens' in context and context['past_tokens']:
        tokenizer = connected_users[sid].pipeline_obj.pipeline_proc.model.tokenizer
        conversation_text = tokenizer.decode(context['past_tokens'], skip_special_tokens=True)
        print(f"Conversation so far: {conversation_text}")
        context_info = f"Context length: {len(context['past_tokens'])} tokens"
    
    # Emit dialog_ss callback event to GUI
    emit_dialog_ss_callback(sid, context_info)
    
    # TODO: Integrate with your external LLM here
    # external_llm_response = your_external_llm.generate(context)
    
    # After external LLM finishes, you would update the context:
    # connected_users[sid].generate_outputs = updated_context_from_external_llm

def emit_state_update(sid, vad_state=None, dialog_state=None, generating=None):
    """Emit state updates to the GUI"""
    if sid in connected_users:
        state_data = {}
        if vad_state is not None:
            state_data['vad_state'] = vad_state
        if dialog_state is not None:
            state_data['dialog_state'] = dialog_state
        if generating is not None:
            state_data['generating'] = generating
        
        if state_data:
            socketio.emit('state_update', state_data, to=sid)

def emit_vad_event(sid, event_type):
    """Emit VAD events to the GUI"""
    if sid in connected_users:
        socketio.emit('ipu_event', {'event_type': event_type}, to=sid)

def emit_dialog_ss_callback(sid, context_info):
    """Emit dialog_ss callback events to the GUI"""
    if sid in connected_users:
        socketio.emit('dialog_ss_callback', {'context_info': context_info}, to=sid)

# Routes
@app.route('/')
def index():
    try:
        return render_template('dialog_state_monitoring.html')
    except Exception as e:
        print(f'Error serving index page: {e}')
        return f'Error loading page: {e}', 500

@app.route('/monitor')
def monitor():
    try:
        return render_template('dialog_state_monitoring.html')
    except Exception as e:
        print(f'Error serving monitor page: {e}')
        return f'Error loading page: {e}', 500

@socketio.on('connect')
def handle_connect():
    try:
        if len(connected_users) >= MAX_USERS:
            print('Too many users connected, disconnecting new user')
            emit('too_many_users')
            return False
        
        sid = request.sid
        print(f'Attempting to connect user {sid}')
        
        # Initialize user parameters
        try:
            connected_users[sid] = DialogStateParams(pipeline_pool, configs)
            print(f'User {sid} parameters initialized')
        except Exception as e:
            print(f'Failed to initialize parameters for user {sid}: {e}')
            emit('connection_failed', {'error': str(e)})
            return False
        
        
        # Set callback for dialog_ss events
        connected_users[sid].set_dialog_callback(dialog_ss_callback)
        
        # Send initial state
        emit_state_update(sid, vad_state=False, dialog_state='dialog_sl', generating=False)
        

        # Start timeout monitor thread
        timeout_thread = threading.Thread(target=timeout_monitor_thread, args=(sid, TIMEOUT))
        timeout_thread.start()
        connected_users[sid].timeout_thread = timeout_thread




        # Start standalone VAD thread if enabled
        if configs['vad']['use_standalone_vad']:
            print(f"User {sid}: Standalone VAD enabled, starting VAD thread.")
            vad_thread = threading.Thread(target=standalone_vad_thread, args=(sid,))
            vad_thread.start()
            connected_users[sid].vad_thread = vad_thread
        else:
            print(f"User {sid}: Standalone VAD disabled, relying on external VAD input via /annotated_audio endpoint.")
        
        
        # Start feature gating thread
        feature_gating_thread_obj = threading.Thread(target=feature_gating_thread, args=(sid,))
        feature_gating_thread_obj.start()
        connected_users[sid].feature_gating_thread = feature_gating_thread_obj


        # Start dialog state prediction thread
        pcm_thread = threading.Thread(target=predict_dialog_state, args=(sid,))
        pcm_thread.start()
        connected_users[sid].pcm_thread = pcm_thread
        
        pipeline_pool.print_info()
        print(f'User {sid} connected successfully')
        
        

        return True
        
    except Exception as e:
        print(f'Error in handle_connect for {request.sid}: {e}')
        if request.sid in connected_users:
            connected_users[request.sid].release()
            del connected_users[request.sid]
        emit('connection_failed', {'error': str(e)})
        return False

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    try:
        if sid in connected_users:
            print(f'Disconnecting user {sid}')
            

            # Stop all threads
            connected_users[sid].stop_all_threads = True
            
            # Wait for threads to finish
            if hasattr(connected_users[sid], 'pcm_thread'):
                connected_users[sid].pcm_thread.join(timeout=2.0)
            if hasattr(connected_users[sid], 'feature_gating_thread'):
                connected_users[sid].feature_gating_thread.join(timeout=2.0)
            if hasattr(connected_users[sid], 'vad_thread'):
                connected_users[sid].vad_thread.join(timeout=2.0)
            if hasattr(connected_users[sid], 'timeout_thread'):
                connected_users[sid].timeout_thread.join(timeout=2.0)
            
            # Release resources
            connected_users[sid].release()

            del connected_users[sid]
            
        pipeline_pool.print_info()
        print(f'User {sid} disconnected successfully')
        
    except Exception as e:
        print(f'Error in handle_disconnect for {sid}: {e}')

@socketio.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in connected_users:
        # Reset VAD, feature gater, and context
        connected_users[sid].reset_context()
        emit_state_update(sid, vad_state=False, dialog_state='dialog_sl')
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        pass
    else:
        disconnect()
    print('Recording stopped')

@socketio.on('prompt_text')
def handle_prompt_text(text):
    sid = request.sid
    if sid in connected_users:
        connected_users[sid].set_prompt(text)
        print(f"Sid: {sid} Prompt set as: {text}")
        socketio.emit('prompt_success', to=sid)
    else:
        disconnect()

@socketio.on('audio')
def handle_audio(data):
    sid = request.sid
    if sid in connected_users:
        
        data = json.loads(data)

        if(data['sr'] != SAMPLING_RATE):
            raise ValueError(f"Expected audio sampling rate {SAMPLING_RATE}, but got {data['sr']}")

        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        
        # Put raw audio into the raw queue for VAD processing
        connected_users[sid].raw_pcm_queue.put(audio_data.astype(np.float32) / 32768.0)
    else:
        disconnect()

@socketio.on('annotated_audio')
def handle_annotated_audio(data):
    """
    New endpoint for receiving audio chunks already annotated with VAD status.
    Expected data format: {'audio': bytes, 'status': 'ipu_cl' | 'ipu_sl' | 'ipu_el' | null}
    """
    sid = request.sid
    if sid in connected_users:

        data = json.loads(data)
        
        if(data['sr'] != SAMPLING_RATE):
            raise ValueError(f"Expected audio sampling rate {SAMPLING_RATE}, but got {data['sr']}")
        
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        
        ## Normalize audio data to float32 range. Note that if standalone VAD is used, it will already be normalized and that's why there we directly put the audio data into the annotated audio queue
        annotated_chunk = {
            "audio": audio_data.astype(np.float32) / 32768.0,
            "status": data.get('status')
        }
        
        # Put annotated audio into the queue for the feature gating thread
        connected_users[sid].annotated_audio_queue.put(annotated_chunk)
    else:
        disconnect()


if __name__ == "__main__":
    print("Starting Freeze-Omni Dialog State Server")
    cert_file = "web/resources/cert.pem"
    key_file = "web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    socketio.run(app, host=configs['connection']['ip'], port=int(configs['connection']['port']), ssl_context=(cert_file, key_file))