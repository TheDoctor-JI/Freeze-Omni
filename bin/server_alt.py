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
            _, init_key_values, _, _, _ = self.pipeline_obj.pipeline_proc.speech_dialogue(None, identity = '', status='pre', 
                                                                        role=server_configs['inference_control']['default_prompt'])
            self.system_role = deepcopy(init_key_values)

            
            # Dialog state prediction context
            self.caches = {} # Holds caches for 'user' and 'system'
            self.past_key_values = None
            self.dialog_state_callback = None
            
            # Timeout tracking
            self.last_activity_time = time.time()


            self.feature_gater = {
                'user': AudioFeatureGating(
                    sample_rate=server_configs['audio']['expected_sampling_rate'],
                    cache_history_size=server_configs['audio_feature_gating']['feature_gating_history_size'],
                    onset_input_chunk_cache_size=server_configs['audio_feature_gating']['onset_input_chunk_cache_size'],
                    fbank_config=server_configs['audio_feature_gating']['fbank']
                ),
                'system': AudioFeatureGating(
                    sample_rate=server_configs['audio']['expected_sampling_rate'],
                    cache_history_size=server_configs['audio_feature_gating']['feature_gating_history_size'],
                    onset_input_chunk_cache_size=server_configs['audio_feature_gating']['onset_input_chunk_cache_size'],
                    fbank_config=server_configs['audio_feature_gating']['fbank']
                )
            }

            if server_configs['vad']['use_standalone_vad']:
                self.standalone_vad = {
                    'user': PureVAD(
                        min_silent_duration_second=server_configs['vad']['min_silent_duration_second'],
                        speech_pad_second=server_configs['vad']['speech_pad_second'],
                        vad_threshold=server_configs['vad']['vad_threshold'],
                        cache_history_size=server_configs['vad']['vad_history_cache_chunk_cnt'],
                        audio_chunk_size=self.feature_gater['user'].expected_frames_per_audio_chunk,
                    ),
                    'system': PureVAD(
                        min_silent_duration_second=server_configs['vad']['min_silent_duration_second'],
                        speech_pad_second=server_configs['vad']['speech_pad_second'],
                        vad_threshold=server_configs['vad']['vad_threshold'],
                        cache_history_size=server_configs['vad']['vad_history_cache_chunk_cnt'],
                        audio_chunk_size=self.feature_gater['system'].expected_frames_per_audio_chunk,
                    )
                }


            # Control flags
            self.stop_all_threads = False
            

            # Thread references
            self.pcm_thread = None
            self.vad_threads = {}
            self.feature_gating_threads = {}
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

            # Audio queues for user and system separately
            self.raw_pcm_queue = {
                'user': PCMQueue(),
                'system': PCMQueue()
            }
            self.annotated_audio_queue = {
                'user': ProcPCMQueue(),
                'system': ProcPCMQueue()
            }
            ## Main queue for dialog state predicting, input is sequentialized
            self.processed_pcm_queue = ProcPCMQueue()


            # Reset feature gater state for both
            if hasattr(self, 'feature_gater'):
                self.feature_gater['user'].reset()
                self.feature_gater['system'].reset()

            # Reset VAD state for both
            if hasattr(self, 'standalone_vad'):
                self.standalone_vad['user'].reset()
                self.standalone_vad['system'].reset()
                

            #Initially or upon reset, only contains a system prompt
            self.past_key_values = deepcopy(self.system_role)

            # Reset caches for both user and system
            self.caches = {
                'user': {
                    'encoder_cache': None,
                    'adapter_cache': None,
                    'pe_index': 0,
                },
                'system': {
                    'encoder_cache': None,
                    'adapter_cache': None,
                    'pe_index': 0,
                }
            }

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

def standalone_vad_thread(sid, identity):
    """
    Standalone VAD thread that consumes raw audio for a specific identity,
    annotates it, and puts it into the corresponding annotated_audio_queue.
    """
    user = connected_users[sid]
    chunk_size = user.standalone_vad[identity].get_chunk_size()
    print(f"Sid: {sid} Starting standalone VAD thread for '{identity}' with chunk size: {chunk_size}")
    
    while not user.stop_all_threads:

        time.sleep(0.01)

        audio_chunk = user.raw_pcm_queue[identity].get(chunk_size)
        if audio_chunk is None:
            continue

        # print(f"Sid: {sid} Received raw audio chunk of size: {len(audio_chunk)}")
            
        # Run VAD prediction to get annotated audio
        annotated_audio = user.standalone_vad[identity].predict(audio_chunk)
        
        # Emit VAD events for monitoring GUI (only for user)
        status = annotated_audio['status']
        if status == 'ipu_sl':
            emit_vad_event(sid, status, identity=identity)
            emit_state_update(sid, vad_state=True, identity=identity)
        elif status == 'ipu_el':
            emit_vad_event(sid, status, identity=identity)
            emit_state_update(sid, vad_state=False,  identity=identity)
        elif status == 'ipu_cl':
            emit_state_update(sid, vad_state=True,  identity=identity)

        # Put annotated audio into the queue for the feature gating thread
        user.annotated_audio_queue[identity].put(annotated_audio)

    print(f"Sid: {sid} Stopping standalone VAD thread for '{identity}'")

def feature_gating_thread(sid, identity):
    """
    This thread gets annotated audio for a specific identity, computes fbank features,
    and gates the features to the shared processing thread.
    """
    user = connected_users[sid]
    print(f"Sid: {sid} Starting feature gating thread for '{identity}'.")

    while not user.stop_all_threads:

        time.sleep(0.01)
        
        # Get annotated audio chunk for the specific identity
        annotated_audio = user.annotated_audio_queue[identity].get()
        

        if(annotated_audio is None):
            continue

        # Update last activity time upon receiving any annotated audio
        user.last_activity_time = time.time()

        # print(f"Sid: {sid} Received annotated audio chunk with status: {annotated_audio['status']}, size: {len(annotated_audio['audio'])}")


        # Process chunk to extract and gate features
        gated_feature_data = user.feature_gater[identity].process_and_gate(annotated_audio)
        
        # If the feature gater returns data, put it in the processed queue for the LLM to process.
        if gated_feature_data:
            # Set the identity for this chunk
            gated_feature_data['identity'] = identity
            
            if gated_feature_data['status'] == 'ipu_sl':
                for i, feature in enumerate(gated_feature_data['feature_last_chunk']):
                    feature_item = {
                        'identity': identity,
                        'feature': feature,
                        'status': 'ipu_sl' if i == 0 else 'ipu_cl'
                    }
                    user.processed_pcm_queue.put(feature_item)
                
                feature_item = {
                    'identity': identity,
                    'feature': gated_feature_data['feature'],
                    'status': 'ipu_cl' if len(gated_feature_data['feature_last_chunk']) > 0 else 'ipu_sl'
                }
                user.processed_pcm_queue.put(feature_item)
            else:
                user.processed_pcm_queue.put(gated_feature_data)


    print(f"Sid: {sid} Stopping feature gating thread for '{identity}'.")

def predict_dialog_state(sid):
    """
    Main loop for processing gated audio and predicting dialog states.
    Uses VAD-provided labels to determine IPU boundaries.
    
    Parameters:
    - sid (str): Session ID
    """
    print(f"Sid: {sid} Starting dialog state prediction thread")
        
    while True:

        time.sleep(0.01)
        
        if connected_users[sid].stop_all_threads:
            print(f"Sid: {sid} Stopping dialog state prediction thread")
            break
                
        # Get processed audio from the gated queue
        feature_data = connected_users[sid].processed_pcm_queue.get()  # Get one item, i.e., one chunk
        if feature_data is None:
            continue
            
        print(f"Sid: {sid} Processing approved audio for dialog state prediction, status: {feature_data['status']}, identity: {feature_data.get('identity', 'N/A')}")
        
        # Always run forward processing
        llm_prefill(feature_data, sid)

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

def llm_prefill(data, sid):
    """
    Processes an audio chunk to update the shared conversational context.
    If the chunk is from the user, it also predicts dialog state probabilities.
    
    Parameters:
    - data (dict): Audio chunk data with features, status, and identity.
    - sid (str): Session ID.
    """
    
    user = connected_users[sid]
    identity = data['identity']
    
    
    # print(f"Sid: {sid} Processing audio chunk with status: {data['status']} from {identity}")
    

    # 1. Assemble the context for this processing step
    outputs = {
        'past_key_values': user.past_key_values, # Shared conversational history
        'identity': identity,
        'status': data['status'],
        **user.caches[identity] # Identity-specific caches (encoder, adapter, pe_index)
    }



    # 2. Call the pipeline
    prediction_probs, past_key_values, cnn_cache, buffer, pe_index = user.pipeline_obj.pipeline_proc.speech_dialogue(
        torch.tensor(data['feature']), **outputs
    )


    # 3. Update the shared and identity-specific contexts
    user.past_key_values = past_key_values # Update shared history
    user.caches[identity] = { # Update identity-specific caches
        'encoder_cache': buffer,
        'adapter_cache': cnn_cache,
        'pe_index': pe_index
    }
    

    # 4. Handle prediction results if the input was from the user
    if identity == 'user' and prediction_probs is not None:

        threshold = configs['dialog_state_decision']['threshold']

        # Check if the system should start generating a new response
        if prediction_probs["state_1"] > threshold:
            emit_state_update(sid, dialog_state='dialog_ss')
            # print(f"Sid: {sid} Dialog state: start preparing response")
            
            # Trigger external callback if set
            if user.dialog_state_callback:
                user.dialog_state_callback(sid)
        elif prediction_probs["state_2"] > threshold:
            emit_state_update(sid, dialog_state='dialog_el')
            # print(f"Sid: {sid} Dialog state: continue listening")
        else:
            emit_state_update(sid, dialog_state='dialog_cl')

def disconnect_user(sid):
    """Disconnect user and cleanup resources"""
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to timeout")
        socketio.emit('out_time', to=sid)
        connected_users[sid].release()
        time.sleep(3)
        del connected_users[sid]

def dialog_ss_callback(sid):
    """
    Callback function called when dialog_ss state is predicted.
    
    Parameters:
    - sid (str): Session ID
    """
    # print(f"Dialog SS callback triggered for user {sid}")

    # Emit dialog_ss callback event to GUI
    context = ''
    emit_dialog_ss_callback(sid, context)
    
def emit_state_update(sid, vad_state=None, dialog_state=None, generating=None, identity = None):
    """Emit state updates to the GUI"""
    if sid in connected_users:
        state_data = {}
        if vad_state is not None:
            state_data['vad_state'] = vad_state
            state_data['identity'] = identity 
        if dialog_state is not None:
            state_data['dialog_state'] = dialog_state
        if generating is not None:
            state_data['generating'] = generating
        if state_data:
            socketio.emit('state_update', state_data, to=sid)

def emit_vad_event(sid, event_type, identity = None):
    """Emit VAD events to the GUI"""
    if sid in connected_users:
        socketio.emit('ipu_event', {'event_type': event_type, 'identity': identity}, to=sid)

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
        


        # Start timeout monitor thread
        timeout_thread = threading.Thread(target=timeout_monitor_thread, args=(sid, TIMEOUT))
        timeout_thread.start()
        connected_users[sid].timeout_thread = timeout_thread


        # Start threads for both 'user' and 'system' identities
        for identity in ['user', 'system']:
            # Start standalone VAD thread if enabled
            if configs['vad']['use_standalone_vad']:
                print(f"User {sid}: Standalone VAD enabled, starting VAD thread for '{identity}'.")
                vad_thread = threading.Thread(target=standalone_vad_thread, args=(sid, identity))
                vad_thread.start()
                connected_users[sid].vad_threads[identity] = vad_thread
            else:
                print(f"User {sid}: Standalone VAD disabled for '{identity}', relying on external VAD input.")
            
            # Start feature gating thread
            feature_gating_thread_obj = threading.Thread(target=feature_gating_thread, args=(sid, identity))
            feature_gating_thread_obj.start()
            connected_users[sid].feature_gating_threads[identity] = feature_gating_thread_obj


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
            
            for identity in ['user', 'system']:
                if identity in connected_users[sid].feature_gating_threads:
                    connected_users[sid].feature_gating_threads[identity].join(timeout=2.0)
                if identity in connected_users[sid].vad_threads:
                    connected_users[sid].vad_threads[identity].join(timeout=2.0)

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

@socketio.on('user_raw_audio')
def handle_user_raw_audio(data):
    sid = request.sid
    if sid in connected_users:
        data = json.loads(data)
        if(data['sr'] != SAMPLING_RATE):
            raise ValueError(f"Expected audio sampling rate {SAMPLING_RATE}, but got {data['sr']}")
        if(data['enc'] != 's16le'):
            raise ValueError(f"Expected audio encoding 's16le', but got {data['enc']}")
        
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        connected_users[sid].raw_pcm_queue['user'].put(audio_data.astype(np.float32) / 32768.0)
    else:
        disconnect()


@socketio.on('system_raw_audio')
def handle_system_raw_audio(data):
    sid = request.sid
    if sid in connected_users:
        data = json.loads(data)
        if(data['sr'] != SAMPLING_RATE):
            raise ValueError(f"Expected audio sampling rate {SAMPLING_RATE}, but got {data['sr']}")
        if(data['enc'] != 's16le'):
            raise ValueError(f"Expected audio encoding 's16le', but got {data['enc']}")
        
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        connected_users[sid].raw_pcm_queue['system'].put(audio_data.astype(np.float32) / 32768.0)
    else:
        disconnect()

@socketio.on('user_annotated_audio')
def handle_user_annotated_audio(data):
    """
    Endpoint for receiving user audio chunks already annotated with VAD status.
    """
    sid = request.sid
    if sid in connected_users:
        data = json.loads(data)
        if(data['sr'] != SAMPLING_RATE):
            raise ValueError(f"Expected audio sampling rate {SAMPLING_RATE}, but got {data['sr']}")
        if(data['enc'] != 's16le'):
            raise ValueError(f"Expected audio encoding 's16le', but got {data['enc']}")

        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        annotated_chunk = {
            "audio": audio_data.astype(np.float32) / 32768.0,
            "status": data.get('status')
        }
        
        connected_users[sid].annotated_audio_queue['user'].put(annotated_chunk)
    else:
        disconnect()


@socketio.on('system_annotated_audio')
def handle_system_annotated_audio(data):
    """
    Endpoint for receiving system audio chunks already annotated with VAD status.
    """
    sid = request.sid
    if sid in connected_users:
        data = json.loads(data)
        if(data['sr'] != SAMPLING_RATE):
            raise ValueError(f"Expected audio sampling rate {SAMPLING_RATE}, but got {data['sr']}")
        if(data['enc'] != 's16le'):
            raise ValueError(f"Expected audio encoding 's16le', but got {data['enc']}")

        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        annotated_chunk = {
            "audio": audio_data.astype(np.float32) / 32768.0,
            "status": data.get('status')
        }
        
        connected_users[sid].annotated_audio_queue['system'].put(annotated_chunk)
    else:
        disconnect()


if __name__ == "__main__":
    print("Starting Freeze-Omni Dialog State Server")
    cert_file = "web/resources/cert.pem"
    key_file = "web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    socketio.run(app, host=configs['connection']['ip'], port=int(configs['connection']['port']), ssl_context=(cert_file, key_file))