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

from web.parms import GlobalParams
from web.pool import pipelineObjectPool
from web.pem import generate_self_signed_cert
from web.queue import PCMQueue, ProcPCMQueue, ThreadSafeQueue

def get_args():
    parser = argparse.ArgumentParser(description='Freeze-Omni Dialog State Server')
    parser.add_argument('--model_path', required=True, help='model_path to load')
    parser.add_argument('--llm_path', required=True, help='llm_path to load')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--ip', required=True, help='ip of server')
    parser.add_argument('--port', required=True, help='port of server')
    parser.add_argument('--max_users', type=int, default=5)
    parser.add_argument('--llm_exec_nums', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--use_standalone_vad', action='store_true', help='Initiate the standalone VAD thread. If not set, relies on external VAD input.')
    args = parser.parse_args()
    print(args)
    return args

def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)

# init parms
configs = get_args()
# max users to connect
MAX_USERS = configs.max_users
# number of inference pipelines to use
PIPELINE_NUMS = configs.llm_exec_nums
# timeout to each user
TIMEOUT = configs.timeout

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

class StandaloneVAD:
    """Standalone VAD component that runs independently"""
    def __init__(self):
        from web.vad import VAD
        self.vad = VAD(cache_history = 10, silence_threshold_ms= 300)
        self.stop_vad = False
        
    def predict(self, audio):
        """Predict VAD status and return whether audio should be processed"""
        res = self.vad.predict(audio)
        
        # Handle VAD state transitions independently
        if res['status'] == 'ipu_sl':
            print("Standalone VAD: Speech start detected")
            return True, res  # Send audio to dialog state prediction
            
        elif res['status'] == 'ipu_cl':
            # Continue processing while in dialog
            return True, res
            
        elif res['status'] == 'ipu_el':
            print("Standalone VAD: Speech end detected")
            self.vad.reset_vad()  
            return False, res  # Stop sending audio
            
        else:
            # No speech detected
            self.vad.reset_vad()  
            return False, res
    
    def reset(self):
        """Reset VAD state"""
        self.vad.reset_vad()
    
    def get_chunk_size(self):
        return self.vad.get_chunk_size()

class DialogStateParams:
    """Simplified version focusing only on dialog state prediction without VAD"""
    def __init__(self, pipeline_pool):
        try:
            self.pipeline_obj = pipeline_pool.acquire()
            if self.pipeline_obj is None:
                raise Exception("Failed to get pipeline object from pool")
                
            # Audio queues
            self.raw_pcm_queue = PCMQueue()  # Raw audio from client
            self.processed_pcm_queue = ProcPCMQueue()  # Audio gated by standalone VAD
            
            # Control flags
            self.stop_pcm = False
            self.stop_vad = False
            
            # Dialog state prediction context
            self.generate_outputs = None
            self.dialog_state_callback = None
            
            if configs.use_standalone_vad:
                # Standalone VAD component
                self.standalone_vad = StandaloneVAD()
                
            # Thread references
            self.timer = None
            self.pcm_thread = None
            self.vad_thread = None
            
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
            self.generate_outputs = {
                'past_key_values': None,
                'stat': 'pre',
                'last_id': None,
                'past_tokens': None,
                'adapter_cache': None,
                'encoder_cache': None,
                'pe_index': 0
            }
            # Set system role
            self.generate_outputs = self.pipeline_obj.pipeline_proc.speech_dialogue(None, **self.generate_outputs)
        except Exception as e:
            print(f"Error resetting context: {e}")
            raise
    
    def set_dialog_callback(self, callback):
        """Set callback function to be called when dialog_ss is predicted"""
        self.dialog_state_callback = callback
    
    def set_prompt(self, prompt):
        """Set system prompt and reset context"""
        try:
            self.generate_outputs = {
                'past_key_values': None,
                'stat': 'pre',
                'last_id': None,
                'past_tokens': None,
                'adapter_cache': None,
                'encoder_cache': None,
                'pe_index': 0,
                'role': prompt
            }
            self.generate_outputs = self.pipeline_obj.pipeline_proc.speech_dialogue(None, **self.generate_outputs)
        except Exception as e:
            print(f"Error setting prompt: {e}")
            raise
    
    def release(self):
        """Release resources"""
        try:
            self.stop_pcm = True
            self.stop_vad = True
            if self.pipeline_obj:
                self.pipeline_obj.release()
                self.pipeline_obj = None
        except Exception as e:
            print(f"Error releasing resources: {e}")


def standalone_vad_thread(sid):
    """
    Standalone VAD thread that gates audio from raw queue to processed queue
    
    Parameters:
    - sid (str): Session ID
    """
    chunk_size = connected_users[sid].standalone_vad.get_chunk_size()
    print(f"Sid: {sid} Starting standalone VAD thread with chunk size: {chunk_size}")
    
    while True:
        if connected_users[sid].stop_vad:
            print(f"Sid: {sid} Stopping standalone VAD thread")
            break
            
        time.sleep(0.01)
        
        # Get audio from raw queue
        audio_chunk = connected_users[sid].raw_pcm_queue.get(chunk_size)
        if audio_chunk is None:
            continue
            
        print(f"Sid: {sid} VAD processing audio chunk of size: {len(audio_chunk)}")
        
        # Run VAD prediction
        should_process, vad_result = connected_users[sid].standalone_vad.predict(audio_chunk)
        
        # Emit VAD events for monitoring gui
        if vad_result['status'] == 'ipu_sl':
            emit_vad_event(sid, vad_result['status'])
            emit_state_update(sid, vad_state=True)
        elif vad_result['status'] == 'ipu_el':
            emit_vad_event(sid, vad_result['status'])
            emit_state_update(sid, vad_state=False)
        elif vad_result['status'] == 'ipu_cl':
            emit_state_update(sid, vad_state=True)
        
        # Gate audio to dialog state prediction based on VAD decision
        if should_process:
            print(f"Sid: {sid} VAD approved audio for dialog state prediction")
            
            # For speech start, add cached chunks first with proper labels
            if vad_result['status'] == 'ipu_sl':
                for i, feature in enumerate(vad_result['feature_last_chunk']):
                    feature_data = {
                        'feature': feature,
                        'status': 'ipu_sl' if i == 0 else 'ipu_cl'  # First cached chunk is ipu_sl, rest are ipu_cl
                    }
                    connected_users[sid].processed_pcm_queue.put(feature_data)
                
                # Current chunk becomes ipu_cl since ipu_sl was assigned to first cached chunk
                feature_data = {
                    'feature': vad_result['feature'],
                    'status': 'ipu_cl'
                }
                connected_users[sid].processed_pcm_queue.put(feature_data)
            
            # For continuing or ending speech, pass through the VAD status
            else:
                feature_data = {
                    'feature': vad_result['feature'],
                    'status': vad_result['status']  # 'ipu_cl' or 'ipu_el'
                }
                connected_users[sid].processed_pcm_queue.put(feature_data)
        else:
            print(f"Sid: {sid} VAD blocked audio from dialog state prediction")


def process_pcm(sid):
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
        if connected_users[sid].stop_pcm:
            print(f"Sid: {sid} Stopping dialog state prediction thread")
            break
        
        time.sleep(0.01)
        
        # Get processed audio from the gated queue
        feature_data = connected_users[sid].processed_pcm_queue.get(1)  # Get one item, i.e., one chunk
        if feature_data is None:
            # Emit an indicator that no dialog state decision is made
            emit_state_update(sid, dialog_state='no_decision')
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
        print(f"Sid: {sid} Start listening to new IPU")
        outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **outputs)
    
    elif data['status'] == 'ipu_el':
        # VAD detected speech end - process final chunk
        print(f"Sid: {sid} VAD end detected - processing final chunk")
        outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **outputs)
    
    elif data['status'] == 'ipu_cl':
        # Continue listening
        print(f"Sid: {sid} within IPU - processing chunks")
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
    
        # Reset to the sl state
        outputs['stat'] = 'dialog_sl'

    return outputs


def disconnect_user(sid):
    """Disconnect user and cleanup resources"""
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to timeout")
        socketio.emit('out_time', to=sid)
        connected_users[sid].stop_pcm = True
        connected_users[sid].stop_vad = True
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
            connected_users[sid] = DialogStateParams(pipeline_pool)
            print(f'User {sid} parameters initialized')
        except Exception as e:
            print(f'Failed to initialize parameters for user {sid}: {e}')
            emit('connection_failed', {'error': str(e)})
            return False
        
        # Set up timeout timer
        timer = Timer(TIMEOUT, disconnect_user, [sid])
        timer.start()
        connected_users[sid].timer = timer
        
        # Set callback for dialog_ss events
        connected_users[sid].set_dialog_callback(dialog_ss_callback)
        
        # Send initial state
        emit_state_update(sid, vad_state=False, dialog_state='dialog_sl', generating=False)
        
        # Start standalone VAD thread if enabled
        if configs.use_standalone_vad:
            print(f"User {sid}: Standalone VAD enabled, starting VAD thread.")
            vad_thread = threading.Thread(target=standalone_vad_thread, args=(sid,))
            vad_thread.start()
            connected_users[sid].vad_thread = vad_thread
        else:
            print(f"User {sid}: Standalone VAD disabled, relying on external VAD input.")
        

        # Start dialog state prediction thread
        pcm_thread = threading.Thread(target=process_pcm, args=(sid,))
        pcm_thread.start()
        connected_users[sid].pcm_thread = pcm_thread
        
        pipeline_pool.print_info()
        print(f'User {sid} connected successfully')
        
        return True
        
    except Exception as e:
        print(f'Error in handle_connect for {request.sid}: {e}')
        if request.sid in connected_users:
            if hasattr(connected_users[request.sid], 'timer'):
                connected_users[request.sid].timer.cancel()
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
            
            # Cancel timer
            if hasattr(connected_users[sid], 'timer'):
                connected_users[sid].timer.cancel()
            
            # Stop all threads
            connected_users[sid].stop_pcm = True
            connected_users[sid].stop_vad = True
            
            # Wait for threads to finish
            if hasattr(connected_users[sid], 'pcm_thread'):
                connected_users[sid].pcm_thread.join(timeout=2.0)
            if hasattr(connected_users[sid], 'vad_thread'):
                connected_users[sid].vad_thread.join(timeout=2.0)
            
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
        if hasattr(connected_users[sid], 'timer'):
            connected_users[sid].timer.cancel()
        connected_users[sid].timer = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid].timer.start()
        
        # Reset both VAD and context
        connected_users[sid].standalone_vad.reset()
        connected_users[sid].reset_context()
        emit_state_update(sid, vad_state=False, dialog_state='dialog_sl')
    else:
        disconnect()
    print('Recording started')

@socketio.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in connected_users:
        if hasattr(connected_users[sid], 'timer'):
            connected_users[sid].timer.cancel()
        connected_users[sid].timer = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid].timer.start()
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
        if hasattr(connected_users[sid], 'timer'):
            connected_users[sid].timer.cancel()
        connected_users[sid].timer = Timer(TIMEOUT, disconnect_user, [sid])
        connected_users[sid].timer.start()
        
        data = json.loads(data)
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        
        # Put raw audio into the raw queue for VAD processing
        connected_users[sid].raw_pcm_queue.put(audio_data.astype(np.float32) / 32768.0)
    else:
        disconnect()

if __name__ == "__main__":
    print("Starting Freeze-Omni Dialog State Server")
    cert_file = "web/resources/cert.pem"
    key_file = "web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    socketio.run(app, host=configs.ip, port=configs.port, ssl_context=(cert_file, key_file))