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
from web.queue import PCMQueue, ThreadSafeQueue

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

class DialogStateParams:
    """Simplified version of GlobalParams focusing only on dialog state prediction"""
    def __init__(self, pipeline_pool):
        try:
            self.pipeline_obj = pipeline_pool.acquire()
            if self.pipeline_obj is None:
                raise Exception("Failed to get pipeline object from pool")
                
            self.pcm_fifo_queue = PCMQueue()
            self.stop_pcm = False
            self.wakeup_and_vad = None
            self.generate_outputs = None
            self.dialog_state_callback = None
            self.timer = None
            self.pcm_thread = None
            
            # Initialize VAD from pipeline
            from web.vad import VAD
            self.wakeup_and_vad = VAD()
            
            # Initialize context with system role
            self.reset_context()
            
        except Exception as e:
            print(f"Error initializing DialogStateParams: {e}")
            # Cleanup on error
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
            # Reset and set new system role
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
            if self.pipeline_obj:
                self.pipeline_obj.release()
                self.pipeline_obj = None
        except Exception as e:
            print(f"Error releasing resources: {e}")

def llm_prefill(data, outputs, sid, is_first_pack=False):
    """
    Prefills the LLM for dialog state prediction only.
    
    Parameters:
    - data (dict): Audio chunk data with status and features
    - outputs (dict): Current conversation context
    - sid (str): Session ID
    - is_first_pack (bool): Whether this is the first pack in an IPU
    
    Returns:
    - outputs (dict): Updated conversation context
    """
    
    if data['status'] == 'ipu_sl':
        # Stage1: start listen
        print("Sid: ", sid, " Start listening to new IPU")
        emit_ipu_event(sid, 'ipu_sl')
        outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **outputs)
        emit_state_update(sid, dialog_state=outputs.get('stat'))
    
    elif data['status'] == 'ipu_el':
        # VAD timeout - reset VAD state
        connected_users[sid].wakeup_and_vad.in_dialog = False
        print("Sid: ", sid, " VAD timeout detected")
        emit_ipu_event(sid, 'ipu_el')
        emit_state_update(sid, vad_state=False)
    
    elif data['status'] == 'ipu_cl':
        if outputs['stat'] == 'dialog_cl':
            # Stage2: continue listen - predict dialog state
            outputs = connected_users[sid].pipeline_obj.pipeline_proc.speech_dialogue(
                torch.tensor(data['feature']), **outputs)
            emit_ipu_event(sid, 'ipu_cl')
        
        if is_first_pack:
            # Force dialog_cl state for first pack
            outputs['stat'] = 'dialog_cl'
        
        emit_state_update(sid, dialog_state=outputs.get('stat'))
        
        # Handle state transitions
        if outputs['stat'] == 'dialog_el':
            # User finished but no response needed
            connected_users[sid].wakeup_and_vad.in_dialog = False
            print("Sid: ", sid, " Dialog end detected (no response)")
            emit_state_update(sid, vad_state=False)
            
            # Update shared memory to preserve this interaction
            connected_users[sid].generate_outputs = deepcopy(outputs)
        
        elif outputs['stat'] == 'dialog_ss':
            # User finished and system should start speaking
            connected_users[sid].wakeup_and_vad.in_dialog = False
            print("Sid: ", sid, " Dialog start speak detected")
            emit_state_update(sid, vad_state=False)
            
            # Update shared memory with current context
            connected_users[sid].generate_outputs = deepcopy(outputs)
            
            # Trigger external callback if set
            if connected_users[sid].dialog_state_callback:
                # Pass the current context to the callback
                connected_users[sid].dialog_state_callback(sid, deepcopy(outputs))
    
    return outputs

def send_pcm(sid):
    """
    Main loop for processing PCM audio data and predicting dialog states.
    
    Parameters:
    - sid (str): Session ID
    """
    
    chunk_size = connected_users[sid].wakeup_and_vad.get_chunk_size()
    print("Sid: ", sid, " Start PCM processing with chunk size of", chunk_size)
    
    while True:
        if connected_users[sid].stop_pcm:
            print("Sid: ", sid, " Stop PCM processing")
            break
        
        time.sleep(0.01)
        e = connected_users[sid].pcm_fifo_queue.get(chunk_size)
        if e is None:
            continue
        
        print("Sid: ", sid, f" Received PCM data of suze: {len(e)}")
        
        # Get VAD prediction
        res = connected_users[sid].wakeup_and_vad.predict(np.float32(e))
        
        # Process based on VAD status
        if res['status'] == 'ipu_sl':
            print("Sid: ", sid, " VAD start detected")
            emit_state_update(sid, vad_state=True)
            
            # Create snapshot of current context from shared memory
            outputs = deepcopy(connected_users[sid].generate_outputs)
            
            # Reset caches for new IPU
            outputs['adapter_cache'] = None
            outputs['encoder_cache'] = None
            outputs['pe_index'] = 0
            outputs['stat'] = 'dialog_sl'
            outputs['last_id'] = None
            
            # Clean up text and hidden state if present
            if 'text' in outputs:
                del outputs['text']
            if 'hidden_state' in outputs:
                del outputs['hidden_state']
            
            # Process cached chunks first
            send_dict = {}
            for i, feature in enumerate(res['feature_last_chunk']):
                send_dict['status'] = 'ipu_sl' if i == 0 else 'ipu_cl'
                send_dict['feature'] = feature
                outputs = llm_prefill(send_dict, outputs, sid, is_first_pack=True)
            
            # Process current chunk
            send_dict['status'] = 'ipu_cl'
            send_dict['feature'] = res['feature']
            outputs = llm_prefill(send_dict, outputs, sid)
        
        elif res['status'] in ['ipu_cl', 'ipu_el']:
            # Process continuing or ending chunks
            send_dict = {
                'status': res['status'],
                'feature': res['feature']
            }
            # Use local outputs that were created during ipu_sl
            if 'outputs' in locals():
                outputs = llm_prefill(send_dict, outputs, sid)

def disconnect_user(sid):
    """Disconnect user and cleanup resources"""
    if sid in connected_users:
        print(f"Disconnecting user {sid} due to timeout")
        socketio.emit('out_time', to=sid)
        connected_users[sid].stop_pcm = True
        connected_users[sid].release()
        time.sleep(3)
        del connected_users[sid]

def dialog_ss_callback(sid, context):
    """
    Callback function called when dialog_ss state is predicted.
    This is where you would integrate with your external audio-to-audio LLM.
    
    Parameters:
    - sid (str): Session ID
    - context (dict): Current conversation context
    """
    print(f"Dialog SS callback triggered for user {sid}")
    print(f"Context keys: {context.keys()}")
    
    # Prepare context info for the GUI
    context_info = "External LLM integration point"
    if 'past_tokens' in context and context['past_tokens']:
        # Decode the conversation so far
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

def emit_ipu_event(sid, event_type):
    """Emit IPU events to the GUI"""
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
            return False  # Explicitly reject connection
        
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
        
        # Send initial state before starting PCM thread
        emit_state_update(sid, vad_state=False, dialog_state='dialog_sl', generating=False)
        
        # Start PCM processing thread
        pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
        pcm_thread.start()
        connected_users[sid].pcm_thread = pcm_thread  # Store reference for cleanup
        
        pipeline_pool.print_info()
        print(f'User {sid} connected successfully')
        
        return True  # Explicitly accept connection
        
    except Exception as e:
        print(f'Error in handle_connect for {request.sid}: {e}')
        if request.sid in connected_users:
            # Cleanup on error
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
            
            # Stop PCM processing
            connected_users[sid].stop_pcm = True
            
            # Wait for PCM thread to finish (with timeout)
            if hasattr(connected_users[sid], 'pcm_thread'):
                connected_users[sid].pcm_thread.join(timeout=2.0)
            
            # Release resources
            connected_users[sid].release()
            
            # Remove from connected users
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
        # Reset context for new recording session
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
        print("Sid: ", sid, "Prompt set as: ", text)
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
        
        # Put audio data into queue for processing
        connected_users[sid].pcm_fifo_queue.put(audio_data.astype(np.float32) / 32768.0)
    else:
        disconnect()

if __name__ == "__main__":
    print("Starting Freeze-Omni Dialog State Server")
    cert_file = "web/resources/cert.pem"
    key_file = "web/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    socketio.run(app, host=configs.ip, port=configs.port, ssl_context=(cert_file, key_file))