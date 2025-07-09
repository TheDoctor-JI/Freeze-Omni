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
from flask_socketio import SocketIO, disconnect
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from bin.pool import pipelineObjectPool
from web.queue import PCMQueue, ProcPCMQueue, ThreadSafeQueue
from models.AudioFeatureGating import AudioFeatureGating
from models.ContextSerializer import ContextSerializer
from flask import Blueprint, request
from flask_socketio import disconnect
from logger.logger import setup_logger
from FloorState.FloorStateEvent import FloorStateDef, FloorEvent, FloorEventType
from FloorState.floor_state_emission import *
from FloorState.IPUHandle import IPUHandle
import shortuuid

def get_args():

    inference_config_path = '/home/eeyifanshen/e2e_audio_LLM/dialog_turntaking_new/Freeze-Omni/configs/dialog_state_pred_config.yaml'

    # Load config from YAML file
    with open(inference_config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Freezeomni configuration loaded:", json.dumps(config, indent=2))
    return config


def custom_print(*args, **kwargs):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    original_print(f'[{current_time}]', *args, **kwargs)


# change print function to add time stamp
original_print = builtins.print
builtins.print = custom_print


'''
Main logic for dialog state prediction
'''
class DialogStateParams:
    """
    This class holds all the parameters and context for dialog state prediction.
    """

    ## Class variables:
    DIALOG_STATE_PRED_CONFIGS = get_args()
    MAX_PIPELINE_NUN = 1
    PIPELINE_POOL = pipelineObjectPool(size=MAX_PIPELINE_NUN, configs=DIALOG_STATE_PRED_CONFIGS)
    EXPECTED_SAMPLING_RATE = DIALOG_STATE_PRED_CONFIGS['audio']['expected_sampling_rate']
    EXPECTED_ENCODING = 's16le'
    RESPONSE_THRESHOLD = DIALOG_STATE_PRED_CONFIGS['dialog_state_decision']['resp_threshold']
    SLEEP_INTERVAL = DIALOG_STATE_PRED_CONFIGS['thread_sleep_interval']

    def __init__(self, sid, socketio, event_outlet, ipu_audio_outlet, parent_logger=None):
        try:
            self.sid = sid
            self.event_outlet = event_outlet
            self.ipu_audio_outlet = ipu_audio_outlet
            if parent_logger is not None:
                self.logger = parent_logger.getChild(f"DialogStateParams")
            else:
                self.logger = setup_logger(f"{self.sid}_DialogStateParams", file_log_level="DEBUG", terminal_log_level="INFO")

            ## Config for dialog state prediction
            self.dialog_state_pred_configs = DialogStateParams.DIALOG_STATE_PRED_CONFIGS

            self.debug_time = self.dialog_state_pred_configs['debug_time']

            ## Shared context
            self.socketio = socketio
            self.pipeline_pool = DialogStateParams.PIPELINE_POOL
            self.pipeline_obj = self.pipeline_pool.acquire()
            if self.pipeline_obj is None:
                raise Exception("Failed to get pipeline object from pool")
            else:
                self.logger.debug(f"Acquired pipeline object {self.pipeline_obj.id} for dialog state prediction.")
                self.pipeline_obj.pipeline_proc.setup_logger(self.logger)

            ## Internal parameters for this class

            # init default prompt
            _, init_key_values, _, _, _ = self.pipeline_obj.pipeline_proc.speech_dialogue(None, identity = '', status='pre', 
                                                                        role=self.dialog_state_pred_configs['inference_control']['default_prompt'])
            self.system_role = deepcopy(init_key_values)

            
            # Dialog state prediction context
            self.caches = {} # Holds caches for 'user' and 'system'
            self.past_key_values = None
            self.dialog_state_callback = None
            
            self.feature_gater = {
                'user': AudioFeatureGating(
                    sample_rate=self.dialog_state_pred_configs['audio']['expected_sampling_rate'],
                    cache_history_size=self.dialog_state_pred_configs['audio_feature_gating']['feature_gating_history_size'],
                    onset_input_chunk_cache_size=self.dialog_state_pred_configs['audio_feature_gating']['onset_input_chunk_cache_size'],
                    fbank_config=self.dialog_state_pred_configs['audio_feature_gating']['fbank']
                ),
                'system': AudioFeatureGating(
                    sample_rate=self.dialog_state_pred_configs['audio']['expected_sampling_rate'],
                    cache_history_size=self.dialog_state_pred_configs['audio_feature_gating']['feature_gating_history_size'],
                    onset_input_chunk_cache_size=self.dialog_state_pred_configs['audio_feature_gating']['onset_input_chunk_cache_size'],
                    fbank_config=self.dialog_state_pred_configs['audio_feature_gating']['fbank']
                )
            }

            if self.dialog_state_pred_configs['vad']['use_standalone_vad']:
                from periphrals.PureVAD import PureVAD
                self.standalone_vad = {
                    'user': PureVAD(
                        min_silent_duration_second=self.dialog_state_pred_configs['vad']['min_silent_duration_second'],
                        speech_pad_second=self.dialog_state_pred_configs['vad']['speech_pad_second'],
                        vad_threshold=self.dialog_state_pred_configs['vad']['vad_threshold'],
                        cache_history_size=self.dialog_state_pred_configs['vad']['vad_history_cache_chunk_cnt'],
                        audio_chunk_size=self.feature_gater['user'].expected_frames_per_audio_chunk,
                    ),
                    'system': PureVAD(
                        min_silent_duration_second=self.dialog_state_pred_configs['vad']['min_silent_duration_second'],
                        speech_pad_second=self.dialog_state_pred_configs['vad']['speech_pad_second'],
                        vad_threshold=self.dialog_state_pred_configs['vad']['vad_threshold'],
                        cache_history_size=self.dialog_state_pred_configs['vad']['vad_history_cache_chunk_cnt'],
                        audio_chunk_size=self.feature_gater['system'].expected_frames_per_audio_chunk,
                    )
                }


            # Initialize context serializer
            self.context_serializer = ContextSerializer()


            # Control flags
            self.stop_all_threads = False
            
            
        except Exception as e:
            self.logger.error(f"Error initializing DialogStateParams: {e}")
            self.release()
            raise
    
    def reset_context(self):
        """Reset the conversation context"""
        try:

            # Audio queues for user and system separately
            self.audio_data_input_queue = ProcPCMQueue()

            self.raw_pcm_queue = {
                'user': ProcPCMQueue(),
                'system': ProcPCMQueue()
            }

            self.all_ipus = {
                'user': {},
                'system': {}
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


            # Reset context serializer
            if hasattr(self, 'context_serializer'):
                self.context_serializer.reset()
                

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
            self.logger.error(f"Error resetting context: {e}")
            self.release()
            raise
    
    def start_all_threads(self):
        """Start all necessary threads for dialog state prediction"""
        try:
            # Start the data input thread to receive audio chunks
            self.data_input_thread = threading.Thread(
                target=self.receive_raw_audio_chunk,
                name="DataInput_Thread"
            )
            self.data_input_thread.start()

            # Start VAD threads for user and system
            if self.dialog_state_pred_configs['vad']['use_standalone_vad']:
                self.vad_threads = {}
                for identity in ['user', 'system']:
                    self.vad_threads[identity] = threading.Thread(
                        target=self.vad_annotation,
                        args=(identity,),
                        name=f"VAD_Thread_{identity}"
                    )
                    self.vad_threads[identity].start()
                
            # Start feature gating threads for user and system
            self.feature_gating_threads = {}
            for identity in ['user', 'system']:
                self.feature_gating_threads[identity] = threading.Thread(
                    target=self.feature_gating,
                    args=(identity,),
                    name=f"FeatureGating_Thread_{identity}"
                )
                self.feature_gating_threads[identity].start()

            # Start context serializer thread
            self.context_serializer_thread = threading.Thread(
                target=self.serialize_context,
                name="ContextSerializer_Thread"
            )
            self.context_serializer_thread.start()

            # Start dialog state prediction thread
            self.dialog_state_prediction_thread = threading.Thread(
                target=self.predict_dialog_state,
                name="DialogStatePrediction_Thread"
            )
            self.dialog_state_prediction_thread.start()

        except Exception as e:
            self.logger.error(f"Error starting threads: {e}")
            self.release()
            raise

    def set_dialog_callback(self, callback):
        """Set callback function to be called when dialog_ss is predicted"""
        self.dialog_state_callback = callback
    
    def set_prompt(self, prompt):
        """Set system prompt and reset context"""
        self.system_role = self.pipeline_obj.pipeline_proc.speech_dialogue(
                                                            audio = None, 
                                                            status='pre', 
                                                            identity = None,
                                                            role=prompt)

    def release(self):
        """Release resources"""
        try:
            self.stop_all_threads = True
            if self.pipeline_obj:
                self.pipeline_pool.release(self.pipeline_obj)

            ## Wait for all threads to finish
            if hasattr(self, 'data_input_thread'):
                self.data_input_thread.join(timeout=2)

            if hasattr(self, 'vad_threads'):
                for thread in self.vad_threads.values():
                    thread.join(timeout=2)

            if hasattr(self, 'feature_gating_threads'):
                for thread in self.feature_gating_threads.values():
                    thread.join(timeout=2)

            if hasattr(self, 'context_serializer_thread'):
                self.context_serializer_thread.join(timeout=2)

            if hasattr(self, 'dialog_state_prediction_thread'):
                self.dialog_state_prediction_thread.join(timeout=2)

        except Exception as e:
            self.logger.error(f"Error releasing resources: {e}")

    def enqueue_audio_data(self, identity, audio_data_dict):
        """
        Enqueue audio data for processing.
        audio_data_dict should be a dictionary with the following keys:
                {
                    'audio': audio_chunk,  # Raw audio data in bytes
                    'sr': sampling_rate,   # Sampling rate, e.g., 16000
                    'enc': encoding,       # Encoding, e.g., 's16le'
                    'time_stamp': timstamp # Timestamp for the audio chunk
                },
        """
        self.audio_data_input_queue.put(
            (
                audio_data_dict,
                identity
            )
        )

    def receive_raw_audio_chunk(self):

        '''
        Expect audio data of the form
        {
            'audio': <bytes>,  # Raw audio data in bytes
            'sr': <int>,      # Sampling rate, e.g., 16000
            'enc': <str>      # Encoding, e.g., 's16le'
            'time_stamp': <float> # timestamp for the audio chunk
        }
        as well as an identity string which can be either 'user' or 'system'.
        from a queue which receives audio data from the top level server
        '''
        try:
            
            while not self.stop_all_threads:

                ## Get the audio data from the input queue
                time.sleep(DialogStateParams.SLEEP_INTERVAL)

                data_item = self.audio_data_input_queue.get()

                if data_item is None:
                    continue

                # self.logger.debug(f"Sid: {self.sid} Received raw audio chunk")

                audio_dat_dict, identity = data_item

                ## Check the data format
                if(audio_dat_dict['sr'] != DialogStateParams.EXPECTED_SAMPLING_RATE):
                    raise ValueError(f"Expected audio sampling rate {DialogStateParams.EXPECTED_SAMPLING_RATE}, but got {audio_dat_dict['sr']}")
                if(audio_dat_dict['enc'] != 's16le'):
                    raise ValueError(f"Expected audio encoding '{DialogStateParams.EXPECTED_ENCODING}', but got {audio_dat_dict['enc']}")
                
                audio_chunk = np.frombuffer(bytes(audio_dat_dict['audio']), dtype=np.int16)
                audio_chunk = audio_chunk.astype(np.float32) / 32767.0

                ## Create a new dict as a deep copy
                new_audio_dat_dict = {
                    'audio': audio_chunk,  # Raw audio data as a numpy array
                    'sr': audio_dat_dict['sr'],  # Sampling rate, e.g., 16000
                    'enc': audio_dat_dict['enc'],  # Encoding, e.g., 's16le'
                    'time_stamp': audio_dat_dict['time_stamp']  # Timestamp for the audio chunk
                }

                ## Enqueue the audio chunk to the respective queue for subsequent processing
                self.raw_pcm_queue[identity].put(new_audio_dat_dict)
        
        except Exception as e:
            self.logger.error(f"Error initializing DialogStateParams: {e}")
            self.release()
            raise

    def vad_annotation(self, identity):
        """
        Standalone VAD thread that consumes raw audio for a specific identity,
        annotates it, and puts it into the corresponding annotated_audio_queue.
        """

        try:
            total_ipus = 0
            chunk_size = self.standalone_vad[identity].get_chunk_size()
            self.logger.debug(f"Sid: {self.sid} Starting standalone VAD thread for '{identity}' with chunk size: {chunk_size}")
            
            
            aggregated_audio_data_dict = {
                'audio': np.array([], dtype=np.float32),  # Initialize with empty array
                'sr': DialogStateParams.EXPECTED_SAMPLING_RATE,
                'enc': DialogStateParams.EXPECTED_ENCODING,
                'time_stamp': None  # Use current time as timestamp
            }

            current_ipu = None
            while not self.stop_all_threads:

                time.sleep(DialogStateParams.SLEEP_INTERVAL)

                new_audio_data_dict = self.raw_pcm_queue[identity].get()
                if new_audio_data_dict is None:
                    continue

                ## Gather sufficient data before running the VAD
                if(len(aggregated_audio_data_dict['audio']) < chunk_size):
                    
                    ## Get more samples
                    samples_to_add = chunk_size - len(aggregated_audio_data_dict['audio'])
                    samples_obtained = len(new_audio_data_dict['audio'])

                    if samples_obtained < samples_to_add:
                        ## Not enough samples, just add the available samples, update the timestamp, and then we will wait for more samples
                        aggregated_audio_data_dict['audio'] = np.concatenate((aggregated_audio_data_dict['audio'], new_audio_data_dict['audio']))
                        aggregated_audio_data_dict['time_stamp'] = new_audio_data_dict['time_stamp']
                        continue
                    else:
                        ## Enough samples, add the samples and update the timestamp
                        aggregated_audio_data_dict['audio'] = np.concatenate((aggregated_audio_data_dict['audio'], new_audio_data_dict['audio'][:samples_to_add]))
                        aggregated_audio_data_dict['time_stamp'] = new_audio_data_dict['time_stamp']

                        ## We will process this aggregated audio data now
                        sufficient_audio_data_dict = aggregated_audio_data_dict

                        ## Whatever is left in the new audio data becomes the new aggregated audio data
                        aggregated_audio_data_dict = {
                            'audio': new_audio_data_dict['audio'][samples_to_add:],  # Remaining audio data
                            'sr': new_audio_data_dict['sr'],  # Sampling rate, e.g.,
                            'enc': new_audio_data_dict['enc'],  # Encoding, e.g., 's16le'
                            'time_stamp': new_audio_data_dict['time_stamp']  # Timestamp for the audio chunk
                        }
                else:
                    ## We have enough samples, just use the aggregated audio data
                    sufficient_audio_data_dict = aggregated_audio_data_dict
                    
                    ## Reset the aggregated audio data
                    aggregated_audio_data_dict = {
                        'audio': np.array([], dtype=np.float32),  # Reset to empty array
                        'sr': DialogStateParams.EXPECTED_SAMPLING_RATE,
                        'enc': DialogStateParams.EXPECTED_ENCODING,
                        'time_stamp': None  # Use current time as timestamp
                    }

                # if(self.debug_time):
                #     self.logger.debug(f"Sid: {self.sid} VAD received raw audio chunk of size: {len(sufficient_audio_data_dict['audio'])} for '{identity}'")
                    
                ## Run VAD prediction to get annotated audio
                ## Return: {'audio': audio_chunk, 'status': 'ipu_sl', 'cached_audio': [chunk1, chunk2...], 'time_stamp': a timestamp}
                annotated_audio = self.standalone_vad[identity].predict(sufficient_audio_data_dict)

                # if(self.debug_time):
                #     self.logger.debug(f"Sid: {self.sid} VAD annotation done for {identity}.")
                    

                # Emit VAD events for monitoring GUI (only for user)
                status = annotated_audio['status']
                if status == 'ipu_sl':

                    ## Obtain a new ID for the new IPU
                    if current_ipu is not None:
                        raise ValueError(f"We currently have IPU {current_ipu.id}, but a new IPU start is detected.")

                    total_ipus += 1
                    
                    ## Instantiate a new IPUHandle object for all audio data associated with this IPU
                    current_ipu = IPUHandle(
                        sid = self.sid,
                        ipu_id = total_ipus,
                        identity = identity,
                        start_timestamp = annotated_audio['time_stamp'],
                        initial_chunk = annotated_audio['audio'],
                        pre_cached_chunks = annotated_audio['cached_audio']
                    )
                    self.all_ipus[identity][current_ipu.id] = current_ipu
                    

                    ## Label the data with the current IPU ID
                    annotated_audio['ipu_id'] = current_ipu.id
                    
                    vad_state = True

                    if(self.debug_time):##VAD annotation usually takes less than 10ms
                        self.logger.debug(f"Sid: {self.sid} SL chunk obtained for {current_ipu.id}.")


                    if identity == 'user':
                        ##Forward the IPU handle object of the user to the LLM
                        self.ipu_audio_outlet(current_ipu)
                        ##Also forward it to the floor state machine
                        self.event_outlet(current_ipu)

                elif status == 'ipu_el':

                    ## Add audio chunk to the current IPU
                    current_ipu.add_chunk(annotated_audio['audio'])

                    ## Set the end timestamp for the current IPU
                    current_ipu.set_end_timestamp(annotated_audio['time_stamp'])

                    ## Use existing IPU ID for the end of the IPU
                    annotated_audio['ipu_id'] = current_ipu.id
                
                    vad_state = False

                    if self.debug_time:##VAD annotation usually takes less than 10ms
                        self.logger.debug(f"Sid: {self.sid} EL chunk obtained for {current_ipu.id}.")


                    ## Since this is the end of an IPU, reset the id
                    current_ipu = None


                elif status == 'ipu_cl':

                    ## Add audio chunk to the current IPU
                    current_ipu.add_chunk(annotated_audio['audio'])

                    ## Use existing IPU ID for the continuation of an IPU
                    annotated_audio['ipu_id'] = current_ipu.id

                    vad_state = True

                else:
                    annotated_audio['ipu_id'] = -1 ## This implies that the audio data do not belong to any IPU

                    vad_state = None

                if vad_state is not None:
                    ## Emit VAD state and event to the GUI for visualization
                    emit_vad_state_update(socketio = self.socketio, sid=self.sid, vad_state=vad_state,  identity=identity)
                    emit_vad_event(socketio = self.socketio, sid=self.sid, event_type = status, identity=identity)

                    # Put annotated audio associated with an IPU into the queue for the feature gating thread
                    self.annotated_audio_queue[identity].put(annotated_audio)

            self.logger.debug(f"Sid: {self.sid} Stopping standalone VAD thread for '{identity}'")
        
        except Exception as e:
            self.logger.error(f"Error initializing DialogStateParams: {e}")
            self.release()
            raise

    def feature_gating(self, identity):
        """
        This thread gets annotated audio for a specific identity, computes fbank features,
        and gates the features to the serializer
        """
        try:
            self.logger.debug(f"Sid: {self.sid} Starting feature gating thread for '{identity}'.")

            while not self.stop_all_threads:

                time.sleep(DialogStateParams.SLEEP_INTERVAL)
                
                # Get annotated audio chunk for the specific identity
                annotated_audio = self.annotated_audio_queue[identity].get()
                

                if(annotated_audio is None):
                    continue

                # self.logger.debug(f"Sid: {self.sid} Received annotated audio chunk with status: {annotated_audio['status']}, size: {len(annotated_audio['audio'])}")


                # Process chunk to extract and gate features
                gated_feature_data = self.feature_gater[identity].process_and_gate(annotated_audio)
                

                # If the feature gater returns data, put it in the processed queue for the LLM to process.
                if gated_feature_data:
                    # Set the identity for this chunk
                    gated_feature_data['identity'] = identity
                    gated_feature_data['ipu_id'] = annotated_audio['ipu_id']  # Use the IPU ID from the annotated audio 

                    # if self.debug_time:
                    #     self.logger.debug(f"Sid: {self.sid} Approved audio feature, status: {gated_feature_data['status']}, identity: {identity}")


                    if 'time_stamp' in annotated_audio:
                        gated_feature_data['time_stamp'] = annotated_audio['time_stamp']
                    
                    if gated_feature_data['status'] == 'ipu_sl':

                        # if(self.debug_time):##fbank feature gating usually takes around 20ms
                        #     self.logger.debug(f"Sid: {self.sid} SL chunk approved for {identity}.")

                        for i, feature in enumerate(gated_feature_data['feature_last_chunk']):
                            feature_item = {
                                'identity': identity,
                                'feature': feature,
                                'status': 'ipu_sl' if i == 0 else 'ipu_cl',
                                'time_stamp': gated_feature_data['time_stamp'],##For these overbleed features from the last chunk, just set the time stamp to be the same as the current chunk
                                'ipu_id': gated_feature_data['ipu_id']  ## Keep these features associated with the same IPU ID
                            }

                            # if self.debug_time:
                            #     self.logger.debug(f"Sid: {self.sid} Adding feature chunk, status: {feature_item['status']}, identity: {identity}")

                            self.context_serializer.add_feature_chunk(feature_item)
                        
                        ## Copy over the returned gated_feature_data, but change the status if necessary
                        feature_item = {
                            'identity': identity,
                            'feature': gated_feature_data['feature'],
                            'status': 'ipu_cl' if len(gated_feature_data['feature_last_chunk']) > 0 else 'ipu_sl',
                            'time_stamp': gated_feature_data['time_stamp'],
                            'ipu_id': gated_feature_data['ipu_id']
                        }

                        # if self.debug_time:
                        #     self.logger.debug(f"Sid: {self.sid} Adding feature chunk, status: {feature_item['status']}, identity: {identity}")

                        self.context_serializer.add_feature_chunk(feature_item)

                    else:

                        # if self.debug_time:
                        #     self.logger.debug(f"Sid: {self.sid} Adding feature chunk, status: {feature_item['status']}, identity: {identity}")

                        self.context_serializer.add_feature_chunk(gated_feature_data)

            self.logger.debug(f"Sid: {self.sid} Stopping feature gating thread for '{identity}'.")

        except Exception as e:
            self.logger.error(f"Error initializing DialogStateParams: {e}")
            self.release()
            raise

    def serialize_context(self):
        """
        Context serializer thread that takes features from both user and system in the priority queue it maintains, and
        serializes them based on timestamps, and outputs to the processed_pcm_queue.
        """

        try:

            self.logger.debug(f"Sid: {self.sid} Starting context serializer thread.")
            
            while not self.stop_all_threads:
                time.sleep(DialogStateParams.SLEEP_INTERVAL)

                ## Get the next feature to process
                feature_to_process = self.context_serializer.get_next_feature()

                if feature_to_process is None:
                    continue

                ## Send to the main processing queue for dialog state prediction
                if feature_to_process is not None:
                    # if self.debug_time:
                    #     self.logger.debug(f"Sid: {self.sid} Comitting feature to processed_pcm_queue, status: {feature_to_process['status']}, identity: {feature_to_process.get('identity', 'N/A')}")

                    self.processed_pcm_queue.put(feature_to_process)
            
            self.logger.debug(f"Sid: {self.sid} Stopping context serializer thread.")

        except Exception as e:
            self.logger.error(f"Error initializing DialogStateParams: {e}")
            self.release()
            raise

    def predict_dialog_state(self):
        """
        Main loop for processing gated audio and predicting dialog states.
        Uses VAD-provided labels to determine IPU boundaries.
        """
        try:

            self.logger.debug(f"Sid: {self.sid} Starting dialog state prediction thread")
                
            while True:

                time.sleep(DialogStateParams.SLEEP_INTERVAL)
                
                if self.stop_all_threads:
                    self.logger.debug(f"Sid: {self.sid} Stopping dialog state prediction thread")
                    break
                        
                # Get processed audio from the gated queue
                feature_data = self.processed_pcm_queue.get()  # Get one item, i.e., one chunk
                if feature_data is None:
                    continue
                    
                # self.logger.debug(f"Sid: {self.sid} Processing approved audio for dialog state prediction, status: {feature_data['status']}, identity: {feature_data.get('identity', 'N/A')}")
                

                # Always run forward processing
                if self.debug_time and feature_data['status'] == 'ipu_sl':
                    
                    self.logger.debug(f"Sid: {self.sid} Starting dialog state prediction for ipu_sl feature data of ipu {feature_data['ipu_id']}")
                    
                predicted_state = self.llm_prefill(feature_data)

                if self.debug_time and feature_data['status'] == 'ipu_sl':
                    self.logger.debug(f"Sid: {self.sid} Dialog state prediction done.")


                ## Update the response requirement of the associated IPU based on the predicted state
                if feature_data['identity'] == 'user':
                    self.logger.debug(f"Sid: {self.sid} Updating dialog state for user IPU {feature_data['ipu_id']}. Latest prediction is {predicted_state}")
                    user_ipu = self.all_ipus['user'].get(feature_data['ipu_id'], None)
                    if user_ipu is not None:
                        user_ipu.register_response_state(predicted_state)

        except Exception as e:
            self.logger.error(f"Error initializing DialogStateParams: {e}")
            self.release()
            raise
        
    def llm_prefill(self, data):
        """
        Processes an audio chunk to update the shared conversational context.
        If the chunk is from the user, it also predicts dialog state probabilities.
        
        Parameters:
        - data (dict): Audio chunk data with features, status, and identity.
        """
        
        identity = data['identity']
        
        
        # self.logger.debug(f"Sid: {self.sid} Processing audio chunk with status: {data['status']} from {identity}")
        

        # 1. Assemble the context for this processing step
        context_input = {
            'past_key_values': self.past_key_values, # Shared conversational history
            'identity': identity,
            'status': data['status'],
            **self.caches[identity] # Identity-specific caches (encoder, adapter, pe_index)
        }



        # 2. Call the pipeline
        prediction_probs, past_key_values, cnn_cache, buffer, pe_index = self.pipeline_obj.pipeline_proc.speech_dialogue(
            torch.tensor(data['feature']), **context_input
        )


        # 3. Update the shared and identity-specific contexts
        self.past_key_values = past_key_values # Update shared history
        self.caches[identity] = { # Update identity-specific caches
            'encoder_cache': buffer,
            'adapter_cache': cnn_cache,
            'pe_index': pe_index
        }
        

        # 4. Handle prediction results if the input was from the user
        if identity == 'user' and prediction_probs is not None:

            # Check if the system should start generating a new response
            if prediction_probs["state_1"] > DialogStateParams.RESPONSE_THRESHOLD:

                predicted_state = 'dialog_ss'  # System should generate a response

                # Trigger external callback
                emit_dialog_ss_callback(socketio=self.socketio, sid=self.sid)

            else:## No point differentiating dialog_cl and dialog_el state for now.

                predicted_state = 'dialog_cl'  # System should continue listening

            ## Emit the state prediction to the GUI for visualization
            emit_dialog_state_update(
                socketio=self.socketio,
                sid=self.sid,
                dialog_state=predicted_state
            )
            ## Also send the dialog state update event to the event outlet for further processing
            self.event_outlet(
                FloorEvent(
                    event_type=FloorEventType.DIALOG_STATE_REPORT,
                    event_data={
                        'dialog_state': predicted_state,
                    }
                )
            )


        else:

            predicted_state = None  # No dialog state prediction for system input

        return predicted_state


    def warmup_compiled_methods(self):
        ## Push a few audio samples to feature gating queue of both human and system
        num_of_cl_chunks = 5
        for identity in ['user', 'system']:
            chunk_size = self.standalone_vad[identity].get_chunk_size()
            ## Push directly to the feature gating queue
            self.annotated_audio_queue[identity].put({
                'audio': np.zeros(chunk_size, dtype=np.float32),
                'sr': DialogStateParams.EXPECTED_SAMPLING_RATE,
                'enc': DialogStateParams.EXPECTED_ENCODING,
                'status': 'ipu_sl',
                'time_stamp': time.time(),
                'ipu_id': 'warmup_ipu'
            })
            for i in range(num_of_cl_chunks):
                self.annotated_audio_queue[identity].put({
                    'audio': np.zeros(chunk_size, dtype=np.float32),
                    'sr': DialogStateParams.EXPECTED_SAMPLING_RATE,
                    'enc': DialogStateParams.EXPECTED_ENCODING,
                    'status': 'ipu_cl',
                    'time_stamp': time.time(),
                    'ipu_id': 'warmup_ipu'
                })
            self.annotated_audio_queue[identity].put({
                'audio': np.zeros(chunk_size, dtype=np.float32),
                'sr': DialogStateParams.EXPECTED_SAMPLING_RATE,
                'enc': DialogStateParams.EXPECTED_ENCODING,
                'status': 'ipu_el',
                'time_stamp': time.time(),
                'ipu_id': 'warmup_ipu'
            })
            time.sleep(1)  # Give some time for the feature gating thread to process these samples before pushing for the other identity

        time.sleep(15)

        ## Wait for the feature gating threads to finish processing
        while self.processed_pcm_queue.queue.qsize() > 0:
            time.sleep(0.1)

        ## Wait a bit longer to make sure the processing of the last chunk is done
        time.sleep(5)

        self.logger.debug(f"DialogParams: Warmed up compiled methods for user {self.sid}.")