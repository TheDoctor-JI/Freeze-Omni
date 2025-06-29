import json
import torch
import threading
import queue
import subprocess
import concurrent.futures

import numpy as np
import torchaudio.compliance.kaldi as k

from silero_vad.model import load_silero_vad
from silero_vad.utils_vad import VADIterator

class VAD:
    def __init__(self, cache_history=10, silence_threshold_ms=None):
        self.step_cnt_per_chunk = 16##How many steps for each audio chunk
        self.step_overlap_cross_chunk = 3##How many steps from the previous audio chunk to be included in the final input feature to the LLM model
        self.feat_dim = 80
        self.frame_size = 400##Number of frames to be processed in each step
        self.frame_shift = 160##Number of frames to be shifted in each step
        self.silence_threshold_ms = silence_threshold_ms
        self.frame_overlap = self.frame_size - self.frame_shift##Number of overlapping frames in each step
        self.CHUNK = self.frame_shift * self.step_cnt_per_chunk##Number of frames expected in each audio chunk
        self.cache_history = cache_history##Number of input chunks to be cached by the VAD model
        self.in_dialog = False

        with torch.no_grad():
            self.load_vad()
            self.reset_vad()
    
    def get_chunk_size(self):
        return self.CHUNK

    def load_vad(self):
        self.vad_model = load_silero_vad()
        self.vad_model.eval()
        # generate vad itertator
        self.vad_iterator = VADIterator(self.vad_model, 
                                        threshold=0.8, 
                                        sampling_rate=16000, 
                                        min_silence_duration_ms=2000 if self.silence_threshold_ms is None else self.silence_threshold_ms,
                                        speech_pad_ms=30)
        self.vad_iterator.reset_states()

    def reset_vad(self):
        # reset all parms
        self.input_chunk = torch.zeros([1, self.step_cnt_per_chunk + self.step_overlap_cross_chunk, self.feat_dim])
        self.input_sample = torch.zeros([1, self.CHUNK + self.frame_overlap , 1])
        self.history = torch.zeros([self.cache_history, self.step_cnt_per_chunk + self.step_overlap_cross_chunk, self.feat_dim])
        self.vad_iterator.reset_states()
        self.in_dialog = False
    
    def run_vad_iterator(self, audio):
        speech_dict_out = None
        # split into chunk with 512
        for i in range(len(audio) // 512):
            speech_dict = self.vad_iterator(audio[i * 512: (i + 1) * 512], return_seconds=True)
            if speech_dict is not None:
                speech_dict_out = speech_dict
        return speech_dict_out
    
    def predict(self,
                audio: torch.Tensor):
        """
        Predict the Voice Activity Detection (VAD) status and return related features.

        Parameters:
        - audio (torch.Tensor): A 1D or 2D tensor representing the input audio signal (audio chunk).

        Returns:
        - return_dict (dict): A dictionary containing the VAD status and related features.
            - 'status' (str): The current VAD status, which can be 'ipu_sl' (speech start), 
                              'ipu_cl' (speech continue), or 'ipu_el' (speech end).
            - 'feature_last_chunk' (list of list of float): The feature of the last chunks.
            - 'feature' (list of list of float): The feature of the current chunk of audio.
            - 'history_feature' (list of list of list of float): The cached features of previous chunks.
        
        """

        # 1. Converts the input audio tensor to the appropriate format.
        # 2. Computes the filter bank features (fbank) for the audio.
        # 3. Updates the input chunk and history based on the new audio segment.
        # 4. Determines the VAD status by running the VAD iterator on the audio.
        # 5. Populates the return dictionary with the VAD status and related features.

        return_dict = {}
        return_dict['status'] = None
        with torch.no_grad():
            # get fbank feature
            audio = torch.tensor(audio)
            sample_data = audio.reshape(1, -1, 1)[:, :, :1] * 32768
            ##Compose the input audio sample for computing the fbank feature of the current audio chunk. Note that we use the last self.frame_overlap frames from the previous audio chunk as the first self.frame_overlap frames of the current audio chunk to ensure that the first step of the current audio chunk has enough frames to compute the fbank feature.
            self.input_sample[:, :self.frame_overlap , :] = self.input_sample[:, -self.frame_overlap:, :].clone()
            self.input_sample[:, self.frame_overlap:, :] = sample_data
            # compute kaldi style feature
            xs = k.fbank(waveform = self.input_sample.squeeze(-1), dither=0, 
                        frame_length=25, frame_shift=10, num_mel_bins=self.feat_dim)
            ##Compose the final input chunk corresponding to the current audio chunk. Note that we use the last self.step_overlap_cross_chunk steps from the previous input chunk as the first self.step_overlap_cross_chunk steps of the current input chunk
            self.input_chunk[:, :self.step_overlap_cross_chunk, :] = self.input_chunk[:, -self.step_overlap_cross_chunk:, :].clone()
            self.input_chunk[:, self.step_overlap_cross_chunk:, :] = xs.squeeze(0)

            # get vad status
            if self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "end" in speech_dict:
                    ## The last chunk which causes exceeding a threshold is labeled as 'ipu_el'. Note that the VAD does not transition back into the in_dialog = False state here. This transition is driven by external server code.
                    return_dict['status'] = 'ipu_el' # Speech end, but not start speaking (e.g., human pause)
                    # reset state
                    self.vad_iterator.reset_states()
                else:
                    ## Most chunks will be labeled as 'ipu_cl' (continue speaking) when the VAD is in in_dialog state, even if those chunks do not contain VA themselves
                    return_dict['status'] = 'ipu_cl'
            if not self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "start" in speech_dict:
                    ## The first chunk that causes the VAD to transition in in_dialog state will be labeld as 'ipu_sl'.
                    return_dict['status'] = 'ipu_sl'
                    self.in_dialog = True
                    # self.vad_iterator.reset_states()
                else:  
                    ## In this case, the chunk is labeled as None.
                    ## cache fbank feature when not in an IPU and is not at the onset of IPU
                    self.history[:-1] = self.history[1:].clone()
                    self.history[-1:] = self.input_chunk

            # return dict
            if return_dict['status'] == 'ipu_sl':
                ##copy last 6 chunks cached. These would be chunks from outside an IPU.
                return_dict['feature_last_chunk'] = self.history[-6:].unsqueeze(1).numpy().tolist()
                return_dict['feature'] = self.input_chunk.numpy().tolist()
                return_dict['history_feature'] = self.history.numpy().tolist()
            elif return_dict['status'] == 'ipu_cl' or return_dict['status'] == 'ipu_el':
                return_dict['feature_last_chunk'] = None
                return_dict['feature'] = self.input_chunk.numpy().tolist()
                return_dict['history_feature'] = self.history.numpy().tolist()

        return return_dict
