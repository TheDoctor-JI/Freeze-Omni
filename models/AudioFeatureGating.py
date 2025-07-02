import torch
import torchaudio.compliance.kaldi as k

class AudioFeatureGating:
    """
    Handles stateful fbank feature extraction for every incoming audio chunk
    and gates the features to the next stage based on VAD status.
    """
    def __init__(self, sample_rate, cache_history_size=10, onset_input_chunk_cache_size=6, fbank_config=None):

        self.sample_rate = sample_rate
        self.fbank_config = fbank_config

        self.cache_history_size = cache_history_size
        self.onset_input_chunk_cache_size = onset_input_chunk_cache_size

        ## Read physical time configs
        if fbank_config:
            self.feat_dim = fbank_config['feat_dim']
            self.expected_audio_chunk_duration_in_sec = fbank_config['expected_audio_chunk_duration_in_sec']
            self.audio_to_proc_per_step_in_sec = fbank_config['audio_to_proc_per_step_in_sec']
            self.step_size_in_sec = fbank_config['step_size_in_sec']
            self.context_duration_in_sec = fbank_config['context_duration_in_sec']

        else: # Fallback to defaults if no config is provided
            self.feat_dim = 80
            self.expected_audio_chunk_duration_in_sec = 0.16 ## 160ms
            self.audio_to_proc_per_step_in_sec = 0.025 ##25ms
            self.step_size_in_sec = 0.01 ## 10ms
            self.context_duration_in_sec = 0.03 ## 30ms

        ## Convert from physical time to frames
        self.frames_to_proc_per_step = int(self.audio_to_proc_per_step_in_sec * self.sample_rate)
        self.audio_to_proc_per_step_in_ms = int(self.audio_to_proc_per_step_in_sec * 1000)
        self.step_size_in_ms = int(self.step_size_in_sec * 1000)
        self.step_size_in_frames = int(self.step_size_in_sec * self.sample_rate)
        self.step_cnt_per_chunk = int(self.expected_audio_chunk_duration_in_sec / self.step_size_in_sec)
        self.context_step_cnt = int(self.context_duration_in_sec / self.step_size_in_sec)
        self.frame_overlap = self.frames_to_proc_per_step - self.step_size_in_frames
        self.expected_frames_per_audio_chunk = self.step_size_in_frames * self.step_cnt_per_chunk

        ## State buffers
        self.input_sample = torch.zeros([1, self.expected_frames_per_audio_chunk + self.frame_overlap, 1])
        self.input_chunk = torch.zeros([1, self.step_cnt_per_chunk + self.context_step_cnt, self.feat_dim])
        self.history = torch.zeros([self.cache_history_size, self.step_cnt_per_chunk + self.context_step_cnt, self.feat_dim])

    def reset(self):
        """Resets all state buffers."""
        self.input_sample.zero_()
        self.input_chunk.zero_()
        self.history.zero_()
        print("AudioFeatureGating state has been reset.")

    def _extract_fbank(self, audio_chunk):
        """Extracts fbank features from a raw audio chunk, managing state."""
        with torch.no_grad():
            audio = torch.tensor(audio_chunk)
            sample_data = audio.reshape(1, -1, 1)[:, :, :1] * 32767
            
            # Manage audio sample buffer for continuous STFT
            self.input_sample[:, :self.frame_overlap, :] = self.input_sample[:, -self.frame_overlap:, :].clone()
            self.input_sample[:, self.frame_overlap:, :] = sample_data
            
            # Compute kaldi style fbank features
            xs = k.fbank(waveform=self.input_sample.squeeze(-1), dither=0, 
                         frame_length=self.audio_to_proc_per_step_in_ms,##Audio signal is broken down into 25ms chunks in fbank
                         frame_shift=self.step_size_in_ms,##The step size is 10ms when computing fbank features 
                         num_mel_bins=self.feat_dim
                        )
            
            # Manage feature chunk buffer for model context
            self.input_chunk[:, :self.context_step_cnt, :] = self.input_chunk[:, -self.context_step_cnt:, :].clone()
            self.input_chunk[:, self.context_step_cnt:, :] = xs.squeeze(0)
            
            return self.input_chunk

    def process_and_gate(self, annotated_audio):
        """
        Processes an annotated audio chunk to extract features and decides whether to forward it.
        
        Parameters:
        - annotated_audio (dict): A dict like {'audio': np.array, 'status': 'ipu_cl', 'cached_audio': [..]}

        Returns:
        - dict or None: A dict with features if it should be processed, otherwise None.
        """
        status = annotated_audio['status']
        
        # Always extract features to maintain state, regardless of VAD status
        current_feature = self._extract_fbank(annotated_audio['audio'])

        # If not in an IPU, update history and do not forward
        if status is None:
            self.history[:-1] = self.history[1:].clone()
            self.history[-1:] = current_feature
            return None

        # If in an IPU, prepare the data for the processing thread
        output_data = {
            'feature': current_feature.numpy().tolist(),
            'status': status,
            'feature_last_chunk': []
        }

        if status == 'ipu_sl' and self.onset_input_chunk_cache_size > 0:
            # For IPU start, retrieve the last 6 chunks from history
            output_data['feature_last_chunk'] = self.history[-self.onset_input_chunk_cache_size:].unsqueeze(1).numpy().tolist()

        return output_data