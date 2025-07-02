import torch
import yaml
import os
import re

from models.utils import init_encoder_llm, load_checkpoint

from logger.logger import setup_logger
import shortuuid

class inferencePipeline():
    def __init__(self, args):
        self.args = args
        self.device = args.get('device', 'cuda:0')
        self.id = shortuuid.uuid()

        self.logger = setup_logger(f'FOPipe_{self.id}', file_log_level="DEBUG", terminal_log_level="INFO")

        self.logger.info(f"Using device: {self.device} for inference pipeline of freeze-omni model.")

        with open(self.args['model_path'] + "/audiollm/train.yaml", 'r') as fin:
            configs = yaml.safe_load(fin)
            configs['cmvn_file'] = self.args['model_path'] + "/audiollm/global_cmvn"
            configs['model_conf']['llm_path'] = self.args['llm_path']

        # Init asr model from configs
        self.model = init_encoder_llm(device = self.device, configs = configs, logger=self.logger)
        
        load_checkpoint(self.model, self.args['model_path'] + "/audiollm/final.pt")
        self.model = self.model.to(self.device)
        self.model.eval()

        # After fully loading the model, we can compile the model for performance
        self.model.init_template_compilation()

        

    def speech_dialogue(self, 
                        audio: tuple, #Audio features
                        identity: str, #Audio source's identity
                        status: str, #current dialogue state
                        role: str=None, #system's role prompt
                        past_key_values=None, #LLM's memory (context, what has been said before)
                        adapter_cache=None,
                        encoder_cache=None,
                        pe_index=0):
        with torch.no_grad():
            ## input fbank
            feats = audio
            if feats is not None:
                feats = feats.to(self.device)
                # feats_lengths = torch.tensor([feats.size(1)]).to(self.device)
            # else:
            #     feats_lengths = None

            extra_inputs = {}
            extra_inputs['identity'] = identity
            extra_inputs['status'] = status##Current dialogue state
            extra_inputs['past_key_values'] = past_key_values
            extra_inputs['adapter_cache'] = adapter_cache
            extra_inputs['encoder_cache'] = encoder_cache
            extra_inputs['pe_index'] = pe_index

            
            if role is not None and past_key_values is None:
                # add <|im_end|> in chat_prefix
                extra_inputs['role_prompt'] = '<|im_start|>system\n' + role # + '<|im_end|>'

            with torch.autocast(device_type="cuda", 
                       dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
                # preprocess system role prompt first ('pre' status)             
                if status == 'pre':
                    ## Pad with system role/prompt as an initial context. Inside the set_system_role function, the model will encode the system role and return the past_key_values.
                    past_key_values = self.model.set_system_role(extra_inputs)

                    return None, past_key_values, None, None, None
                    
                else:
                    # # Standard processing for user/system audio chunks
                    # feats = audio
                    # feats_lengths = torch.tensor([feats.size(1)]).to(self.device)

                    prediction_probs, past_key_values, adapter_cache, encoder_cache, pe_index = self.model.recognize(
                                speech=feats,
                                # feats.to(self.device),
                                # feats_lengths,
                                extra_inputs=extra_inputs)

                    # Return the updated context and prediction results
                    return prediction_probs, past_key_values, adapter_cache, encoder_cache, pe_index

    def post_process(self, text):
        """
        Post-processes the input text to standardize various characters and formatting.

        Parameters:
        - text (str): The input text string to be post-processed.

        Actions:
        1. Replaces various Chinese and English punctuation marks with standardized ones.
        2. Removes newline, tab, and other unwanted whitespace characters.
        3. Removes special characters like asterisks, underscores, backticks, and tildes.
        4. Condenses whitespace following periods and colons.
        5. Adjusts the format of numbered lists to use appropriate separators
        6. Ensures the text ends with an appropriate punctuation mark

        Returns:
        - str: The post-processed text string.
        """
        text = text.replace('、', '，')
        text = text.replace('(', ',')
        text = text.replace(')', ',')
        text = text.replace('（', '，')
        text = text.replace('）', '，')

        text = re.sub(r'[\n\r\t]', '', text)
        text = re.sub(r'[*_`~]', '', text)

        text = re.sub(r'(\.|\:)\s+', r'\1', text)
        
        if re.search(r'[\u4e00-\u9fa5]', text):
            text = re.sub(r'(\d+)\.\s*([\u4e00-\u9fa5A-Za-z])', r'\1：\2', text)
        else:
            text = re.sub(r'(\d+)\.\s*([\w])', r'\1:\2', text)
        
        if text and text[-1] not in ["。", "？", "！", ".", "?", "!"]:
            if text[-1] in [",", "，", ";", "；", ":", "：", "、"]:
                text = text[:-1] + "。"
            else:
                text += "。"
        
        return text
