import torch
import yaml
import os
import re

from models.utils import init_encoder_llm, load_checkpoint

class inferencePipeline():
    def __init__(self, args):
        self.args = args

        with open(self.args.model_path + "/audiollm/train.yaml", 'r') as fin:
            configs = yaml.safe_load(fin)
            configs['cmvn_file'] = self.args.model_path + "/audiollm/global_cmvn"
            configs['model_conf']['llm_path'] = self.args.llm_path

        # Init asr model from configs
        self.model = init_encoder_llm(configs)
        
        load_checkpoint(self.model, self.args.model_path + "/audiollm/final.pt")
        device = torch.device('cuda')
        self.model = self.model.to(device)
        self.model.eval()

    def speech_dialogue(self, 
                        audio: tuple, #Audio features
                        role: str=None, #system's role
                        stat: str='dialog_sl', #current dialogue state
                        past_key_values=None, #LLM's memory (context, what has been said before)
                        last_id=None,
                        past_tokens=None,
                        adapter_cache=None,
                        encoder_cache=None,
                        pe_index=0):
        with torch.no_grad():
            ## input fbank
            feats = audio
            if feats is not None:
                feats = feats.to('cuda')
                feats_lengths = torch.tensor([feats.size(1)]).to('cuda')
            else:
                feats_lengths = None

            extra_inputs = {}
            extra_inputs['top_p'] = self.args.top_p
            extra_inputs['top_k'] = self.args.top_k
            extra_inputs['temperature'] = self.args.temperature
            extra_inputs['past_key_values'] = past_key_values
            extra_inputs['stat'] = stat##Current dialogue state
            extra_inputs['last_id'] = last_id
            extra_inputs['adapter_cache'] = adapter_cache
            extra_inputs['encoder_cache'] = encoder_cache
            extra_inputs['pe_index'] = pe_index
            if role is not None and past_key_values is None:
                # add <|im_end|> in chat_prefix
                extra_inputs['role'] = '<|im_start|>system\n' + role # + '<|im_end|>'

            with torch.autocast(device_type="cuda", 
                       dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
                # preprocess system role first ('pre' state)             
                if stat == 'pre':
                    ## Pad with system role/prompt as an initial context. Inside the set_system_role function, the model will encode the system role and return the past_key_values.
                    past_key_values = self.model.set_system_role(extra_inputs)
                    stat = 'dialog_sl'  # Set the state to 'dialog_sl' after setting the system role
                else:
                    ## Dialogue state prediction 
                    (last_id, stat, past_key_values, adapter_cache, 
                            encoder_cache, pe_index, hidden_state) = self.model.recognize(
                                feats,
                                feats_lengths,
                                extra_inputs=extra_inputs)

            ## This now contains the updated state: the updated past_key_values, as well as the forward pass outputs, i.e., either the stat or the last_id.
            outputs = dict(
                past_key_values=past_key_values,
                stat=stat,
                last_id=last_id,
                adapter_cache=adapter_cache,
                encoder_cache=encoder_cache,
                pe_index=pe_index,
            )

            if stat == 'dialog_cs':
                ## The recognize pass in this case outputs the next text token which can then be used for TTS
                if past_tokens is None:
                    past_tokens = []
                past_tokens.append(last_id[0][0])
                text = self.model.tokenizer.decode(past_tokens, skip_special_tokens=True)
                outputs['hidden_state'] = hidden_state
                outputs['text'] = text
                outputs['past_tokens'] = past_tokens
            
            return outputs

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
