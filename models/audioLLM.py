import random
import torch
import copy
import re

from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from models.adapter import *
import shortuuid

IGNORE_ID = -1





class AudioLLM(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        llm_path: str,
        device: str = "cuda:0",  # Add device parameter
        freeze_llm: bool = True,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 3,
        IGNORE_ID: int = -100,
        adpter_type: str = 'cnn',
        add_audio_bos_eos: bool = False,
        task_num: int = 10,
        add_ctc_prompt_ratio: float = 0.0,
        lang_dict: dict = None,
        ctc: torch.nn.Module = None,
        tokenize_ctc_char: bool = False,
        task_before_audio: bool = False,
        hyp_before_task: bool = False,
        prompt_finetune: bool = False,
        add_prompt_before: bool = False,
        prompt_num: int = 5,
        prefix_finetune: bool = False,
        prefix_num: int = 5,
        llm_head_num: int = 32,
        num_key_value_heads: int = None,
        task_type: str = 'prompt',
        freeze_encoder: bool = False,
        freeze_adpter: bool = False,
        activation_func: str = 'relu',
        norm: str = 'batch',
        use_lora: bool = False,
        clone_encoder: torch.nn.Module = None,
        chat_template: str = None,
        predict_usr_state: int = 0,
        chunk_size: int = -1,
    ):
        super().__init__()
        self.id = shortuuid.uuid()
        self.device = torch.device(device)

        self.encoder_user = encoder
        self.encoder_system = copy.deepcopy(encoder)
        
        self.llm_decoder = AutoModelForCausalLM.from_pretrained(llm_path, 
                                                    torch_dtype="auto",
                                                    trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, 
                                                    trust_remote_code=True)
        self.freeze_llm =  freeze_llm
        self.enc_out_dim = enc_out_dim
        self.llm_embed_dim = llm_embed_dim
        self.IGNORE_ID = IGNORE_ID
        self.add_audio_bos_eos = add_audio_bos_eos
        self.add_ctc_prompt_ratio = add_ctc_prompt_ratio
        self.lang_dict = lang_dict
        self.tokenize_ctc_char = tokenize_ctc_char
        self.task_before_audio = task_before_audio
        self.hyp_before_task = hyp_before_task
        self.prompt_finetune = prompt_finetune
        self.add_prompt_before = add_prompt_before
        self.prompt_num = prompt_num
        self.prefix_finetune = prefix_finetune
        self.prefix_num = prefix_num
        self.llm_head_num = llm_head_num
        if num_key_value_heads is None:
            self.num_key_value_heads = llm_head_num
        else:
            self.num_key_value_heads = num_key_value_heads
        self.kv_cache_dim = llm_embed_dim // self.llm_head_num * self.num_key_value_heads
        self.task_type = task_type
        self.freeze_encoder = freeze_encoder
        self.freeze_adpter = freeze_adpter
        self.predict_usr_state = predict_usr_state
        self.chunk_size = chunk_size

        if not hasattr(self.tokenizer, "eod_id"):
            self.tokenizer.eod_id = self.tokenizer.eos_token_id
        if not hasattr(self.llm_decoder, "transformer"):
            self.llm_decoder.transformer = self.llm_decoder.model
            self.llm_decoder.transformer.h = self.llm_decoder.transformer.layers
        if not hasattr(self.llm_decoder.transformer, "wte"):
            self.llm_decoder.transformer.wte = \
                self.llm_decoder.transformer.embed_tokens

        # for chat mode
        if chat_template is not None:
            self.tokenizer.eod_id = self.tokenizer('<|im_end|>'
                                                )['input_ids'][0]
            self.chat_template = {}
            chat_template = chat_template.split('<audio>')
            chat_prefix = chat_template[0].split('<|im_end|>')
            chat_role = chat_prefix[0] + '<|im_end|>'
            self.chat_template['role_prompt'] = self.tokenizer(
                        [chat_role], return_tensors="pt")['input_ids']
            self.chat_template['prefix_for_user_utterance'] = self.tokenizer(
                        [chat_prefix[1]], return_tensors="pt")['input_ids']
            self.chat_template['prefix_for_system_utterance'] = self.tokenizer(
                        [chat_template[1]], return_tensors="pt")['input_ids']
        else:
            self.chat_template = None

        # for CTC prompt
        if self.add_ctc_prompt_ratio > 0.0:
            assert lang_dict is not None
            assert ctc is not None
            self.ctc = ctc.eval()
            if clone_encoder is None:
                self.clone_encoder = copy.deepcopy(encoder)
            else:
                self.clone_encoder = clone_encoder
            self.clone_encoder.eval()
            for (name, param) in self.clone_encoder.named_parameters():
                param.requires_grad = False
            for (name, param) in self.ctc.named_parameters():
                param.requires_grad = False
        else:
            self.clone_encoder = None

        if self.freeze_llm:
            self.llm_decoder.eval()
            for (name, param) in self.llm_decoder.named_parameters():
                param.requires_grad = False
        
        if use_lora:
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=UNET_TARGET_MODULES,
                lora_dropout=args.lora_dropout,
                bias=args.lora_bias,
            )

        if adpter_type == 'cnn':
            self.adpter_user = CNNAdapter(enc_out_dim, llm_embed_dim, kernel_size)
        elif adpter_type == 'linear':
            self.adpter_user = LinearAdapter(enc_out_dim, llm_embed_dim)
        elif adpter_type == 'subsampling':
            self.adpter_user = CNNSubsampling(enc_out_dim, llm_embed_dim, 
                                        kernel_size, activation_func, norm)
        self.adpter_system = copy.deepcopy(self.adpter_user)


        self.task_embeddings = torch.nn.Embedding(task_num, llm_embed_dim)
        if task_type == 'prefix_for_user_utterance':
            self.prefix_embeddings = nn.ModuleList(
                    [
                        torch.nn.ModuleList(
                            [nn.Embedding(task_num, self.kv_cache_dim),
                            nn.Embedding(task_num, self.kv_cache_dim)]
                        )
                        for i in range(len(self.llm_decoder.transformer.h))
                    ]
                )

        if self.prompt_finetune or self.prefix_finetune:
            if self.prompt_finetune:
                self.prompt_embeddings = nn.Embedding(prompt_num, llm_embed_dim)
                self.prompt_ids = torch.Tensor([i for i in range(prompt_num)]).long()
            if self.prefix_finetune:
                self.prefix_embeddings = nn.ModuleList(
                    [
                        torch.nn.ModuleList(
                            [nn.Embedding(prefix_num, self.kv_cache_dim),
                            nn.Embedding(prefix_num, self.kv_cache_dim)]
                        )
                        for i in range(len(self.llm_decoder.transformer.h))
                    ]
                )
                self.prefix_ids = torch.Tensor([i for i in range(prefix_num)]).long()

        if self.freeze_encoder:
            self.encoder_user.eval()
            self.encoder_system.eval()
            for (name, param) in self.encoder_user.named_parameters():
                param.requires_grad = False
            for (name, param) in self.encoder_system.named_parameters():
                param.requires_grad = False

        if self.freeze_adpter:
            self.adpter_user.eval()
            self.adpter_system.eval()
            for (name, param) in self.adpter_user.named_parameters():
                param.requires_grad = False
            for (name, param) in self.adpter_system.named_parameters():
                param.requires_grad = False


        if self.predict_usr_state:
            self.predictor_head = nn.Linear(llm_embed_dim, 4) 
        else:
            self.predictor_head = None

        # define task ids
        self.task_ids = {
            "sot": 0,
            "transcribe": 1,
            "translate": 2,
            "zh": 3,
            "en": 4,
            "audio": 5,
            "/audio": 6,
            "hyps": 7,
            "/hyps": 8,
        }


        ## Warm up will be done in the main service code using data of the same dimensionality as actual data

    def setup_logger(self, parent_logger=None):
        """
        Set up the logger for this class.
        If a parent logger is provided, it will create a child logger.
        Otherwise, it will create a new logger.
        """
        if parent_logger is not None:
            self.logger = parent_logger.getChild(f"AudioLLM_{self.id}")


    def init_template_compilation(self):
        """Pre-compute prompt embeddings"""

        ## Pre-compute chat template embeddings
        self.system_chat_prefix_embeds, self.system_chat_prefix_mask = self.initialize_chat_template_embeds('system')
        self.user_chat_prefix_embeds, self.user_chat_prefix_mask = self.initialize_chat_template_embeds('user')
        

        """Compile performance-critical methods"""

        # # Compile the LLM decoder model
        try:
            self.llm_decoder = torch.compile(
                self.llm_decoder,
                mode="reduce-overhead"
            )
        except Exception as e:
            print(f"Warning: Could not compile LLM decoder: {e}")
        
        # Compile encoders for inference
        try:
            self.encoder_user = torch.compile(
                self.encoder_user,
                mode="reduce-overhead"
            )
            self.encoder_system = torch.compile(
                self.encoder_system,
                mode="reduce-overhead"
            )
        except Exception as e:
            print(f"Warning: Could not compile encoders: {e}")
        
        # Compile adapters
        try:
            self.adpter_user = torch.compile(
                self.adpter_user,
                mode="reduce-overhead"
            )
            self.adpter_system = torch.compile(
                self.adpter_system,
                mode="reduce-overhead"
            )
        except Exception as e:
            print(f"Warning: Could not compile adapters: {e}")

    def initialize_chat_template_embeds(self, identity):

        if self.chat_template is not None:
            ## Apply chat template.
            if identity == 'user':
                chat_prefix_tokens = self.chat_template['prefix_for_user_utterance']
                chat_prefix_tokens = torch.cat((torch.tensor([[self.tokenizer.eod_id]]), chat_prefix_tokens), 1)## chat_prefix_tokens = <|im_end|>\n<|im_start|>user\n
            elif identity == 'system':
                chat_prefix_tokens = self.chat_template['prefix_for_system_utterance']## <|im_end|>\n<|im_start|>assistant\n
            else:
                raise ValueError(f"Unknown identity: {identity}. Must be 'user' or 'system'.")

            chat_prefix_tokens = chat_prefix_tokens.to(self.device)  # Move to CUDA device
            chat_prefix_embeds = self.llm_decoder.transformer.wte(chat_prefix_tokens)
            chat_prefix_mask = torch.full(chat_prefix_tokens.shape, 
                            True).to(self.device)

            return chat_prefix_embeds, chat_prefix_mask

        else:
            return None, None

    def set_system_role(
        self,
        extra_inputs: Optional[dict] = None,
    ):
        # Ensure 'past_key_values' does not exist in extra_inputs, raise an exception if it does
        assert extra_inputs.get('past_key_values', None) is None, "past key values already exist!!!"
        
        # If 'role_prompt' key is present in extra_inputs, use that role as the chat prefix
        if extra_inputs.get('role_prompt', None) is not None:
            ## <|im_start|>system\n[PROMPT CONTENT]
            
            chat_prefix = self.tokenizer([extra_inputs['role_prompt']], 
                return_tensors="pt")['input_ids'].to(self.device)  # Convert role to tokens and move to CUDA device
        else:
            # If no 'role_prompt' is provided, use the default chat template and remove the last token (<|im_end|>)
            chat_prefix = self.chat_template['role_prompt'][:, :-1].to(self.device)
        
        # Use the LLM decoder's word embedding layer to convert the chat prefix into embeddings
        inputs_embeds = self.llm_decoder.transformer.wte(chat_prefix)
        
        # Create an attention mask with the same shape as the chat prefix, all values set to True
        attention_mask = torch.full(chat_prefix.shape, 
                            True).to(inputs_embeds.device) 
        
        # Prepare the input dictionary containing embeddings and attention mask
        inputs = {
                'inputs_embeds': inputs_embeds.half(),  # Convert embeddings to half precision floats
                'attention_mask': attention_mask,
            }

        # Call the _generate_one_step method to generate one step output, including past_key_values, etc.
        # The main purpose here is to get the system role and prompts encoded.
        _, past_key_values, _ = self._generate_one_step(
                                                copy.deepcopy(inputs), "dialog_sl")
                                                
        # Return the generated past_key_values
        return past_key_values

    def recognize(
        self,
        speech: torch.Tensor,
        extra_inputs: Optional[dict] = None,
    ):
        ## At the beginning, past_key_values will only contain system role/prompt
        assert extra_inputs.get('past_key_values', None) is not None, "must set system role first!!!"

        identity = extra_inputs['identity']
        status = extra_inputs['status']

        buffer = extra_inputs.get('encoder_cache', None)
        cnn_cache = extra_inputs.get('adapter_cache', None)
        pe_index = extra_inputs.get('pe_index', 0)


        # 1. Identity-Based Selection of Encoder/Adapter
        if identity == 'user':
            current_encoder = self.encoder_user
            current_adapter = self.adpter_user
        else: # 'system'
            current_encoder = self.encoder_system
            current_adapter = self.adpter_system


        # 2. Process audio through selected Encoder and Adapter
        # Encoder
        if buffer is None:
            buffer = [None] * current_encoder.enc[1].num_blocks
        
        encoder_out, buffer, _, _, pe_index = current_encoder.infer(speech, buffer, 
                                                                0, None, pe_index)
        encoder_mask = torch.full(encoder_out.shape[:2], True).unsqueeze(1
                                                        ).to(encoder_out.device)

        # Adapter
        inputs_embeds, encoder_mask, cnn_cache = current_adapter(encoder_out, encoder_mask, 
                                        cache=cnn_cache, return_cache=True) # 1, T, D
        attention_mask = encoder_mask.squeeze(1) # 1, T


        # 3. Some identity based notion
        if identity == 'user':
            do_prediction = True
            chat_prefix_embeds = self.user_chat_prefix_embeds
            chat_prefix_mask = self.user_chat_prefix_mask
        elif identity == 'system':
            do_prediction = False
            chat_prefix_embeds = self.system_chat_prefix_embeds
            chat_prefix_mask = self.system_chat_prefix_mask
        else:
            raise ValueError(f"Unknown identity: {identity}. Must be 'user' or 'system'.")

        ## Apply chat template
        if self.chat_template is not None and status == 'ipu_sl':
            inputs_embeds = torch.cat((chat_prefix_embeds, inputs_embeds), 1)
            attention_mask = torch.cat((chat_prefix_mask, attention_mask), 1)

        # 4. Prepare inputs for the LLM
        inputs = {
            'inputs_embeds': inputs_embeds.half(),
            'attention_mask': attention_mask,
        }


        # Add kv cache, i.e., context from previous steps integrating both user input and system output
        inputs['past_key_values'] = extra_inputs['past_key_values']
        past_mask = torch.full([1, inputs['past_key_values'][0][0].size(2)],True).to(self.device)
        attention_mask = torch.cat((past_mask, attention_mask), 1)
        inputs['attention_mask'] = attention_mask

        ## This is where we actually run forward inference.
        prediction_probs, past_key_values, _ = self._generate_one_step(
                                                    ## No need to deepcopy here
                                                    # copy.deepcopy(inputs), 
                                                    inputs,
                                                    do_prediction=do_prediction
                                                )
                                                
        return prediction_probs, past_key_values, cnn_cache, buffer, pe_index
    
    def _post_decode(self, output, temperature=1.0, top_k=0, top_p=0.0):
        """
        Decoding function, based on the posterior probability output, 
        uses top_k, top_p, and temperature parameters for sampling.

        Parameters:
        - output: torch.Tensor, shaped as (1, 1, D), represents the posterior probability output by the model.
        - top_k: int, indicates selecting the top k tokens with the highest probability for sampling.
                      If 0, no top_k filtering is performed.
        - top_p: float, indicates selecting tokens with cumulative probability not exceeding p for sampling.
                        If 0.0, no top_p filtering is performed.
        - temperature: float, represents the sampling temperature parameter. 
                              The higher the value, the more random the sampling; 
                            the lower the value, the more deterministic the sampling.

        Returns:
        - Selected token index.
        """
        output = output.squeeze(0).squeeze(0)

        # temperature
        if temperature != 1.0:
            output = output / temperature

        probs = torch.nn.functional.softmax(output, dim=-1)

        # top_k
        if top_k > 0:
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            probs = torch.zeros_like(probs).scatter_(0, top_k_indices, top_k_probs)
            probs = probs / probs.sum()

        # top_p
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            if sorted_indices_to_remove[0]:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / probs.sum()

        token_index = torch.multinomial(probs, 1)
        return token_index.unsqueeze(0)
    
    def _llm_forward_core(self, input_context: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Core LLM forward pass - optimized for compilation"""
                
        outputs = self.llm_decoder.model(**input_context)

        return outputs['last_hidden_state'], outputs['past_key_values']
    
    def _prediction_head_forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Separate prediction head forward pass"""
        predictor_output = self.predictor_head(hidden_state)
        state_logits = predictor_output[0, :]
        prob = torch.nn.functional.softmax(state_logits[:, :-1], dim=-1)
        state_prob = prob[-1].clone()
        
        return state_prob

    def _generate_one_step( self, inputs, do_prediction):
        """
        Generates the model's next output based on the current input and state.

        Parameters:
        - inputs: The input tensor containing the model's input data.
        - do_prediction: whether to perform prediction or just update the context.

        Returns:
        - prediction_probs: A dictionary with state probabilities if identity is 'user', otherwise None.
        - past_key_values: The model's historical key-value pairs.
        - hidden_state: The model's last hidden state.
        """


        # #Note that this forward pass automatically update the past_key_values to be the most updated, i.e., the state used in this decoding step
        last_hidden_state, new_past_key_values = self._llm_forward_core(inputs)


        prediction_probs = None
        if do_prediction and self.predictor_head is not None:

            # Use compiled prediction head
            state_prob = self._prediction_head_forward(last_hidden_state)

            # Extract probabilities (this part stays uncompiled for flexibility)
            prediction_probs = {
                "state_1": state_prob[1].item(),
                "state_2": state_prob[2].item()
            }

        # We only update context (past_key_values), no token generation. If it's not user input, we just return the past_key_values but do not return prediction_probs.
        return prediction_probs, new_past_key_values, last_hidden_state