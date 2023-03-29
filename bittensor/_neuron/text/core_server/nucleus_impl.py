import argparse
import math
import bittensor
import torch
from torch import nn
import torch.nn.functional as F
from types import SimpleNamespace
from typing import Tuple, Optional

import transformers
from transformers import AutoModel,AutoTokenizer,AutoConfig, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence
from bittensor.utils.tokenizer_utils import prep_tokenizer, get_translation_map, translate_logits_to_probs_std, \
    translate_special_token_text, pad_offsets, topk_token_phrases, compact_topk_token_phrases

from loguru import logger; logger = logger.opt(colors=True)

class server(torch.nn.Module):
    def __init__(self, 
                config: 'bittensor.config' = None,
                pretrained: bool = None,
                model_name: str = None,
                padding: bool =None, 
                interpolate: bool =None,
                inter_degree: str = None,
                model = None,
                tokenizer = None,
                mapping_function = None,
                token_remap = None,
                checking= None):
        r"""" Creates a server that serves up a pretrained miner on the bittensor network
        Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.server.config()
                pretrained (:obj:bool , `optional`):
                    if the model should pretrained or not
                model_name (:obj:string , `optional`):
                    name of the pretrained model from huggingface to use
                padding (:obj:bool, `optional`):
                    If the server should pad out to match the hidden units that the bittensor network is using
                    If set to False, it will instead create a mapping layer to do the same thing.
                interpolate (:obj:bool, `optional`):
                    If the server should interpolate between sequence length differences.
                    If set to false, there should be a mapping function that takes care of the differnces
                inter_degree (:obj:str, `optional`):
                    The Interpolate algorithm (nearest | linear | bilinear | bicubic | trilinear | area)
                model (:obj:torch.module, `optional`):
                    Overrides the huggingface pretrained model with your own pretrained model
                tokenizer (:obj:huggingface.tokenizer, `optional`):
                    Overrides the huggingface tokenizer with your tokenizer
                mapping_function (:obj:Callable, `optional`):
                    Custom mapping function that maps between sequence length differences between tokenizers
                token_remap (:obj:Callable, `optional`):
                    Custom function that maps between tokenizers (defaults to self.remapping_token)
        """
        super(server, self).__init__()
        if config == None: config = server.config()
        self.config = config;print(config)
        self.std_tokenizer = bittensor.tokenizer()
        self.device = config.neuron.device

        #setting up pretrained model
        self.model_name = model_name if model_name != None else config.neuron.model_name
        self.pretrained = pretrained if pretrained != None else config.neuron.pretrained
        if self.pretrained == True:
            self.pre_model = model if model != None else AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = tokenizer
            if tokenizer is None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                except ValueError:  # when fast not available as in https://github.com/huggingface/tokenizers/pull/1005
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        elif self.pretrained == False:
            model_config = AutoConfig.from_pretrained(self.model_name)
            model_config.vocab_size= bittensor.__vocab_size__
            self.pre_model = model if model != None else AutoModel.from_config(model_config)
            self.tokenizer = bittensor.tokenizer()

        # Define PAD Token = EOS Token (GPT2 generate convention, when PAD Token is None)
        # https://github.com/huggingface/transformers/blob/49c8c67fb815a277405f84dea4a66353e19fb347/tests/models/gpt2/test_modeling_gpt2.py#L532
        if self.pre_model.config.pad_token_id is None and self.pre_model.config.eos_token_id is not None:
            self.pre_model.config.pad_token_id = self.pre_model.config.eos_token_id

        self.tokenizer = prep_tokenizer(self.tokenizer, self.std_tokenizer)
        self.to_translation_map = get_translation_map(self.tokenizer, self.std_tokenizer)
        self.from_translation_map = get_translation_map(self.std_tokenizer, self.tokenizer)
        self.split_map_cache = {}

        if self.config.neuron.local_train or self.config.neuron.remote_train:
            self.pre_model.train()
            self.set_fine_tuning_params()

        else:
            self.pre_model.eval()

        if self.config.neuron.autocast and self.device[:4] == 'cuda':
            self.pre_model.half()

        #parameters of the models
        self.final_dim =  bittensor.__network_dim__
        self.pre_dimension = self.pre_model.config.hidden_size
        self.padding = padding if padding != None else config.neuron.padding
        self.interpolate = interpolate if interpolate != None else config.neuron.interpolate
        self.inter_degree = inter_degree if inter_degree != None else config.neuron.inter_degree
        self.checking = checking if checking != None else config.neuron.checking
        self.mapping_function= mapping_function
        self.token_remap = token_remap if token_remap is not None else self.remapping_token

        if self.config.neuron.padding == False:
            self.mapping = torch.nn.Linear( self.pre_dimension, self.final_dim)

        self.decoder = torch.nn.Linear( self.final_dim, bittensor.__vocab_size__ , bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
        self.outputs_cache = None
        self.gradients_cache = None
        self.best_loss = math.inf
        self.best_remote_loss = math.inf

        #checking if the parameters of the server makes sense
        if self.checking and pretrained == True:
            self.check()

        # -- keeps track of gradients applied
        self.backward_gradients_count = 0 
        self.remote_losses = [] 

    def set_fine_tuning_params(self) -> Tuple[bool, str]:
        r''' Set to tune only the parameter of the last layer
            Returns: 
                reached_last_layer (:type:`bool`):
                    If we have set partial of the model to requires grad.
                
                last_layer_name (:type:`string`):
                    The name of the last layer that user specified or we found.
                    None if the user did not specify and we couldnt find it. 
        '''
        def find_last_layer(model: torch.nn.Module) -> Optional[str]:    
            r''' Recursively find the last layer in a nn.ModuleList
                Args:
                    model (:obj:`torch.module`):
                        The model (or sub-model) to fine the last layer from. 
                Returns:
                    name (:type:`str`):
                        The name (or sub-name) of the last layer.
                        None if not found
            '''
            reverted_child_list = [(name, child) for name, child in model.named_children()]
            reverted_child_list.reverse()

            for name, child in reverted_child_list:    
                if isinstance(child, nn.ModuleList):
                    if self.config.neuron.finetune.num_layers > len(child):
                        logger.warning(f'Number of finetune layers was set higher then the layers avaliable {len(child)}')
                        return None
                    return (name + '.' +str(len(child) - self.config.neuron.finetune.num_layers))
                
            for name, child in reverted_child_list:    
                name_ = find_last_layer(child)
                if name_ != None:
                    return (name+'.'+ name_)

            return None     

        if self.config.neuron.finetune.layer_name == None:
            last_layer_name = find_last_layer(self.pre_model)
        else:
            last_layer_name = self.config.neuron.finetune.layer_name

        reached_last_layer = False

        # set the non-last layer parameters not to require grads
        if (self.config.neuron.finetune.all) or (last_layer_name == None):
            return False, last_layer_name

        logger.success(f'Set to finetune layer {last_layer_name} and onwards')
        
        for name, param in self.pre_model.named_parameters():
            if last_layer_name in name or reached_last_layer == True:
                param.requires_grad = True
                reached_last_layer = True
            else:
                param.requires_grad = False

        if reached_last_layer == False:
            if self.config.neuron.finetune.all:
                logger.warning('Set to finetune the whole model, this will significantly increase the memory usage.')
            else:
                logger.warning(f'Cannot identify the last layer of the model with name {last_layer_name}, setting to finetune on all of the parameters.')

        return reached_last_layer, last_layer_name

    def remapping_token(self, token_batch, std_tokenizer=None, return_offsets_mapping=False):
        r""" Tokenizer remapping; decodes the message and then remaps the message using a new tokenizer
            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    token_batch to be retokenized, [batch_size, sequence_len]
                std_tokenizer ( :obj:`transformers.Tokenizer`, `optional`):
                    The standard tokenizer which was used to tokenize the input.
                return_offsets_mapping ( :obj:`bool`, `required`):
                    Return offsets_mapping in tokenization to delineate token segment positions.
        """
        if std_tokenizer is None:
            std_tokenizer = self.std_tokenizer

        text_batch = std_tokenizer.batch_decode(token_batch)  # decode tokens to original text
        result = translate_special_token_text(text_batch, std_tokenizer, self.tokenizer)  # translate special tokens
        to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch = result

        tokens = self.tokenizer(to_text_batch, padding=True, truncation=True, max_length=token_batch.size(1), return_tensors='pt',
                                add_special_tokens=False).to(self.device)  # assume tokenizer.padding_side = 'left'

        if return_offsets_mapping:  # get offsets_mapping in tokenization to delineate token segment positions
            server_tokens = self.tokenizer(to_text_batch, return_offsets_mapping=True, add_special_tokens=False)
            std_tokens = std_tokenizer(text_batch, return_offsets_mapping=True)  # encode again to get offsets mapping

            # pad offsets so that special token offset widths match for continued correct alignment
            tokens['offset_mapping'] = pad_offsets(server_tokens['offset_mapping'], to_offsets_batch, pad_offsets_batch)
            tokens['offset_mapping_std'] = pad_offsets(std_tokens['offset_mapping'], from_offsets_batch,
                                                       pad_offsets_batch)
        return tokens

    def forward(self, inputs, tokenizer=None):
        """
            Forward pass through the whole server model. Returns the loss and decoded predictions.

            Args:
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                tokenizer (:obj:'huggingface.tokenizer', optional):
                    The tokenizer which was used to tokenize the inputs
             Returns:
                loss (:obj:`torch.FloatTensor`):
                    MLM loss from the inputs
                decoded_targets (:obj:`torch.FloatTensor`):
                    Decoded predictions of the next token in the sentence.

        """
        message, model_output, decoded_targets = self.local_forward(inputs, tokenizer)
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )

        return loss, decoded_targets

    def local_forward(self, token_batch, tokenizer=None, encode_len=bittensor.__network_dim__, model_output = None):
        r""" Forward pass through the pretrained model and possible mappings between hidden units.
             The response tensor should be the hidden units computed using the local context and
             with shape: [batch_size, sequence_len, __vocab_size__].

            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    torch inputs to be forward processed, [batch_size, sequence_len]
                tokenizer ( huggingface.tokenizer, `optional`):
                    The tokenizer which was used to tokenize the inputs
                encode_len ( :obj:`int`, `optional`):
                    logit encoding length, default bittensor.__network_dim__ length
                model_output (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `optional`):
                    The output of huggingface auto model.

            Returns:
                model_outputs (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `required`):
                    The output of huggingface auto model.
                
                logits (:obj:`torch.FloatTensor`):
                    The nucleus's logit outputs as a torch tensor of shape [batch_size, sequence_len, __vocab_size__]
        """
        tokens = self.token_remap(token_batch, std_tokenizer=tokenizer, return_offsets_mapping=True)  # remap to server tokenizer
        if model_output == None:
            if self.config.neuron.local_train:
                model_output = self.pre_model(input_ids=tokens['input_ids'],
                                                attention_mask=tokens['attention_mask'],
                                                output_hidden_states=True)
            else:
                with torch.no_grad():
                    model_output = self.pre_model(input_ids=tokens['input_ids'],
                                                    attention_mask=tokens['attention_mask'],
                                                    output_hidden_states=True)

            self.model_output_check(model_output)
        return None, model_output, model_output.logits
    
    def encode_forward(self,inputs,tokenizer=None, model_output = None):
        r""" Forward pass through the pretrained model and possible mappings between hidden units. 
             The response tensor should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                tokenizer ( huggingface.tokenizer, `optional`):
                    The tokenizer which was used to tokenize the inputs
                model_outputs (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `optional`):
                    The output of huggingface auto model.

            Returns:
                model_outputs (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `required`):
                    The output of huggingface auto model.
                    
                encoded_hidden (:type:`torch.Tensor`, `required`)
                    The hidden layer output as a torch tensor of shape [batch_size, sequence_len, __network_dim__ ]
        """
        transformers.set_seed(0)
        transformers.enable_full_determinism(0)

        sen_len = inputs.size()
        tokens = self.token_remap(inputs, tokenizer)  # remap to server tokenizer

        if model_output == None:
            if self.config.neuron.remote_train:
                model_output = self.pre_model(input_ids=tokens['input_ids'],
                                                attention_mask=tokens['attention_mask'],
                                                output_hidden_states=True)
            else:
                with torch.no_grad():
                    model_output = self.pre_model(input_ids=tokens['input_ids'],
                                                    attention_mask=tokens['attention_mask'],
                                                    output_hidden_states=True)

        self.model_output_check(model_output)
        pre_hidden = model_output.hidden_states[-1]

        if self.interpolate and sen_len[1] != pre_hidden.size()[1]:
            down= F.interpolate(pre_hidden.unsqueeze(1),size=[sen_len[1],pre_hidden.size()[2]],mode=self.inter_degree).squeeze(1)
        elif self.mapping_function:
            down = self.mapping_function(pre_hidden)
        else:
            down = pre_hidden

        if self.padding:
            padding_l = (self.final_dim-self.pre_dimension)//2
            padding_r = (self.final_dim-self.pre_dimension) - padding_l
            encoded_hidden = F.pad(down, (padding_l, padding_r),  "constant", 0)
        else:
            encoded_hidden = self.mapping(down)

        return None, model_output, encoded_hidden

    def encode_forward_causallm(self, token_batch, tokenizer=None, encode_len=bittensor.__network_dim__, model_output=None):
        r""" Forward pass through the pretrained model and possible mappings between hidden units.
             The response tensor should be the hidden units computed using the local context and
             with shape: [batch_size, sequence_len, __vocab_size__].

            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    torch inputs to be forward processed, [batch_size, sequence_len]
                tokenizer ( huggingface.tokenizer, `optional`):
                    The tokenizer which was used to tokenize the inputs
                encode_len ( :obj:`int`, `optional`):
                    logit encoding length, default bittensor.__network_dim__ length
                model_output (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `optional`):
                    The output of huggingface auto model.

            Returns:
                model_outputs (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `required`):
                    The output of huggingface auto model.
                
                logits_std (:obj:`torch.FloatTensor`):
                    The nucleus's logit outputs as a torch tensor of shape [batch_size, sequence_len, __vocab_size__]
        """
        transformers.set_seed(0)
        transformers.enable_full_determinism(0)

        tokens = self.token_remap(token_batch, std_tokenizer=tokenizer, return_offsets_mapping=True)  # remap to server tokenizer

        def _forward(_model_output=model_output):
            if _model_output is None:
                # transformer models like gerpt2 typically perform worse with left-side attention mask, so turning it off
                _model_output = self.pre_model(input_ids=tokens['input_ids'],
                                                #attention_mask=tokens['attention_mask'],
                                               output_hidden_states=True)
                self.model_output_check(_model_output)
            pre_logits = _model_output.logits  # [batch_size, sequence_len, self.tokenizer.vocab_len]
            probs_std = translate_logits_to_probs_std(pre_logits,
                                                      tokens['offset_mapping'], tokens['offset_mapping_std'],
                                                      self.tokenizer, self.std_tokenizer,
                                                      self.split_map_cache,
                                                      self.to_translation_map, self.from_translation_map,
                                                      tokens['input_ids'], token_batch)
            probs_std = probs_std.to(self.device)
            logits_std = torch.log(probs_std + 1e-40)
            #removing the loss calculation for stablity testing
            original_loss = self.get_loss_fct(pre_logits, tokens['input_ids']).item()
            translated_loss = self.get_loss_fct(logits_std, token_batch).item()
            message = f'Loss: {original_loss:.2f} → {translated_loss:.2f}'
            
            return message, _model_output, logits_std

        if self.config.neuron.remote_train:
            return _forward()  # track gradients for training

        with torch.no_grad():
            return _forward()  # no gradients

    def encode_forward_causallmnext(self, token_batch, std_tokenizer=None, topk: int = 4096, model_output=None):
        r"""
        Forward pass through the pretrained model and select topk tokenizer logits and retokenize with std_tokenizer,
        then compact new token phrases and probabilities
        into 1-D tensor [ >= batch_size * (2 * topk + 1)] prob + at least 1 token per phrase + floor_prob.
        The floor probability is the mean probability of token phrases not captured in topk, required since
        the server tokenizer vocab_size may not be known to the receiver/validator.

            Args:
                token_batch ( :obj:`torch.LongTensor`, `required`):
                    torch inputs to be forward processed, [batch_size, std_sequence_len].
                std_tokenizer ( :obj:`PreTrainedTokenizerBase`, `optional`):
                    The standard tokenizer which was used to tokenize the inputs.
                topk ( :obj:`int`, `optional`):
                    Amount of std_tokenized server phrases with highest probability to produce.
                model_output (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `optional`):
                    The output of transformers AutoModel.

            Returns:
                model_outputs (:obj:`transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`, `required`):
                    The output of transformers AutoModel.
                topk_tensor (:obj:`torch.Tensor`, `required`):
                    [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                    in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                    Content structure:
                    [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                      [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                      [...],
                      [prob_floor_b=0, ignore_index, ..., ignore_index]],
                     [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                      [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                      [...],
                      [prob_floor_b=1, ignore_index, ..., ignore_index]],
                     [...]]
        """
        transformers.set_seed(0)
        transformers.enable_full_determinism(0)
        
        if std_tokenizer is None:
            std_tokenizer = self.std_tokenizer

        tokens = self.token_remap(token_batch, std_tokenizer)

        def _forward(_model_output=model_output):
            if _model_output is None:
                _model_output = self.pre_model(input_ids=tokens['input_ids'],
                                               attention_mask=tokens['attention_mask'],
                                               output_hidden_states=True)
                self.model_output_check(_model_output)
            # model_output.logits: [batch_size, sequence_len, server_vocab_size]
            last_logits = _model_output.logits[:, -1, :]  # [batch_size] server prediction of continuation, right-aligned

            # Select topk tokenizer logits and retokenize with std_tokenizer,
            # then compact new token phrases and probabilities into 1-D tensor
            topk_tensor = topk_token_phrases(last_logits, self.tokenizer, topk=topk)  # [batch_size, (topk + 1), max_len]

            original_loss = self.get_loss_fct(_model_output.logits, tokens['input_ids']).item()

            message = f'Loss: {original_loss:.2f}'

            _model_output.loss = original_loss
            return message, _model_output, topk_tensor

        if self.config.neuron.remote_train:
            return _forward()  # track gradients for training

        with torch.no_grad():
            return _forward()  # no gradients

    def model_output_check(self, model_output: transformers.modeling_outputs.CausalLMOutputWithPast):
        """
            Verify the model has been ran correctly with valid output.

            Args:
                model_output (:obj:`transformers.modeling_outputs.CausalLMOutputWithPast`, required):
                    The output of transformers AutoModel.

            Returns:
                check_status (:type: `bool`):
                    True if the model_output is valid.
        """
        if hasattr(model_output, 'hidden_states') and model_output.hidden_states[-1].isnan().sum() > 0:
            raise ValueError("Got nan value from model last hidden state. If you are using cuda with autocast, try remove setting --neuron.autocast.")

        if model_output.logits.isnan().sum() > 0:
            raise ValueError("Got nan value from model logits. If you are using cuda with autocast, try remove setting --neuron.autocast.")

        return True

    def get_loss_fct(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Calculate loss_fct, CausalLM loss, next-token prediction loss.
            Args:
                logits (:obj:`torch.FloatTensor`, `required`):
                    [batch_size, sequence_len, bittensor.__network_dim__]
                labels (:obj:`torch.LongTensor`, `required`):
                    [batch_size, sequence_len]

            Returns:
                loss (:obj:`torch.FloatTensor`):
                    scalar
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def check(self):
        r"""Checks the server settings
        """
        assert self.tokenizer.name_or_path == self.pre_model.name_or_path, 'incorrect model ({}) and tokenizer ({})'.format(self.pre_model.name_or_path,self.tokenizer.name_or_path)
        if self.interpolate == False:
            assert self.mapping_function != None, 'Incorrect Settings; needs atleast one mapping function for sequence length changes'

    def save(self, path):
        try:
            state_dict = {
                'model': self.pretrained,
                'pretrained_model': self.pre_model.state_dict(), 
                'decoder': self.decoder.state_dict(),
                'best_loss': self.best_loss,
                'best_remote_loss': self.best_remote_loss,
            }
            if self.padding == False:
                state_dict['mapping'] = self.mapping.state_dict()
            torch.save( state_dict, "{}/model.torch".format( path) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def load(self, path):
        try:
            state_dict=  torch.load("{}/model.torch".format( path ))
            if self.pretrained == state_dict['model']:
                self.pre_model.load_state_dict(state_dict['pretrained_model'], strict=False)
                self.decoder.load_state_dict(state_dict['decoder'])
                if self.padding == False:
                    self.mapping.load_state_dict(state_dict['mapping'])
                self.best_loss = state_dict['best_loss']
                self.best_remote_loss = state_dict['best_remote_loss']
                bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( path ))


        except Exception as e:
            logger.warning('No saved model found with error: {}', e)

    @staticmethod
    def config ():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')

        # ML model arguements
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.01)
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.model_name', type=str, help='pretrained model from hugging face',default='gpt2')
        parser.add_argument('--neuron.pretrained', action='store_false', help='if the model should be pretrained',default=True)
        parser.add_argument('--neuron.padding', action='store_false', help='To pad out final dimensions',default=True)
        parser.add_argument('--neuron.interpolate', action='store_false', help='To interpolate between sentence length',default=True)
        parser.add_argument('--neuron.inter_degree', type=str, help='Interpolate algorithm (nearest | linear | bilinear | bicubic | trilinear | area)', default='nearest')
        parser.add_argument('--neuron.autocast',  action='store_true', help='(experimental) autocasts the model to float16. Must require cuda', default=False)
        parser.add_argument('--neuron.local_train', action='store_true', help='''If true, allow local training''', default=False)
        parser.add_argument('--neuron.remote_train', action='store_true', help='''If true, allow remote training''', default=False)
        parser.add_argument('--neuron.finetune.all', action='store_true', help='Finetune your whole model instead of only on the last (few) layers', default=False)
        parser.add_argument('--neuron.finetune.num_layers', type=int, help='The number of layers to finetune on your model.', default=1)
        parser.add_argument('--neuron.finetune.layer_name', type=str, help='Specify since which layer to finetune. eg. encoder.layer.11', default=None)
        
        # Miner arguements
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_server')
        parser.add_argument('--neuron.checking', action='store_false', help='To check if server settings are correct',default=True)
        parser.add_argument('--neuron.restart', action='store_true', help='If True, train the neuron from the beginning', default=False)
        parser.add_argument('--neuron.no_set_weights', action='store_true', help='If True, the model does not set weights.', default=False)
        parser.add_argument('--neuron.blacklist.stake', type=float, help='Amount of stake (tao) in order not to get blacklisted', default=10)
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch', default=10)
        parser.add_argument('--neuron.blacklist.time', type=int, help='how often a peer can query you (seconds) ', default=1)
        parser.add_argument('--neuron.blocks_per_set_weights', type=float, help='how often to set weights', default=-1)
        parser.add_argument('--neuron.metagraph_sync', type=float, help='how often to sync the metagraph', default=100000)
        parser.add_argument('--neuron.blacklist_allow_non_registered', action='store_true', help='''If true, allow non-registered peers''', default=False)
        parser.add_argument('--neuron.disable_blacklist', action='store_true', help='Turns off blacklisting', default=False)
        parser.add_argument('--neuron.disable_priority', action='store_true', help='Turns off priority threadpool', default=False)
        parser.add_argument('--neuron.num_remote_loss', type=int, help='Number of past remote loss to keep in stat.', default=20)
        parser.add_argument('--neuron.max_batch_size', type=int, help='The maximum batch size for forward requests.', default=-1)
        parser.add_argument('--neuron.max_sequence_len', type=int, help='The maximum sequence length for forward requests.', default=-1)
        parser.add_argument('--neuron.blacklist.hotkeys', type=str, required=False, nargs='*', action='store', help='To blacklist certain hotkeys', default=[])

        # Synapse Arguements
        parser.add_argument('--neuron.lasthidden', action='store_false', help='To turn off last hidden synapse', default=True)
        parser.add_argument('--neuron.causallm', action='store_false', help='To turn off causallm synapse', default=True)
        parser.add_argument('--neuron.causallmnext', action='store_false', help='To turn off causallmnext synapse', default=True)
        parser.add_argument('--neuron.seq2seq', action='store_false', help='To turn off seq2seq synapse', default=True)
        parser.add_argument('--neuron.lasthidden_stake', type = float, help='the amount of stake to run last hidden synapse',default=0)
        parser.add_argument('--neuron.causallm_stake',  type = float, help='the amount of stake to run causallm synapse',default=0)
        parser.add_argument('--neuron.causallmnext_stake', type=float, help='the amount of stake to run causallmnext synapse', default=0)
        parser.add_argument('--neuron.seq2seq_stake',  type = float, help='the amount of stake to run seq2seq synapse',default=0)

        # Netuid Arg
        parser.add_argument('--netuid', type=int , help='Subnet netuid', default=1)

        bittensor.wallet.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.prioritythreadpool.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )
