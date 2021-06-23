'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''

import argparse
import random
import torch
from torch import nn
import torch.nn.functional as F
import transformers

from typing import Callable
from transformers import BertModel, BertConfig
from types import SimpleNamespace

import bittensor

class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertNucleusBase(torch.nn.Module):
    def __init__(self, routing_callback, config: 'bittensor.Config', **kwargs):
        r""" Init a new base-bert nucleus.

            Args:
                config (:obj:`bittensor.Config`, `required`): 
        """
        super(BertNucleusBase, self).__init__()
        if config == None:
            config = BertNucleusBase.config()
        BertNucleusBase.check_config(config)
        self.config = config

        # To be set.
        self.routing_callback = routing_callback

        # Hugging face config item.
        huggingface_config = BertConfig(    vocab_size=bittensor.__vocab_size__, 
                                            hidden_size=bittensor.__network_dim__, 
                                            num_hidden_layers=config.nucleus.num_hidden_layers, 
                                            num_attention_heads=config.nucleus.num_attention_heads, 
                                            intermediate_size=bittensor.__network_dim__, 
                                            is_decoder=False)

        # encoder_layer: encodes tokenized sequences to network dim.
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.transformer = BertModel( huggingface_config, add_pooling_layer=True )

        # hidden_layer: transforms context and encoding to network_dim hidden units.
        # [batch_size, sequence_dim, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.hidden_layer = torch.nn.Linear( bittensor.__network_dim__, bittensor.__network_dim__ )

        # pooling_layer: transforms teh hidden layer into a pooled representation by taking the encoding of the first token
        # [batch_size, sequence_dim,  bittensor.__network_dim__] -> [batch_size, bittensor.__network_dim__]
        self.pooler = BertPooler( huggingface_config )

        self.to(self.device)

    @staticmethod   
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser(); 
        BertNucleusBase.add_args(parser) 
        config = bittensor.config( parser ); 
        return config

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):    
        r""" Add custom params to the parser.
        """
        parser.add_argument('--nucleus.num_hidden_layers', default=2, type=int, 
                            help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--nucleus.num_attention_heads', default=2, type=int, 
                            help='Number of attention heads for each attention layer in the Transformer encoder.')
        parser.add_argument('--nucleus.n_block_filter', default=100, type=int, help='Stale neurons are filtered after this many blocks.')

    @staticmethod
    def check_config( config: 'bittensor.Config' ):    
        r""" Add custom checks to the config.
        """
        pass

    def attach_routing_callback(self, routing_callback: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ] ):
        """ Assigns the routing_callback call to this neuron.

            Returns:
                routing_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                    Routing function to call on self.route()
        """
        self.routing_callback = routing_callback

    @property
    def route( self, inputs: torch.Tensor, query: torch.Tensor ) -> torch.FloatTensor:
        """ Calls this nucleus's subscribed routing function. self.routing_callback must be set before this call is made.

        Args:
            inputs (:obj:`torch.LongTensor` of shape :obj:`( batch_size, sequence_len )`, `required`): 
                    Batch_size length list of tokenized sentences.

            query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, query_dimension)`, `required`): 
                    Context tensor used to select which neurons to query for each example.
            
        Returns:
            remote_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                joined responses from network call.
        """
        if self.routing_callback == None:
            raise RuntimeError('The routing function must be set on this nucleus before a remote_forward call can execute.')
        else:
            return self.routing_callback( inputs = inputs, query = query )

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the BERT NSP Nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.base_local_forward( inputs=inputs ).local_hidden
        return hidden

    def base_local_forward(self, inputs: torch.LongTensor, attention_mask: torch.LongTensor = None):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `optional`): 
                    Mask to avoid performing attention on padding token indices.
                    Mask values selected in ``[0, 1]``:
                        - 1 for tokens that are **not masked**,
                        - 0 for tokens that are **maked**.    

             SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_pooled (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Local hidden state pooled by returning the encoding of the first token.
                }
        """        
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        # Return vars to be filled.
        output = SimpleNamespace()
   
        # local_context: distilled version of remote_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.transformer( input_ids = inputs, return_dict = True, attention_mask = attention_mask ).last_hidden_state

        # local_hidden: hidden layer encoding of sequence using local context
        # local_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_hidden = self.hidden_layer( output.local_context )
        output.local_pooled = self.pooler( output.local_hidden )

        return output

    def base_remote_forward(self, inputs: torch.LongTensor, attention_mask: torch.LongTensor = None):
        """Forward pass inputs and labels through the remote BERT networks.

        Args:
            inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.                

            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `optional`): 
                    Mask to avoid performing attention on padding token indices.
                    Mask values selected in ``[0, 1]``:
                        - 1 for tokens that are **not masked**,
                        - 0 for tokens that are **maked**.        

        Returns:
            SimpleNamespace ( 
                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    router (:obj:`SimpleNamespace`, `required`): 
                        Outputs from the pkm dendrite.
                )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        output = self.base_local_forward( inputs = inputs, attention_mask = attention_mask )

        # remote_context: joined responses from a bittensor.forward_text call.
        # remote_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_context = self.route( inputs = inputs, query = output.local_pooled )

        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss( output.local_context, output.remote_context.detach() )

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_hidden = self.hidden_layer( output.remote_context )
        output.remote_pooled = self.pooler( output.remote_hidden )

        return output

class BertNSPNucleus (BertNucleusBase):
    def __init__( self, config: 'bittensor.Config', **kwargs):
        r""" Init a new bert nsp nucleus module.

            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    BertNSP configuration class.
        """
        super(BertNSPNucleus, self).__init__(config = config, **kwargs)

        # Hugging face config item.
        huggingface_config = BertConfig(    vocab_size=bittensor.__vocab_size__, 
                                            hidden_size=bittensor.__network_dim__, 
                                            num_hidden_layers=config.nucleus.num_hidden_layers, 
                                            num_attention_heads=config.nucleus.num_attention_heads, 
                                            intermediate_size=bittensor.__network_dim__, 
                                            is_decoder=False)
        
        # target_layer: maps from hidden layer to vocab dimension for each token. Used by MLM loss.
        # [batch_size, sequence_len, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__vocab_size__]
        self.target_layer = transformers.models.bert.modeling_bert.BertOnlyNSPHead( huggingface_config )

        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the BERT NSP Nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.local_forward( inputs = inputs ).local_hidden
        return hidden


    def local_forward(self, inputs: torch.LongTensor, attention_mask: torch.LongTensor = None, targets: torch.Tensor = None):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `optional`): 
                    Mask to avoid performing attention on padding token indices.
                    Mask values selected in ``[0, 1]``:
                        - 1 for tokens that are **not masked**,
                        - 0 for tokens that are **maked**.        

                targets (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
                    Targets for computing the next sequence prediction (classification) loss. 
                    Indices should be in ``[0, 1]``:
                        - 0 indicates sequence B is a continuation of sequence A,
                        - eqw1 indicates sequence B is a random sequence.

        BertNucleusBase.local_forward + SimpleNamespace ( 
                    
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        BERT NSP Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        BERT NSP loss using local_context.
                )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        # Call forward method from bert base.
        output = BertNucleusBase.base_local_forward( self, inputs = inputs, attention_mask = attention_mask ) 
        if targets is not None:
            # local_target: projection the local_hidden to target dimension.
            # local_target.shape = [batch_size, 2]
            local_target = self.target_layer( output.local_pooled )
            output.local_target = F.softmax( local_target, dim=1 )
            output.local_target_loss = self.loss_fct( output.local_target.view(-1, 2), targets )            
        return output

    def remote_forward(self, inputs: torch.LongTensor, attention_mask: torch.LongTensor = None, targets: torch.Tensor = None):
        r""" Forward pass inputs and labels through the NSP BERT module. (with queries to the network)

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `optional`): 
                    Mask to avoid performing attention on padding token indices.
                    Mask values selected in ``[0, 1]``:
                        - 1 for tokens that are **not masked**,
                        - 0 for tokens that are **maked**.        

                targets (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
                    Targets for computing the next sequence prediction (classification) loss. 
                    Indices should be in ``[0, 1]``:
                        - 0 indicates sequence B is a continuation of sequence A,
                        - eqw1 indicates sequence B is a random sequence.

        BertNucleusBase.remote_forward + SimpleNamespace ( 
                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        BERT NSP Target predictions produced using remote_context. 

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        BERT NSP loss using remote_target_loss.
                )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        output = BertNucleusBase.base_remote_forward( self, attention_mask = attention_mask, inputs = inputs)
        if targets is not None:
            # local_target: projection the local_hidden to target dimension.
            # local_target.shape = [batch_size, 2]
            local_target = self.target_layer( output.local_pooled )
            output.local_target = F.softmax( local_target, dim=1 )
            output.local_target_loss = self.loss_fct( output.local_target.view(-1, 2), targets ) 

            # remote_target: projection the local_hidden to target dimension.
            # remote_target.shape = [batch_size, 2]
            remote_target = self.target_layer( output.remote_pooled )
            output.remote_target = F.softmax( remote_target, dim=1 )
            output.remote_target_loss = self.loss_fct( remote_target.view(-1, 2), targets )
        return output

        
class BertMLMNucleus (BertNucleusBase):
    def __init__(self, config: 'bittensor.Config', **kwargs):
        r""" Bert nucleus for MLM training

            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    BertNSP configuration class.

        """
        super(BertMLMNucleus, self).__init__(config = config, **kwargs)

        # Hugging face config item.
        huggingface_config = BertConfig(    vocab_size=bittensor.__vocab_size__, 
                                            hidden_size=bittensor.__network_dim__, 
                                            num_hidden_layers=config.nucleus.num_hidden_layers, 
                                            num_attention_heads=config.nucleus.num_attention_heads, 
                                            intermediate_size=bittensor.__network_dim__, 
                                            is_decoder=False)
      
        # target_layer: maps from hidden layer to vocab dimension for each token. Used by MLM loss.
        # [batch_size, sequence_len, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__vocab_size__]
        self.target_layer = transformers.models.bert.modeling_bert.BertLMPredictionHead( huggingface_config )

        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the BERT NSP Nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.local_forward( inputs = inputs, targets = None ).local_hidden
        return hidden

    def local_forward(self, inputs: torch.LongTensor, targets: torch.LongTensor = None):
        r""" Forward pass inputs and labels through the MLM BERT module.

            Args:
                inputs (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `required`):
                    Batch_size length list of tokenized sentences.
                
                targets (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
                    Targets for computing the masked language modeling loss.
                    Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                    Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with targets
                    in ``[0, ..., config.vocab_size]``   

            BertNucleusBase.local_forward + SimpleNamespace ( 
                    
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        BERT NSP Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        BERT NSP loss using local_context.
            )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        # Call forward method from bert base.
        output = BertNucleusBase.base_local_forward( self, inputs = inputs ) 

        if targets is not None:
            # local_target: projection the local_hidden to target dimension.
            # local_target.shape = [batch_size, bittensor.__vocab_size__]
            local_target = self.target_layer( output.local_hidden )
            output.local_target = F.softmax( local_target, dim=1 )
            output.local_target_loss = self.loss_fct( output.local_target.view( -1, bittensor.__vocab_size__ ), targets.view(-1) )
        return output


    def remote_forward(self, inputs: torch.LongTensor, targets: torch.LongTensor = None):
        r""" Forward pass inputs and labels through the MLM BERT module. (with queries to the network)

            Args:
                inputs (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `required`):
                    Batch_size length list of tokenized sentences.
                
                targets (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
                    Targets for computing the masked language modeling loss.
                    Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                    Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with targets
                    in ``[0, ..., config.vocab_size]``   

            BertNucleusBase.remote_forward + SimpleNamespace ( 
                    
                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        BERT NSP Target predictions produced using local_context. 

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        BERT NSP loss using local_context.
            )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.
        # Call forward method from bert base.
        output = BertNucleusBase.base_remote_forward( self, inputs = inputs ) 

        if targets is not None:
            # local_target: projection the local_hidden to target dimension.
            # local_target.shape = [batch_size, bittensor.__vocab_size__]
            output.local_target =  F.softmax(self.target_layer( output.local_hidden ), dim=1)
            output.local_target_loss = self.loss_fct(output.local_target.view(-1, bittensor.__vocab_size__), targets.view(-1))
            
            # remote_target_loss: logit(1) > logit(0) if next_inputs are the real next sequences.
            # remote_target_loss: [1]
            output.remote_target = F.softmax(self.target_layer( output.remote_hidden ), dim=1)
            output.remote_target_loss = self.loss_fct(output.remote_target.view(-1, bittensor.__vocab_size__), targets.view(-1))
        return output