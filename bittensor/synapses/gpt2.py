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
import torch.nn.functional as F

from transformers import GPT2Config, GPT2Model
from torch import nn
from munch import Munch
from types import SimpleNamespace

import bittensor
from bittensor.routers.pkm import PKMRouter

def nextbatch(data, batch_size, tokenizer):
    """ Returns a random batch of sentences from text dataset.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            batch_inputs torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
    """
    batch_text = []
    for _ in range(batch_size):
        batch_text.append(data[random.randint(0, len(data))]['text'])
    batch_inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True)['input_ids']
    return batch_inputs

class GPT2Pooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class GPT2LMSynapse(bittensor.synapse.Synapse):
    """ A Bittensor Synapse training GPT2 with Masked Language Modelling (MLM)
    """
    def __init__(self, config: Munch):
        r""" Init a new ffnn synapse module.

            Args:
                config (:obj:`munch.Munch`, `required`): 
                    munched config class.
        """
        super(GPT2LMSynapse, self).__init__(config = config)
        if config == None:
            config = GPT2LMSynapse.build_config()

        # Build hugging face config.
        huggingface_config = GPT2Config(
                vocab_size=bittensor.__vocab_size__, 
                n_embd=bittensor.__network_dim__,
                n_layer=config.synapse.n_layer,
                n_head=config.synapse.n_head, 
                n_inner=config.synapse.n_inner, 
                activation_function=config.synapse.activation_function, 
                resid_pdrop=config.synapse.resid_pdrop, 
                embd_pdrop=config.synapse.embd_pdrop, 
                attn_pdrop=config.synapse.attn_pdrop, 
                layer_norm_epsilon=config.synapse.layer_norm_epsilon, 
                initializer_range=config.synapse.initializer_range, 
                summary_type=config.synapse.summary_type, 
                summary_use_proj=config.synapse.summary_use_proj, 
                summary_activation=config.synapse.summary_activation, 
                summary_proj_to_labels=config.synapse.summary_proj_to_labels, 
                summary_first_dropout=config.synapse.summary_first_dropout, 
        )

        # encoder_layer: encodes tokenized sequences to network dim.
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.transformer = GPT2Model(huggingface_config)

        # pooler_layer: pools the hidden units for use by the pkm dendrite rpc query.
        # [batch_size, bittensor.__network_dim__, sequence_len] -> [batch_size, bittensor.__network_dim__]
        self.pooler = GPT2Pooler(huggingface_config)

        # router: (PKM layer) queries network using pooled embeddings as context.
        # [batch_size, bittensor.__network_dim__] -> topk * [batch_size, bittensor.__network_dim__]
        self.router = PKMRouter(config, query_dim = bittensor.__network_dim__)

        # hidden_layer: transforms context and encoding to network_dim hidden units.
        # [batch_size, sequence_dim, 2 * bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.hidden_layer = torch.nn.Linear( bittensor.__network_dim__, bittensor.__network_dim__ )

        # target_layer: maps from hidden layer to vocab dimension for each token. Used by MLM loss.
        # [batch_size, sequence_len, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__vocab_size__]
        self.target_layer = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__, bias=False )
        
        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.to(self.device)

    @staticmethod   
    def build_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        GPT2LMSynapse.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        GPT2LMSynapse.check_config(config)
        return config
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):    
        r""" Add custom params to the parser.
        """
        parser.add_argument('--synapse.n_head', default=1, type=int, 
                            help='Number of attention heads for each attention layer in the Transformer encoder.')
        parser.add_argument('--synapse.n_layer', default=2, type=int, 
                            help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--synapse.n_inner', default=8, type=int, 
                            help='The dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd')
        parser.add_argument('--synapse.activation_function', default='gelu_new', type=str, 
                            help='Activation function, to be selected in the list :obj:`["relu", "silu", "gelu", "tanh", "gelu_new"]')
        parser.add_argument('--synapse.resid_pdrop', default=0.1, type=float, 
                            help='GPT residual dropout probabilit.')
        parser.add_argument('--synapse.embd_pdrop', default=0.1, type=float, 
                            help='GPT embedding dropout probability.')
        parser.add_argument('--synapse.attn_pdrop', default=0.1, type=float, 
                            help='GPT attention dropout probability.')
        parser.add_argument('--synapse.layer_norm_epsilon', default=1e-05, type=float, 
                            help='GPT the epsilon to use in the layer normalization layers')
        parser.add_argument('--synapse.summary_type', default='cls_index', type=str, 
                            help='Supply a Tensor of classification token position (like GPT/GPT-2).')
        parser.add_argument('--synapse.initializer_range', default=0.02, type=float, 
                            help='The standard deviation of the truncated_normal_initializer for initializing all weight matrices.')
        parser.add_argument('--synapse.summary_use_proj', default=True, type=bool, 
                            help='Whether or not to add a projection after the vector extraction.')
        parser.add_argument('--synapse.summary_activation', type=str, 
                            help='Pass "tanh" for a tanh activation to the output, any other value will result in no activation.')
        parser.add_argument('--synapse.summary_proj_to_labels', default=True, type=bool, 
                            help='Whether the projection outputs should have config.num_labels or config.hidden_size classes.')
        parser.add_argument('--synapse.summary_first_dropout', default=0.1, type=float, 
                            help='The dropout ratio to be used after the projection and activation.')
        parser.add_argument('--synapse.n_block_filter', default=100, type=int, help='Stale neurons are filtered after this many blocks.')
        PKMRouter.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        pass

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the MLM GPT Synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.local_forward(inputs=inputs.to(self.device), training = False).local_hidden
        return hidden

    def local_forward(self, inputs: torch.LongTensor, training: bool = True) -> SimpleNamespace:
        r""" Forward pass through GPT synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes an MLM loss.

            SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        GPT MLM loss using local_context.
                }
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.

        # Return vars to be filled.
        output = SimpleNamespace()
        
        # local_context: distilled version of remote_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.transformer(input_ids=inputs, return_dict=True).last_hidden_state

        # local_hidden: hidden layer encoding of sequence with local_context.
        # local_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_hidden = self.hidden_layer(output.local_context)

        if training:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.local_target = self.target_layer(output.local_hidden)

            # local_target_loss: MLM loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.local_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                   
        return output

    def remote_forward(self, neuron: bittensor.neuron.Neuron, inputs: torch.LongTensor, training: bool) -> SimpleNamespace:
        """ Forward pass inputs and labels through the GPT2 module.


        Args:
            neuron (:obj: `bittensor.neuron.Neuron`, `required`):
                    Bittensor neuron, used for making queries to the remote network.

            inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

            training (:obj:`bool')`, `optional`, defaults to True):
                Switch to True if this forward pass computes an MLM loss.

        Returns:
            self.local_forward() + SimpleNamespace ( 

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        GPT MLM loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    router (:obj:`SimpleNamespace`, `required`): 
                        Outputs from the pkm dendrite.
            )
        """
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__) # Filter out of range tokens.

        # Run the local model.
        # output = SimpleNamespace
        output = self.local_forward(inputs, training)

        # pooled: pooled hidden layer from local run, used as our query context.
        # pooled.shape = [batch_size, bittensor.__network_dim__]
        pooled = self.pooler(output.local_hidden.detach())

        # remote_context: joined responses from a dendrite.forward_text call.
        # remote_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.router = self.router.forward_text(neuron, inputs.to(self.device), pooled)
        remote_context = output.router.response

        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_context, remote_context.detach())

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_hidden = self.hidden_layer(remote_context)

        if training:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.remote_target = self.target_layer(output.remote_hidden)

            # remote_target_loss: MLM loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            shift_logits = output.remote_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.remote_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return output


