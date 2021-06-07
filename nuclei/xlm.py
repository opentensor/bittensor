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
import bittensor
import random
import torch
import argparse
import torch.nn.functional as F

from transformers import XLMConfig, XLMModel
from torch import nn

from collections.abc import Callable
from types import SimpleNamespac

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
        batch_text.append(data[random.randint(0, len(data))]['review_body'])
    batch_inputs = tokenizer(batch_text, return_tensors='pt', padding=True, truncation=True, max_length=bittensor.__network_dim__)['input_ids']
    return batch_inputs


class XLMPooler(nn.Module):
    
    def __init__(self, xlm_config):
        super().__init__()
        self.dense = nn.Linear(xlm_config.emb_dim, xlm_config.emb_dim)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        """We "pool" the model by simply taking the hidden state corresponding to the first token

        Args:
            hidden_states (:obj:`nn.Linear`): hidden layer encoding of sequence with local_context.

        Returns:
            :obj:`nn.Linear`: Linear layer with [bittensor.__network_dim__, bittensor.__network_dim__] dimensionality.
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class XLMNucleus (torch.nn.Module):
    """A Bittensor Nucleus training XLM 

    Args:
        nucleus (:obj:`Nucleus`): The Nucleus superclass, which contains fwd and backward logic.

    """

    def __init__(self, routing_callback, config: 'bittensor.Config' = None, **kwargs):
        """ Initialize a new XLM nucleus module.

        Args:
            config (:obj:`bittensor.Config`, `required`): 
                    munched config class.
        """
        super(XLMNucleus, self).__init__()
        if config == None:
            config = XLMNucleus.config()
        XLMNucleus.check_config(config)
        self.config = config

        # To be set.
        self.routing_callback = routing_callback
        
        # Build config.
        xlm_config = XLMConfig (
            vocab_size=bittensor.__vocab_size__, 
            emb_dim=bittensor.__network_dim__,
            n_layers=config.nucleus.n_layers,
            n_heads=config.nucleus.n_heads, 
            # More needed
        )

        # model layer: encodes tokenized sequences to network dim.
        self.xlm = XLMModel(xlm_config)

        # pooler layer: pools the hidden units for use by the pkm dendrite rpc query.
        self.pooler = XLMPooler(xlm_config)

        # hidden layer: transforms context and encoding to network dimension hidden units.
        self.hidden_layer = nn.Linear( bittensor.__network_dim__, bittensor.__network_dim__ )

        # target layer: maps from hidden layer to vocab dimension for each token.
        self.target_layer = nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__, bias=False )

        # Loss function
        self.loss_fct = nn.CrossEntropyLoss()

        self.to(self.device)
    
    @staticmethod
    def config() -> 'bittensor.Config':
        parser = argparse.ArgumentParser()
        XLMNucleus.add_args(parser)
        config = bittensor.config( parser )
        return config
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        """ Add custom params to the Nucleus

        Args:
            parser (:obj:`argparse.AgumentParser`): Argument Parser object.

        """
        parser.add_argument('--nucleus.emb_dim', default=bittensor.__network_dim__, type=int,
                                help='Dimensionality of the encoder layers and the pooler layer.')
        parser.add_argument('--nucleus.n_layers', default=12, type=int,
                                help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--nucleus.n_heads', default=16, type=int,
                                help='Number of attention heads for each attention layer in the Transformer encoder.')
        parser.add_argument('--nucleus.dropout', default=0.1, type=float,
                                help='The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.')
        parser.add_argument('--nucleus.attention_dropout', default=0.1, type=float,
                                help='The dropout probability for the attention mechanism.')
        parser.add_argument('--nucleus.gelu_activation', default=True, type=bool, 
                                help='Whether or not to use gelu for the activations instead of relu.')
        parser.add_argument('--nucleus.sinusoidal_embeddings', default=False, type=bool, 
                                help='Whether or not to use sinusoidal positional embeddings instead of absolute positional embeddings.')
        parser.add_argument('--nucleus.causal', default=False, type=bool,
                                help='Whether or not the model should behave in a causal manner. Causal models use a triangular attention mask in order to only attend to the left-side context instead if a bidirectional context.')
        parser.add_argument('--nucleus.asm', default=False, type=bool,
                                help='Whether or not to use an adaptive log softmax projection layer instead of a linear layer for the prediction layer.')
        parser.add_argument('--nucleus.n_langs', default=1, type=int,
                                help='The number of languages the model handles. Set to 1 for monolingual models.')
        parser.add_argument('--nucleus.use_lang_emb', default=True, type=bool, 
                                help='Whether to use language embeddings. Some models use additional language embeddings, see the multilingual models page for information on how to use them.')
        parser.add_argument('--nucleus.max_position_embeddings', default=512, type=bool,
                                help='The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 512 or 1024 or 2048).')
        parser.add_argument('--nucleus.embed_init_std', default=pow(2048,-0.5), type=float,
                                help='The standard deviation of the truncated_normal_initializer for initializing the embedding matrices.')
        parser.add_argument('--nucleus.init_std', default=50257, type=int,
                                help='The standard deviation of the truncated_normal_initializer for initializing all weight matrices except the embedding matrices.')
        parser.add_argument('--nucleus.layer_norm_eps', default=pow(1,-12), type=float,
                                help='The epsilon used by the layer normalization layers.')
        parser.add_argument('--nucleus.bos_index', default=0, type=int,
                                help='The index of the beginning of sentence token in the vocabulary.')
        parser.add_argument('--nucleus.eos_index', default=1, type=int,
                                help='The index of the end of sentence token in the vocabulary.')
        parser.add_argument('--nucleus.pad_index', default=2, type=int,
                                help='The index of the padding token in the vocabulary.')
        parser.add_argument('--nucleus.unk_index', default=3, type=int,
                                help='The index of the unknown token in the vocabulary.')
        parser.add_argument('--nucleus.mask_index', default=5, type=int,
                                help='The index of the masking token in the vocabulary.')
        parser.add_argument('--nucleus.is_encoder', default=True, type=bool,
                                help='Whether or not the initialized model should be a transformer encoder or decoder as seen in Vaswani et al.')
        parser.add_argument('--nucleus.summary_type', default="first", type=str,
                                help='Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.')
        parser.add_argument('--nucleus.summary_use_proj', default=True, type=bool,
                                help='Argument used when doing sequence summary. Used in the sequence classification and multiple choice models. Whether or not to add a projection after the vector extraction.')
        parser.add_argument('--nucleus.summary_activation', type=str, 
                                help='Pass "tanh" for a tanh activation to the output, any other value will result in no activation.')
        parser.add_argument('--nucleus.summary_proj_to_labels', default=True, type=bool,
                                help='Whether the projection outputs should have config.num_labels or config.hidden_size classes.')
        parser.add_argument('--nucleus.summary_first_dropout', default=0.1, type=float,
                                help='The dropout ratio to be used after the projection and activation.')
        parser.add_argument('--nucleus.start_n_top', default=5, type=int,
                                help=' Used in the SQuAD evaluation script.')
        parser.add_argument('--nucleus.end_n_top', default=5, type=int,
                                help='Used in the SQuAD evaluation script.')
        parser.add_argument('--nucleus.mask_token_id', default=0, type=int,
                                help='Model agnostic parameter to identify masked tokens when generating text in an MLM context.')
        parser.add_argument('--nucleus.lang_id', default=1, type=int,
                                help='The ID of the language used by the model. This parameter is used when generating text in a given language.')
        
    @staticmethod
    def check_config(config: 'bittensor.Config'):
        assert config.nucleus.n_layers > 0, "Number of hidden layers in the Transformer encoder must be > 0"
        assert config.nucleus.n_heads > 0, "Number of attention heads for each attention layer in the Transformer encoder must be > 0"

    def attach_routing_callback(self, routing_callback: Callable[ [torch.Tensor, torch.Tensor], torch.Tensor ] ):
        """ Assigns the routing_callback call to this neuron.

            Returns:
                routing_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                    Routing function to call on self.route()
        """
        self.routing_callback = routing_callback

    @property
    def route( self, inputs: torch.Tensor, query: torch.Tensor) -> torch.FloatTensor:
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
    
    def forward_text (self, inputs: torch.LongTensor):
        """ Local forward inputs through the XLM Nucleus.

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
        """ Forward pass through XLM nucleus.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes an CLM loss.

            SimpleNamespace {
                    local_context (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer context.

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        XLM CLM Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        XLM CLM loss using local_context.
                }
        """

        # return variables to be filled.
        output = SimpleNamespace()

        # local_context: distilled version of remote context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_context = self.xlm(input_ids=inputs, return_dict=True).last_hidden_state

        # local_hidden: hidden layer encoding of sequence with local_context.
        # local_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.local_hidden = self.hidden_layer(output.local_context)

        if training:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.local_target = self.target_layer(output.local_hidden)

            # local_target_loss: XLM loss between local_target and ground truth targets (passed targets)
            shift_logits = output.local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.local_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return output

    def remote_forward(self, inputs: torch.LongTensor, training: bool) -> SimpleNamespace:
        """ Forward pass inputs and labels through the XLM module.


        Args:
            inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

            training (:obj:`bool')`, `optional`, defaults to True):
                Switch to True if this forward pass computes a CLM loss.

        Returns:
            self.local_forward() + SimpleNamespace ( 

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        XLM CLM Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        XLM CLM loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    router (:obj:`SimpleNamespace`, `required`): 
                        Outputs from the pkm dendrite.
            )
        """
        # Filter out of range tokens
        inputs = torch.clamp(inputs, 0, bittensor.__vocab_size__)

        # Run local model
        # output = SimpleNamespace
        output = self.local_forward(inputs, training)

        # pooled: pooled hidden layer from local run, used as our query context.
        # pooled.shape = [batch_size, bittensor.__network_dim__]
        pooled = self.pooler(output.local_hidden.detach())

        # remote_context: joined responses from a dendrite.forward_text call.
        # remote_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        output.remote_context = self.route( inputs = inputs.to(self.device), query = pooled )
        output.remote_context = output.remote_context

        # Distillation loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        output.distillation_loss = F.mse_loss(output.local_context, output.remote_context.detach())

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, sequence_length, bittensor.__network_dim__]
        output.remote_hidden = self.hidden_layer( output.remote_context )

        if training:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            output.remote_target = self.target_layer(output.remote_hidden)

            # remote_target_loss: CLM oss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            shift_logits = output.remote_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            output.remote_target_loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return output