import bittensor
from bittensor.dendrites.pkm import PKMDendrite
from bittensor.synapse import Synapse
from bittensor.synapse import SynapseOutput
from bittensor.session import Session

import argparse
from munch import Munch
import random
import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

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
    batch_inputs = tokenizer(batch_text, return_tensors='pt', padding=True)['input_ids']
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

class GPT2LMSynapse(Synapse):
    """ A Bittensor Synapse training GPT2 with Masked Language Modelling (MLM)
    """

    def __init__(self,
                 config: Munch,
                 session: Session):
        r""" Init a new ffnn synapse module.

            Args:
                config (:obj:`munch.Munch`, `required`): 
                    munched config class.

                session (:obj:`bittensor.Session`, `required`): 
                    bittensor session object. 
                    Defaults to bittensor.session global if exists.
        """
        super(GPT2LMSynapse, self).__init__(
            config = config,
            session = session)

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
                bos_token_id=50256, 
                eos_token_id=50256
        )

        # encoder_layer: encodes tokenized sequences to network dim.
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.encoder_transformer = GPT2Model(huggingface_config)

        # pooler_layer: pools transformed sequence to network_dim for router.
        # [batch_size, bittensor.__network_dim__, sequence_len] -> [batch_size, bittensor.__network_dim__]
        self.pooler = GPT2Pooler(huggingface_config)

        # dendrite: (PKM layer) queries network using pooled embeddings as context.
        # [batch_size, bittensor.__network_dim__] -> topk * [batch_size, bittensor.__network_dim__]
        self.dendrite = PKMDendrite(config, session, query_dim = bittensor.__network_dim__)

        # context_transformer: distills the remote_context from inputs
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.context_transformer = GPT2Model(huggingface_config)

        # hidden_layer: transforms context and encoding to network_dim hidden units.
        # [batch_size, sequence_dim, 2 * bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.hidden_layer = torch.nn.Linear(2 * bittensor.__network_dim__, bittensor.__network_dim__)

        # target_layer: maps from hidden layer to vocab dimension for each token. Used by MLM loss.
        # [batch_size, sequence_len, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__vocab_size__]
        self.target_layer = nn.Linear(bittensor.__network_dim__, bittensor.__vocab_size__, bias=False)
        
        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = torch.nn.CrossEntropyLoss()
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):    
        r""" Add custom params to the parser.
        """
        parser.add_argument('--synapse.n_head', default=2, type=int, 
                            help='Number of attention heads for each attention layer in the Transformer encoder.')
        parser.add_argument('--synapse.n_layer', default=4, type=int, 
                            help='Number of hidden layers in the Transformer encoder.')
        parser.add_argument('--synapse.n_inner', default=None, type=int, 
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
        PKMDendrite.add_args(parser)

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the MLM GPT Synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.forward(inputs=inputs, training = False, remote = False).local_hidden
        return hidden

    def forward(self, 
                inputs: torch.LongTensor, 
                training: bool = True, 
                remote: bool = False):
        r""" Forward pass through GPT MLM synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes an MLM loss.

                remote (:obj:`bool')`, `optional`):
                    Switch to True if this forward pass queries the network for the remote_context.

            bittensor.SynapseOutput (
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation used by loss.backward()

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        GPT MLM loss using local_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        GPT MLM Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        GPT MLM loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    weights (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each active neuron.

                    requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(metagraph.state.n)`, `optional`): 
                        number of requests sent to each uid in this batch.

                    return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[metagraph.state.n]`, `required`):
                        dendrite call return codes. 0 for success.

                    metadata (:obj:`dict {'accuracy', torch.FloatTensor} ` of shape :obj:`(1)`, `optional`):
                        additional metadata output, specifically accuracy.
                )
        """

        # Return vars to be filled.
        output = SynapseOutput(loss = torch.tensor(0.0))
        
        # encoding: transformer encoded sentences.
        # encoding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        encoding = self.encoder_transformer(input_ids=inputs, return_dict=True).last_hidden_state
        
        # pooled: pooled encodings by taking the hidden units of the last token.
        # pooled.shape = [batch_size, bittensor.__network_dim__]
        pooled = self.pooler(encoding)
        
        # local_context: distilled version of remote_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        local_context = self.context_transformer(input_ids=inputs, return_dict=True).last_hidden_state

        # local_hidden: hidden layer encoding of sequence with local_context.
        # local_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        local_hidden = torch.cat([encoding, local_context], dim=2)
        local_hidden = self.hidden_layer(local_hidden)
        output.local_hidden = local_hidden

        if training:
            # local_target: projection of local_hidden onto target dimension.
            # local_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            local_target = self.target_layer(local_hidden)
            output.local_target = local_target

            # local_target_loss: MLM loss between local_target and passed targets.
            # local_target_loss.shape = [1]
            shift_logits = local_target[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()
            local_target_loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            output.loss = output.loss + local_target_loss
            output.local_target_loss = local_target_loss
                   
        if remote:
            output = self.forward_remote(local_context, local_hidden, inputs, pooled, encoding, training, output)

        return output

    def forward_remote(self, local_context, local_hidden, inputs, pooled, encoding, training, output):
        """ Forward pass inputs and labels through the GPT2 module.


        Args:
            local_context (:obj: `torch.FloatTensor` of shape :obj: `(batch_size, sequence_len, bittensor.__network_dim__)`, `required`)
                    Distillation model for remote_context.
            
            local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

            
            inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            encoding (:obj:`torch.LongTensor` of shape :obj:`` 
                    Transformer encoded sentences
            
            training (:obj:`bool')`, `optional`, defaults to True):
                    Switch to True if this forward pass computes an MLM loss.

            output (SynapseOutput): Object being populated by local forward.

        Returns:
            bittensor.SynapseOutput ( 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation used by loss.backward()

                    local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Hidden layer encoding produced using local_context.

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
                        BERT NSP Target predictions produced using local_context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        BERT NSP loss using local_context.

                    remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
                        Hidden layer encoding produced using the remote_context.

                    remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
                        BERT NSP Target predictions using the remote_context.

                    remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        BERT NSP loss using the remote_context.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss between local_context and remote_context.

                    weights (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                        weights for each active neuron.

                    requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(metagraph.state.n)`, `optional`): 
                        number of requests sent to each uid in this batch.

                    return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                        dendrite call return codes. 0 for success.

                    metadata (:obj:`dict {'accuracy', torch.FloatTensor} ` of shape :obj:`(1)`, `optional`):
                        additional metadata output, specifically accuracy.
                )
        """
        # remote_context: joined responses from a bittensor.forward_text call.
        # remote_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        remote_context, weights, sizes, return_codes = self.dendrite.forward_text(inputs, pooled)
        output.weights = weights
        output.return_codes = return_codes
        output.request_sizes = sizes 

        # distillation_loss: distillation loss between local_context and remote_context
        # distillation_loss.shape = [1]
        distillation_loss = F.mse_loss(local_context, remote_context.detach())
        output.loss = output.loss + distillation_loss
        output.distillation_loss = distillation_loss

        # remote_hidden: hidden layer encoding using remote_context.
        # remote_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        remote_hidden = torch.cat([encoding, remote_context], dim=2)
        remote_hidden = self.hidden_layer(remote_hidden)

        if training:
            # remote_target: projection of remote_hidden onto target dimension.
            # remote_target.shape = [batch_size, sequence_len, bittensor.__vocab_size__]
            remote_target = self.target_layer(local_hidden)
            output.remote_target = remote_target

            # remote_target_loss: MLM loss between remote_target and passed targets.
            # remote_target_loss.shape = [1]
            shift_logits = remote_target[..., :-1, :].contiguous()
            
            shift_labels = inputs[..., 1:].contiguous()
            remote_target_loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            output.loss = output.loss + remote_target_loss
            output.remote_target_loss = remote_target_loss
        

        return output


