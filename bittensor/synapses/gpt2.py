import bittensor
from bittensor.router import Router
from bittensor.synapse import Synapse
from bittensor.synapse import SynapseConfig
from bittensor.synapse import SynapseOutput
from bittensor.session import BTSession

import random
import torch
from torch import nn
import torch.nn.functional as F
import transformers
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

class GPT2MLMConfig (SynapseConfig):
    r"""
    This is the configuration class for a :class:`~GPT2LMSynapse`.
    

    Args:
        huggingface_config (:obj:`transformers.GPT2Config`, `required`, defaults to GPT2MLMConfig.__default_huggingface_config__):
            huggingface config for underlying transformer model.      

    examples:

        >>> from bittensor.synapses.gpt2 import GPT2LMConfig, GPT2LMSynapse

        >>> # Initializing a GPT2MLMConfig configuration.
        >>> configuration = GPT2MLMConfig()

        >>> # Initializing the model from configuration.
        >>> configuration = GPT2LMSynapse ( configuration )
    """

    __default_huggingface_config__ = GPT2Config(    vocab_size=bittensor.__vocab_size__, 
                                                    n_embd=bittensor.__network_dim__,
                                                    n_layer=3,
                                                    n_head=2, 
                                                    n_inner=None, 
                                                    activation_function='gelu_new', 
                                                    resid_pdrop=0.1, 
                                                    embd_pdrop=0.1, 
                                                    attn_pdrop=0.1, 
                                                    layer_norm_epsilon=1e-05, 
                                                    initializer_range=0.02, 
                                                    summary_type='cls_index', 
                                                    summary_use_proj=True, 
                                                    summary_activation=None, 
                                                    summary_proj_to_labels=True, 
                                                    summary_first_dropout=0.1, 
                                                    bos_token_id=50256, 
                                                    eos_token_id=50256
                                                )
    
    def __init__(self, **kwargs):
        super(GPT2MLMConfig, self).__init__(**kwargs)
        self.huggingface_config = kwargs.pop("huggingface_config", self.__default_huggingface_config__)
        self.run_checks()
    
    def run_checks(self):
        assert isinstance(self.huggingface_config, transformers.GPT2Config)
        assert self.huggingface_config.n_embd == bittensor.__network_dim__, "GPT embedding dim {} != {}".format(self.huggingface_config.n_embd, bittensor.__network_dim__)
        assert self.huggingface_config.vocab_size == bittensor.__vocab_size__, "GPT vocab size must match bittensor.__vocab_size {} != {}".format(self.huggingface_config.vocab_size, bittensor.__vocab_size__)

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
                 config: GPT2MLMConfig,
                 session: BTSession):
        r""" Init a new ffnn synapse module.

            Args:
                config (:obj:`bittensor.gpt2.GPT2MLMConfig`, `required`): 
                    GPTMLM configuration class.

                session (:obj:`bittensor.BTSession`, `required`): 
                    bittensor session object. 
                    Defaults to bittensor.session global if exists.
        """
        super(GPT2LMSynapse, self).__init__(
            config = config,
            session = session)
        # encoder_layer: encodes tokenized sequences to network dim.
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.encoder_transformer = GPT2Model(self.config.huggingface_config)

        # pooler_layer: pools transformed sequence to network_dim for router.
        # [batch_size, bittensor.__network_dim__, sequence_len] -> [batch_size, bittensor.__network_dim__]
        self.pooler = GPT2Pooler(self.config.huggingface_config)

        # router: (PKM layer) queries network using pooled embeddings as context.
        # [batch_size, bittensor.__network_dim__] -> topk * [batch_size, bittensor.__network_dim__]
        self.router = Router(x_dim=bittensor.__network_dim__, key_dim=100, topk=10)

        # context_transformer: distills the remote_context from inputs
        # [batch_size, sequence_len] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.context_transformer = GPT2Model(self.config.huggingface_config)

        # hidden_layer: transforms context and encoding to network_dim hidden units.
        # [batch_size, sequence_dim, 2 * bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__network_dim__]
        self.hidden_layer = torch.nn.Linear(2 * bittensor.__network_dim__, bittensor.__network_dim__)

        # target_layer: maps from hidden layer to vocab dimension for each token. Used by MLM loss.
        # [batch_size, sequence_len, bittensor.__network_dim__] -> [batch_size, sequence_len, bittensor.__vocab_size__]
        self.target_layer = nn.Linear(bittensor.__network_dim__, bittensor.__vocab_size__, bias=False)
        
        # Loss function: MLM cross-entropy loss.
        # predicted: [batch_size, sequence_len, 1], targets: [batch_size, sequence_len, 1] -> [1]
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the MLM GPT Synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Hidden layer representation produced using the local_context.
        """
        hidden = self.forward(inputs=inputs.to(self.device), training = False, remote = False).local_hidden
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

        # remote_context: joined responses from a bittensor.forward_text call.
        # remote_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        if remote:
            # network = torch.Tensor(batch_size, bittensor.__network_dim__)
            synapses = self.session.metagraph.synapses()  # Returns a list of synapses on the network.
            requests, _ = self.router.route(synapses, pooled, inputs)  # routes inputs to network.
            responses = self.sessio.dendrite.forward_text(synapses, requests)  # Makes network calls.
            remote_context = self.router.join(responses)  # Join responses with scores.

        # local_context: distilled version of remote_context.
        # local_context.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        local_context = self.context_transformer(input_ids=inputs, return_dict=True).last_hidden_state
        if remote:
            # distillation_loss: distillation loss between local_context and remote_context
            # distillation_loss.shape = [1]
            distillation_loss = F.mse_loss(local_context, remote_context.detach())
            output.distillation_loss = distillation_loss
            output.loss = output.loss + distillation_loss

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
            output.local_target_loss = local_target_loss
            output.loss = output.loss + local_target_loss


        if remote:
            # remote_hidden: hidden layer encoding using remote_context.
            # remote_hidden.shape = [batch_size, sequence_len, bittensor.__network_dim__]
            remote_hidden = torch.cat([encoding, remote_context], dim=2)
            remote_hidden = self.hidden_layer(remote_hidden)
            output.remote_hidden = remote_hidden

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
