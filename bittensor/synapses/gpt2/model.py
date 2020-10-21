import bittensor

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import GPT2Config, GPT2Model
from typing import List, Tuple, Dict, Optional


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


class GPT2LMSynapse(bittensor.Synapse):
    """ A Bittensor Synapse training GPT2 with Masked Language Modelling (MLM)
    """

    def __init__(self, config: GPT2Config):
        super(GPT2LMSynapse, self).__init__()
        self.config = config
        self.router = bittensor.Router(x_dim=bittensor.__network_dim__,
                                       key_dim=100,
                                       topk=10)
        self.transformer = GPT2Model(self.config)
        self.student_transformer = GPT2Model(self.config)
        self.pooler = GPT2Pooler(self.config)
        self.joiner = torch.nn.Linear(2 * bittensor.__network_dim__,
                                      bittensor.__network_dim__)
        self.head = nn.Linear(self.config.n_embd,
                              self.config.vocab_size,
                              bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the NSP BERT Synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by using the local student distillation model as context.
        """
        return self.forward(inputs=inputs.to(self.device), query=False)['local_output']

    def forward(self, inputs: torch.LongTensor, query: bool = False):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                query (:obj:`bool')`, `optional`):
                    Switch to True if this forward pass makes a remote call to the network. 

            dictionary with { 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encoding of inputs produced by using the local student distillation model as 
                        context rather than the network. 

                    network_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encoding of inputs produced by using the network inputs as context to the local model rather than 
                        the student.

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Next sentence prediction loss computed using the local_output and with respect to the passed labels.

                    network_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):  
                        Next sentence prediction loss computed using the network_output and with respect to the passed labels.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss produced by the student with respect to the network context.
                }
        """

        # Return vars.
        loss = torch.tensor(0.0)
        local_output = None
        network_output = None
        network_target_loss = None
        local_target_loss = None
        distillation_loss = None

        # Run GPT
        local_encoding = self.transformer(input_ids=inputs, return_dict=True)
        local_pooled = self.pooler(local_encoding.last_hidden_state)

        # If query == True make a remote network call.
        if query:
            # network = torch.Tensor(batch_size, bittensor.__network_dim__)
            synapses = bittensor.metagraph.synapses(
            )  # Returns a list of synapses on the network.
            requests, _ = self.router.route(synapses, local_pooled,
                                            inputs)  # routes inputs to network.
            responses = bittensor.dendrite.forward_text(
                synapses, requests)  # Makes network calls.
            network_encoding = self.router.join(
                responses)  # Joins responses based on scores..

        # Student transformer model which learns a mapping from the embedding to the network inputs
        # student_pooled = torch.Tensor (batch_size, config.hidden_size)
        student_encoding = self.student_transformer(input_ids=inputs,
                                                    return_dict=True)
        if query:
            # Distillation loss between student_pooled and network inputs.
            distillation_loss = F.mse_loss(student_encoding.last_hidden_state,
                                           network_encoding)
            loss = loss + distillation_loss

        # Join student and local embedding.
        local_output = self.joiner(
            torch.cat([
                local_encoding.last_hidden_state,
                student_encoding.last_hidden_state
            ],
                      dim=2))
        local_logits = self.head(local_output)
        if query:
            network_output = self.joiner(
                torch.cat([local_encoding.last_hidden_state, network_encoding],
                          dim=2))
            network_logits = self.head(network_output)

        # Compute loss for local_logits
        # Shift so that tokens < n predict n
        shift_logits = local_logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        local_target_loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss + local_target_loss

        if query:
            # Compute loss for local_logits
            # Shift so that tokens < n predict n
            network_shift_logits = network_logits[..., :-1, :].contiguous()
            network_shift_labels = inputs[..., 1:].contiguous()
            network_target_loss = self.loss_fct(
                network_shift_logits.view(-1, network_shift_logits.size(-1)),
                network_shift_labels.view(-1))
            loss = loss + network_target_loss

        return {
            'loss': loss,
            'local_output': local_output,
            'local_target_loss': local_target_loss,
            'network_output': network_output,
            'network_target_loss': network_target_loss,
            'distillation_loss': distillation_loss
        }
