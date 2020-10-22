import bittensor

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, BertConfig


class BertMLMSynapse(bittensor.Synapse):
    """ A Bittensor Synapse training a BERT transformer with Masked Language Modelling.
    """

    def __init__(self, config: BertConfig):
        super(BertMLMSynapse, self).__init__()
        self.config = config
        self.router = bittensor.Router(x_dim=bittensor.__network_dim__,
                                       key_dim=100,
                                       topk=10)
        self.transformer = BertModel(self.config, add_pooling_layer=True)
        self.student = BertModel(self.config, add_pooling_layer=False)
        self.predictions = transformers.modeling_bert.BertLMPredictionHead(
            self.config)
        self.joiner = nn.Linear(2 * bittensor.__network_dim__,
                                bittensor.__network_dim__)
        self.loss_fct = torch.nn.CrossEntropyLoss() 
        self.to(self.device)

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs through the NSP BERT Synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `required`):
                    Batch_size length list of tokenized sentences.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by using the local student distillation model as context.
        """
        return self.forward(inputs=inputs.to(self.device), labels=None,
                            query=False)['local_output']

    def forward(self,
                inputs: torch.LongTensor,
                labels: torch.LongTensor = None,
                query: bool = False):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `required`):
                    Batch_size length list of tokenized sentences.

                labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
                    Labels for computing the masked language modeling loss.
                    Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                    Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                    in ``[0, ..., config.vocab_size]``

                query (:obj:`bool')`, `optional`):
                    Switch to True if this forward pass makes a remote call to the network. 

            Returns:
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

        # Run local and student models.
        local_encoding = self.transformer(inputs, return_dict=True)

        # If query == True make a remote network call.
        if query:
            # network = torch.Tensor(batch_size, bittensor.__network_dim__)
            synapses = bittensor.metagraph.synapses(
            )  # Returns a list of synapses on the network.
            requests, _ = self.router.route(synapses,
                                            local_encoding.pooler_output,
                                            inputs)  # routes inputs to network.
            responses = bittensor.dendrite.forward_text(
                synapses, requests)  # Makes network calls.
            network_encoding = self.router.join(
                responses)  # Joins responses based on scores..

        # Distillation model.
        student_encoding = self.student(inputs,
                                        return_dict=True).last_hidden_state
        if query:
            # Distillation loss between student_pooled and network inputs.
            distillation_loss = F.mse_loss(student_encoding, network_encoding)
            loss = loss + distillation_loss

        # Join encodings.
        local_output = self.joiner(
            torch.cat([local_encoding.last_hidden_state, student_encoding],
                      dim=2))
        if query:
            network_output = self.joiner(
                torch.cat([local_encoding.last_hidden_state, network_encoding],
                          dim=2))

        # MLM predictions
        local_prediction = self.predictions(local_output)
        if query:
            network_prediction = self.predictions(network_output)

        # Target loss.
        if labels is not None:
            local_target_loss = self.loss_fct(
                local_prediction.view(-1, bittensor.__vocab_size__),
                labels.view(-1))
            loss = loss + local_target_loss
            if query:
                network_target_loss = self.loss_fct(
                    network_prediction.view(-1, bittensor.__vocab_size__),
                    labels.view(-1))
                loss = loss + network_target_loss

        return {
            'loss': loss,
            'local_output': local_output,
            'local_target_loss': local_target_loss,
            'network_output': network_output,
            'network_target_loss': network_target_loss,
            'distillation_loss': distillation_loss
        }


class BertNSPSynapse(bittensor.Synapse):
    """ A Bittensor Synapse training a BERT transformer with Next Sentence Prediction (NSP).
    """

    def __init__(self, config: BertConfig):
        super(BertNSPSynapse, self).__init__()
        self.config = config
        self.router = bittensor.Router(x_dim=bittensor.__network_dim__,
                                       key_dim=100,
                                       topk=10)
        self.transformer = BertModel(self.config, add_pooling_layer=True)
        self.student = BertModel(self.config, add_pooling_layer=True)
        self.joiner = nn.Linear(2 * bittensor.__network_dim__,
                                bittensor.__network_dim__)
        self.pooler = transformers.modeling_bert.BertPooler(self.config)
        self.nsp_head = transformers.modeling_bert.BertOnlyNSPHead(self.config)
        self.nsp_loss_fct = torch.nn.CrossEntropyLoss()

    def forward_text(self, inputs: torch.LongTensor):
        """ Local forward inputs and labels through the NSP BERT Synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by using the local student distillation model as context.
        """
        return self.forward(inputs=inputs, query=False)['local_output']

    def forward(self,
                inputs: torch.LongTensor,
                attention_mask: torch.LongTensor = None,
                labels: torch.Tensor = None,
                query: bool = False):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.

                token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `optional`): 
                    Token Type IDs for training to distinguish between the sentence context and the next sentence.

                attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `optional`): 
                    Mask to avoid performing attention on padding token indices.
                    Mask values selected in ``[0, 1]``:
                        - 1 for tokens that are **not masked**,
                        - 0 for tokens that are **maked**.        

                labels (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
                    Labels for computing the next sequence prediction (classification) loss. 
                    Indices should be in ``[0, 1]``:
                        - 0 indicates sequence B is a continuation of sequence A,
                        - 1 indicates sequence B is a random sequence.

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
  
        # Run local and student models.
        local_encoding = self.transformer(inputs,
                                          attention_mask=attention_mask,
                                          return_dict=True)

        # If query == True make a remote network call.
        if query:
            # network = torch.Tensor(batch_size, bittensor.__network_dim__)
            synapses = bittensor.metagraph.synapses(
            )  # Returns a list of synapses on the network.
            requests, _ = self.router.route(synapses,
                                            local_encoding.pooler_output,
                                            inputs)  # routes inputs to network.
            responses = bittensor.dendrite.forward_text(
                synapses, requests)  # Makes network calls.
            network_encoding = self.router.join(
                responses)  # Joins responses based on scores..

        # Distillation model.
        student_encoding = self.student(inputs,
                                        attention_mask=attention_mask,
                                        return_dict=True).last_hidden_state
        if query:
            # Distillation loss between student_pooled and network inputs.
            distillation_loss = F.mse_loss(student_encoding, network_encoding)
            loss = loss + distillation_loss

        # Join encodings.
        local_output = self.joiner(
            torch.cat([local_encoding.last_hidden_state, student_encoding],
                      dim=2))
        if query:
            network_output = self.joiner(
                torch.cat([local_encoding.last_hidden_state, network_encoding],
                          dim=2))

        # NSP predictions
        if labels is not None:
            # Compute the NSP loss by projecting the output to torch.Tensor(2)
            # logit(1) > logit(0) if next_inputs are the real next sequences.
            local_pooled = self.pooler(local_output)
            local_prediction = self.nsp_head(local_pooled)
            local_prediction = F.softmax(local_prediction, dim=1)
            # Compute NSP loss for network outputs. Only run this if we have passed network inputs.
            if query:
                # Compute the NSP loss by projecting the network_output to torch.Tensor(2)
                # logit(1) > logit(0) if next_inputs are the real next sequences.
                network_pooled = self.pooler(network_output)
                network_prediction = self.nsp_head(network_pooled)
                network_prediction = F.softmax(network_prediction, dim=1)

        # Target loss.
        if labels is not None:
            local_target_loss = self.nsp_loss_fct(local_prediction.view(-1, 2),
                                                  labels)
            loss = loss + local_target_loss
            if query:
                network_target_loss = self.nsp_loss_fct(
                    network_prediction.view(-1, 2), labels)
                loss = loss + network_target_loss

        return {
            'loss': loss,
            'local_output': local_output,
            'local_target_loss': local_target_loss,
            'network_output': network_output,
            'network_target_loss': network_target_loss,
            'distillation_loss': distillation_loss
        }
