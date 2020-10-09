import bittensor

import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer
from typing import List, Tuple, Dict, Optional

class BertNSPSynapse(bittensor.Synapse):
    """ A Bittensor Synapse training a BERT transformer with Next Sentence Prediction (NSP).
    """

    def __init__(self, config: transformers.modeling_bert.BertConfig):
        super(BertNSPSynapse, self).__init__()                
        self.config = config
        self.router = bittensor.Router(x_dim = bittensor.__network_dim__, key_dim = 100, topk = 10)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embeddings = transformers.modeling_bert.BertEmbeddings(self.config)
        self.encoder = transformers.modeling_bert.BertEncoder(self.config)
        self.pooler = transformers.modeling_bert.BertPooler(self.config)
        self.joiner = nn.Linear(config.hidden_size + config.hidden_size, bittensor.__network_dim__)
        self.student_encoder = transformers.modeling_bert.BertEncoder(self.config)
        self.student_pooler = transformers.modeling_bert.BertPooler(self.config)
        self.nsp = transformers.modeling_bert.BertOnlyNSPHead(self.config) 
        self.nsp_loss_fct = torch.nn.CrossEntropyLoss()
        self.device

    def forward_text(self, inputs: List[str]):
        """ Local forward inputs and labels through the NSP BERT Synapse.

            Args:
                inputs (List[str]): batch_size length list of text sentences.
            
            Returns:
                local_output torch.Tensor(n, bittensor.__network_dim__): (Required) Output encoding of inputs 
                    produced by using the local student distillation model as context.
        """
        return self.forward(sentences = inputs, 
                            next_sentences = None, 
                            next_sentence_labels = None, 
                            query = False) ['local_output']
        
    def forward(    self, 
                    sentences: List[str], 
                    next_sentences: List[str] = None, 
                    next_sentence_labels: torch.Tensor = None, 
                    query: bool = False):
    
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                sentences (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`): 
                    Batch_size length list of text sentences.

                next_sentences (:obj:`List[str]` of shape :obj:`(batch_size)`, `optional`): 
                    Batch_size length list of (potential) next sentences.

                next_sentence_labels (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
                    Labels for computing the next sequence prediction (classification) loss. 
                    Indices should be in ``[0, 1]``:
                        - 0 indicates sequence B is a continuation of sequence A,
                        - 1 indicates sequence B is a random sequence.

                query (:obj:`bool')`, `optional`):
                    Switch to True if this forward pass makes a remote call to the network. 

            Returns:
                dictionary with { 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                        Output encoding of inputs produced by using the local student distillation model as 
                        context rather than the network. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Next sentence prediction loss computed using the local_output and with respect to the passed labels.

                    network_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `optional`): 
                        Output encoding of inputs produced by using the network inputs as context to the local model rather than 
                        the student.

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
                    
        # Tokenize inputs: dict
        #  tokenized = dict {
        #       'input_ids': torch.Tensor(batch_size, max_sequence_len),
        #       'token_type_ids': torch.Tensor(batch_size, max_sequence_len),
        #       'attention_mask': torch.Tensor(batch_size, max_sequence_len)
        # }
        if next_sentence_labels is not None:
            # During training we tokenize both sequences and return token_type_ids which tell
            # the model which token belongs to which sequence.
            # i.e tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]),
            tokenized = self.tokenizer(sentences, text_pair = next_sentences, return_tensors='pt', padding=True).to(self.device)
        else:
            # During inference we only tokenize the inputs, padding them to the longest sequence len.
            tokenized = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)

        # Embed tokens into a common dimension.
        # embedding = torch.Tensor(batch_size, max_sequence_len, config.hidden_size)
        embedding = self.embeddings(input_ids=tokenized['input_ids'], token_type_ids=tokenized['token_type_ids'])

        # Bert transformer returning the last hidden states from the transformer model.
        # encoding = List [
        #   hidden_states = torch.Tensor(batch_size, max_sequence_len, config.hidden_size), 
        # ]
        #import pdb; pdb.set_trace()
        encoding = self.encoder(embedding)

        # Pooling, "pool" the model by simply taking the hidden state corresponding
        # to the first token. first_token_tensor = encoding[:, 0]. Applies a dense linear
        # layer to the encoding for the first token. 
        # pooled = torch.Tensor (batch_size, config.hidden_size)
        pooled = self.pooler(encoding[0])

        # If query == True make a remote network call.
        if query:
            # network = torch.Tensor(batch_size, bittensor.__network_dim__)
            synapses = bittensor.metagraph.synapses() # Returns a list of synapses on the network.
            requests, _ = self.router.route( synapses, pooled, sentences ) # routes inputs to network.
            responses = bittensor.dendrite.forward_text( synapses, requests ) # Makes network calls.
            network = self.router.join( responses ) # Joins responses based on scores..

        # Student transformer model which learns a mapping from the embedding to the network inputs
        # student_pooled = torch.Tensor (batch_size, config.hidden_size)
        student_encoding = self.student_encoder (embedding.detach())
        student_pooled = self.student_pooler(student_encoding[0])
        if query:
            # Distillation loss between student_pooled and network inputs.
            distillation_loss = F.mse_loss(student_pooled, network) 
            loss = loss + distillation_loss

        # Output from the forward pass using only the local and student models.
        # local_ouput = torch.Tensor ( batch_size, bittensor.__network_dim__)
        local_output = self.joiner(torch.cat([pooled, student_pooled], dim=1))
        if next_sentence_labels is not None:
            # Compute the NSP loss by projecting the output to torch.Tensor(2)
            # logit(1) > logit(0) if next_inputs are the real next sequences.
            local_prediction = self.nsp(local_output).to(self.device)
            local_prediction = F.softmax(local_prediction, dim=1)
            local_target_loss = self.nsp_loss_fct(local_prediction.view(-1, 2), next_sentence_labels.to(self.device))
            loss = loss + local_target_loss
            
            # Compute NSP loss for network outputs. Only run this if we have passed network inputs.
            if query:
                # Compute the NSP loss by projecting the network_output to torch.Tensor(2)
                # logit(1) > logit(0) if next_inputs are the real next sequences.
                network_output = self.joiner(torch.cat([pooled, network], dim=1))
                network_prediction = self.nsp(network_output).to(self.device)
                network_prediction = F.softmax(network_prediction, dim=1)
                network_target_loss = self.nsp_loss_fct(network_prediction.view(-1, 2), next_sentence_labels.to(self.device))
                loss = loss + network_target_loss
    
        return {
            'loss': loss,
            'local_output': local_output,
            'local_target_loss': local_target_loss,
            'network_output': network_output,
            'network_target_loss': network_target_loss,
            'distillation_loss': distillation_loss
        }