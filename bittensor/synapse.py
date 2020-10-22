from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

import bittensor
import bittensor.dendrite
import bittensor.metagraph
from bittensor import bittensor_pb2

class Synapse(nn.Module):
    """ Bittensor synapse class.
    """

    def __init__(   self,
                    config: bittensor.SynapseConfig,
                    dendrite: bittensor.dendrite.Dendrite = None,
                    metagraph: bittensor.metagraph.Metagraph = None):
        r""" Init synapse module.

            Args:
                config (:obj:`bittensor.SynapseConfig`, `required`): 
                    Base synapse config configuration class.

                dendrite (:obj:`bittensor.dendrite.Dendrite`, `optional`, bittensor.dendrite): 
                    bittensor dendrite object used for queries to remote synapses.
                    Defaults to bittensor.dendrite global.

                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`, bittensor.metagraph): 
                    bittensor metagraph containing network graph information. 
                    Defaults to bittensor.metagraph global.

        """
        super().__init__()

        self.config = config
        
        # Bittensor dendrite object used for queries to remote synapses.
        # Defaults to bittensor.dendrite global object.
        self.dendrite = dendrite
        if self.dendrite == None:
            self.dendrite = bittensor.dendrite
            if bittensor.dendrite == None:
                raise Warning ('Synapse initialized without a valid dendrite. Call bittensor.init() to create a global dendrite instance.')

        # Bttensor metagraph containing network graph information.
        # Defaults to bittensor.metagraph global object.
        self.metagraph = metagraph
        if self.metagraph == None:
            self.metagraph = bittensor.metagraph
            if bittensor.metagraph == None:
                raise Warning ('Synapse initialized without a valid metagraph. Call bittensor.init() to create a global metagraph instance.')


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Send model to appropriate device (CPU or CUDA)
        self.to(self.device)

    def synapse_key(self) -> str:
        return self.config.synapse_key

    def set_synapse_key(self, key):
        self.config.synapse_key = key

    def deepcopy(self):
        """ Returns a copy of this synapse by passing the model params to load_state_dict.

            Returns:
                synapse_copy (:obj:`self.__class__`, `required`): 
                    Deep copy synapse object.
        """
        SynapseClass = self.__class__
        synapse_copy = SynapseClass(self.config)
        synapse_copy.load_state_dict(self.state_dict())
        return synapse_copy

    def forward_text(self, inputs: torch.Tensor):
        """ Local forward inputs through the bittensor.Synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by the synapse.
        """
        raise NotImplementedError

    def forward_image(self, inputs: torch.Tensor):
        r""" Forward pass inputs through the bittensor.synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    batch_size list of image tensors. (batch index, sequence_len, channel, row, col) produced for images
                    by calling PIL.toTensor().
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.network_size)`, `required`): 
                    Output encoding of inputs produced by the synapse.
        """
        raise NotImplementedError

    def forward_tensor(self, inputs: torch.Tensor):
        """ Forward tensor inputs through the bittensor.synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Batch_size length sequences of tensors.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by the synapse.
        """
        raise NotImplementedError

    def call_forward(self, inputs: torch.Tensor,
                     modality: bittensor_pb2.Modality) -> torch.Tensor:
        """
        Apply forward pass to the bittensor.synapse given inputs and modality.
        """
        # TODO(const): check schema (inputs, input_schema)
        with torch.no_grad():
            try:
                if modality == bittensor_pb2.Modality.TEXT:
                    outputs = self.forward_text(inputs)
                elif modality == bittensor_pb2.Modality.IMAGE:
                    outputs = self.forward_image(inputs)
                elif modality == bittensor_pb2.Modality.TENSOR:
                    outputs = self.forward_tensor(inputs)
                else:
                    raise NotImplementedError
            except NotImplementedError:
                # Input modality not implemented.
                # Returns None.
                return None
            except Exception as e:
                logger.error(e)
                return None
        return outputs

    def call_backward(self, inputs: object,
                      grads: torch.Tensor) -> torch.Tensor:
        """
        Apply a backward pass to the nn.module given grads and inputs.
        """
        # NOTE(const): removing gradient application here, needs to be replaced with gradient queueing.
        # with torch.enable_grad():
        #    outputs = self.forward(inputs)
        #    torch.autograd.backward(outputs, grad_tensors=grads.to(self.device), create_graph=False, retain_graph=False)
        #    self.apply_gradients()
        # TODO(const): check instance type.
        return torch.zeros((1, 1))

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        pass
        #self.optimizer.step()
        #self.optimizer.zero_grad()
