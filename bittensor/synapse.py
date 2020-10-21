from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional

import bittensor
from bittensor import bittensor_pb2


class Synapse(nn.Module):
    """ Bittensor synapse class.
    """

    def __init__(self):
        super().__init__()
        self._synapse_key = bittensor.Crypto.public_key_to_string(
            bittensor.Crypto.generate_private_ed25519().public_key())
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def synapse_key(self) -> str:
        return self._synapse_key

    def set_synapse_key(self, key):
        self._synapse_key = key

    def deepcopy(self, config = None):
        """ Returns a copy of this synapse by passing the model params to load_state_dict.

            config: (:obj:`config.class`, `optional`, defaults to model.config): 
                    model config used to re-init the model.

            Returns:
                synapse_copy (:obj:`self.__class__`, `required`): 
                    Deep copy synapse object.
        """
        SynapseClass = self.__class__
        
        model_config = None
        if config == None:
            # If no passed config, try model config.
            if self.config == None:
                raise ValueError('Deep copy requires a passed model config object or a member model.config')
            model_config = self.config
        synapse_copy = SynapseClass(model_config)
        synapse_copy.load_state_dict(self.state_dict())
        synapse_copy.set_synapse_key(self._synapse_key)
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
