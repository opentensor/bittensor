import argparse
from loguru import logger
from munch import Munch
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

import bittensor
from bittensor import bittensor_pb2

class SynapseOutput(object):
    """ Synapse output container.
        loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
            Total loss acumulation used by loss.backward()

        local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
            Hidden layer encoding produced using local_context.

        local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__vocab_size__)`, `optional`):
            Target predictions produced using local_context. 

        local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
            Target loss using local_context.

        remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `optional`): 
            Hidden layer encoding produced using the remote_context.

        remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,  bittensor.__vocab_size__)`, `optional`):
            Target predictions using the remote_context.

        remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
            Target oss using the remote_context.

        distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
            Distillation loss between local_context and remote_context.
    """
    def __init__(   
                self,
                loss: torch.Tensor = None,
                local_hidden: torch.Tensor  = None,
                local_target: torch.Tensor = None,
                local_target_loss: torch.Tensor = None,
                remote_hidden: torch.Tensor = None, 
                remote_target: torch.Tensor = None,
                remote_target_loss: torch.Tensor = None,
                distillation_loss: torch.Tensor = None,
        ):
        self.loss = loss
        self.local_hidden = local_hidden
        self.local_target = local_target
        self.local_target_loss = local_target_loss
        self.remote_hidden = remote_hidden
        self.remote_target = remote_target
        self.remote_target_loss = remote_target_loss
        self.distillation_loss = distillation_loss


class Synapse(nn.Module):
    """ Bittensor synapse class. 

    """

    def __init__(   self,
                    config: Munch,
                    session):
        r""" Init synapse module.

            Args:
                config (:obj:`SynapseConfig`, `required`): 
                    Base synapse config configuration class.

                session (:obj:`bittensor.BTSession`, `optional`): 
                    bittensor training session.
        """
        super().__init__()

        self.config = config
        self.session = session
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        return config

    def deepcopy(self):
        """ Returns a copy of this synapse by passing the model params to load_state_dict.

            Returns:
                synapse_copy (:obj:`self.__class__`, `required`): 
                    Deep copy synapse object.
        """
        SynapseClass = self.__class__
        synapse_copy = SynapseClass(self.config, self.session)
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
