# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor
from transformers import AutoModel, AutoTokenizer, AutoConfig 

class GatingModel( torch.nn.Module ):
    """
    This class is a PyTorch module that encapsulates the gating model functionality.

        - The backward method runs a backward pass through the model using the mean squared error between the normalized scores and the normalized rewards as the loss function.
        - The forward method runs a forward pass through the model, encoding the input message and generating scores for each uid in the network. The scores are returned as a tensor.
    """

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        """
        Adds command line arguments to the parser that are used to configure the gating model.
        The arguments added are:
        - `--gating.model_name`: Name of the pre-trained transformer-based language model to use as the encoding layer for the gating model. (default: 'EleutherAI/gpt-neo-125m')
        - `--gating.num_uids`: Number of uids to gate on. (default: 4096)
        - `--gating.learning_rate`: Learning rate for the gating model optimizer. (default: 0.01)
        - `--gating.momentum`: Momentum for the gating model optimizer. (default: 0.9)
        """
        parser.add_argument('--gating.model_name', type=str, default='EleutherAI/gpt-neo-125m', help='Name of the model to use as the encoding layer for the gating model')
        parser.add_argument('--gating.num_uids', type=int, default=4096, help='Number of uids to gate on')
        parser.add_argument('--gating.learning_rate', type=float, default=0.01, help='Learning rate for the gating model')
        parser.add_argument('--gating.momentum', type=float, default=0.9, help='Momentum for the gating model')

    @classmethod
    def config ( cls ):
        """
        Returns a configuration object that contains the command line arguments for the gating model.
        """
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        """
        Validates the configuration object for the gating model.
        """
        pass

    def __init__( 
            self, 
            metagraph: 'bittensor.metagraph.Metagraph',
            config: 'bittensor.config' = None, 
            model_name: str = None,
            num_uids: int = None
        ):
        """
        Initializes the gating model.
        - `metagraph`: A reference to the Bittensor metagraph object.
        - `config`: Configuration object for the gating model. If `None`, the default configuration is used.
        - `model_name`: Name of the pre-trained transformer-based language model to use as the encoding layer for the gating model. If `None`, the default model name specified in the configuration is used.
        - `num_uids`: Number of uids to gate on. If `None`, the default number specified in the configuration is used.
        """
        super(GatingModel, self).__init__()
        if config is None: config = GatingModel.config()
        if model_name is not None: config.gating.model_name = model_name
        config.gating.num_uids = num_uids if num_uids is not None else metagraph.n
        self.config = config
        self.num_uids = config.gating.num_uids
        self.device = torch.device( self.config.neuron.device )
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.gating.model_name )
        self.model = AutoModel.from_config( AutoConfig.from_pretrained(self.config.gating.model_name) ) #TODO: add pretrained flag
        self.linear = torch.nn.Linear( self.model.config.hidden_size, config.gating.num_uids  )
        self.optimizer = torch.optim.SGD(
            [ {"params": self.parameters()} ],
            lr = self.config.gating.learning_rate,
            momentum = self.config.gating.momentum,
        )

    def backward( self, scores: torch.FloatTensor, rewards: torch.FloatTensor ): 
        """ Runs a backward pass through the model.
            Args:
                scores (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Scores for each uids as output by the gating model.
                rewards (:obj:`torch.FloatTensor` of shape :obj:`(metagraph.n)`):
                    Rewards for each uids as output by the reward model.
        """   
        normalized_scores = torch.nn.functional.softmax( scores, dim=0 ).to( self.device )
        nomralized_rewards = torch.nn.functional.softmax( rewards, dim=0 ).to( self.device )
        loss = torch.nn.functional.mse_loss( normalized_scores, nomralized_rewards.detach() )
        loss.backward()
        self.optimizer.step()

    def forward( self, message: str ) -> 'torch.FloatTensor':
        """ Runs a forward pass through the model.
            Args:
                message (:obj:`str`): 
                    text message to be encoded.
            Returns:
                scores (:obj:`torch.FloatTensor` of shape :obj:`(network_size)`):
                    Scores for each uids as output by the gating model.
        """
        inputs = self.tokenizer( message, return_tensors="pt" ).to( self.device )
        with torch.no_grad():
            hidden_states = self.model( **inputs ).last_hidden_state[0, -1, :]
        return self.linear( hidden_states )


