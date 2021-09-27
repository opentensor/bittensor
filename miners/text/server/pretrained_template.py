#!/bin/python3
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
""" The Exodus base client.

Example:
    $ python miners/text/server/template_client.py

"""
import argparse
from bittensor._metagraph.metagraph_impl import Metagraph
from logging import Logger, raiseExceptions
from loguru import logger; logger = logger.opt(colors=True)
import bittensor
import torch
import time
import wandb
import datetime
from qqdm import qqdm
from transformers import AutoModel,AutoTokenizer,AutoConfig
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from threading import Thread, Lock

import os
import torch.nn.functional as F

class server(torch.nn.Module):
    def __init__(self, 
                config: 'bittensor.config' = None,
                pretrained: bool = None,
                model_name: str = None,
                padding: bool =None, 
                interpolate: bool =None,
                inter_degree: str = None,
                model = None,
                tokenizer = None,
                mapping_function = None,
                token_remap = None,
                checking= None):
        r"""" Creates a server that serves up a pretrained miner on the bittensor network
        Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.server.config()
                pretrained (:obj:bool , `optional`):
                    if the model should pretrained or not
                model_name (:obj:string , `optional`):
                    name of the pretrained model from huggingface to use
                padding (:obj:bool, `optional`):
                    If the server should pad out to match the hidden units that the bittensor network is using
                    If set to False, it will instead create a mapping layer to do the same thing.
                interpolate (:obj:bool, `optional`):
                    If the server should interpolate between sequence length differences.
                    If set to false, there should be a mapping function that takes care of the differnces
                inter_degree (:obj:str, `optional`):
                    The Interpolate algorithm (nearest | linear | bilinear | bicubic | trilinear | area)
                model (:obj:torch.module, `optional`):
                    Overrides the huggingface pretrained model with your own pretrained model
                tokenizer (:obj:huggingface.tokenizer, `optional`):
                    Overrides the huggingface tokenizer with your tokenizer
                mapping_function (:obj:Callable, `optional`):
                    Custom mapping function that maps between sequence length differences between tokenizers
                token_remap (:obj:Callable, `optional`):
                    Custom function that maps between tokenizers (defaults to self.remapping_token)
        """
        super(server, self).__init__()
        if config == None: config = server.config()
        self.config = config;print(config)
        
        #setting up pretrained model
        self.model_name = model_name if model_name != None else config.server.model_name
        self.pretrained = pretrained if pretrained != None else config.server.pretrained
        if self.pretrained == True:
            self.pre_model = model if model != None else AutoModel.from_pretrained(self.model_name)
            self.tokenizer = tokenizer if tokenizer != None else AutoTokenizer.from_pretrained(self.model_name)
        elif self.pretrained == False:
            self.pre_model = model if model != None else AutoModel.from_config(AutoConfig.from_pretrained(self.model_name))
            self.tokenizer = bittensor.tokenizer()

        #parameters of the models
        self.final_dim =  bittensor.__network_dim__
        self.pre_dimension = self.pre_model.config.hidden_size
        self.device = config.server.device
        self.padding = padding if padding != None else config.server.padding
        self.interpolate = interpolate if interpolate != None else config.server.interpolate
        self.inter_degree = inter_degree if inter_degree != None else config.server.inter_degree
        self.checking = checking if checking != None else config.server.checking
        self.mapping_function= mapping_function
        self.token_remap = token_remap if token_remap != None else self.remapping_token
        self.axon = None


        if padding == False:
            self.mapping = torch.nn.Linear( self.pre_dimension, self.final_dim)

        self.decoder = torch.nn.Linear( self.final_dim, bittensor.__vocab_size__ , bias=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        
        #checking if the parameters of the server makes sense
        if self.checking and pretrained == True:
            self.check()
        
        
    def forward(self, inputs,tokenizer=None):
        """
            Forward pass through the whole server model. Returns the loss and decoded predictions.

            Args:
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                tokenizer (:obj:'huggingface.tokenizer', optional):
                    The tokenizer which was used to tokenize the inputs
             Returns:
                loss (:obj:`torch.FloatTensor`):
                    MLM loss from the inputs
                decoded_targets (:obj:`torch.FloatTensor`):
                    Decoded predictions of the next token in the sentence.

        """
        decoded_targets = self.decoder(self.encode_forward(inputs,tokenizer))
        
        shift_logits = decoded_targets[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()     
        loss = self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) ) 

        return loss, decoded_targets
    
    def encode_forward(self,inputs,tokenizer=None):
        r""" Forward pass through the pretrained model and possible mappings between hidden units. 
             The response tensor should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                tokenizer ( huggingface.tokenizer, `optional`):
                    The tokenizer which was used to tokenize the inputs

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        sen_len = inputs.size()
        inputs = self.token_remap(inputs,tokenizer)
        pre_hidden = self.pre_model(inputs).last_hidden_state

        if self.interpolate:
            down= F.interpolate(pre_hidden.unsqueeze(1),size=[sen_len[1],pre_hidden.size()[2]],mode=self.inter_degree).squeeze(1)
        elif self.mapping_function:
            down = self.mapping_function(pre_hidden)
        else:
            raise Exception('interpolation off but no mapping function found. Please attach a mapping function')

        if self.padding:
            padding_l = (self.final_dim-self.pre_dimension)//2
            padding_r = (self.final_dim-self.pre_dimension) - padding_l
            encoded_hidden = F.pad(down, (padding_l, padding_r),  "constant", 0)
        else:
            encoded_hidden = self.mapping(down)
        return encoded_hidden

    def remapping_token(self,input, old_tokenizer):
        r""" Default remapping of tokenizers; decodes the message and then remaps the message using a new tokenizer
            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                old_tokenizer ( huggingface.tokenizer, `required`):
                    The tokenizer which was used to tokenize the input  (defaults to bittensor tokenizer if none is given)
        """
        if old_tokenizer == None:
            old_tokenizer = bittensor.tokenizer()
        new_data = []
        for i in range(input.shape[0]):
            decoded = old_tokenizer.decode(input[i]) 
            hugging = self.tokenizer(decoded)
            new_data += [torch.LongTensor(hugging.input_ids)]
        new_data = pad_sequence(new_data,batch_first=True)
        return new_data
    
    def start(self,wallet,optimizer,metagraph,mutex=None, forward=None, backward=None,blacklist=None, single_thread= False):
        r""" Starts the server and subscribes to the chain. 
            Args:
                wallet ( :obj:`bittensor.wallet`, `required`):
                    bittensor wallet that is attached to the axon
                optimizer ( torch.optimizer, `required`):
                    The optimizer which is used to optimize the parameters of the model
        """
        if self.axon != None:
            self.axon.start().subscribe()
        else:
            self.mutex = mutex
            self.optimizer = optimizer
            if single_thread == False:
                self.axon = bittensor.axon (
                                wallet = wallet,
                                forward_text = forward if forward != None else self.forward_text,
                                backward_text = backward if backward != None else self.backward_text,
                                blacklist= blacklist if blacklist != None else self.blacklist,
                            )
                self.threadpool = bittensor.prioritythreadpool(config=self.config)

            elif single_thread == True:
                self.axon = bittensor.axon (
                                wallet = wallet,
                                forward_text = forward if forward != None else self.forward_text,
                                backward_text = backward if backward != None else self.backward_text,
                                blacklist= blacklist if blacklist != None else self.blacklist,
                            )

            self.metagraph = metagraph
            self.axon.start().subscribe()

    # Define our forward function.
    def forward_text (self, pubkey, inputs_x ):
        r""" Forward function that is called when the axon recieves a forward request from other peers
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """ 
        def call(inputs):
            return self.encode_forward( inputs )
        uid =self.metagraph.hotkeys.index(pubkey)
        priority = self.metagraph.S[uid].item()
        future = self.threadpool.submit(call,inputs=inputs_x.to(self.device),priority=priority)
        try:
            return future.result(timeout= self.config.server.timeout)
        except:
            raise TimeoutError('TimeOutError')

    # Define our forward function.
    def forward_text_single (self, pubkey, inputs_x ):
        r""" Single threaded version of the Forward function that is called when the axon recieves a forward request from other peers
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """ 

        return self.encode_forward( inputs_x )

    # Define our backward function.
    def backward_text (self, pubkey:str, inputs_x, grads_dy ):
        r"""Backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                    
        """
        def call(input,grad):
            with torch.enable_grad():
                with torch.autograd.set_detect_anomaly(True):
                    self.mutex.acquire()
                    outputs_y = self.encode_forward( input )
                    torch.autograd.backward (
                        tensors = [ outputs_y ],
                        grad_tensors = [ grad ]
                    )
                    self.mutex.release()
        uid =self.metagraph.hotkeys.index(pubkey)
        priority = self.metagraph.S[uid].item()
        future = self.threadpool.submit(call, input=inputs_x.to( self.device ), grad=grads_dy.to( self.device ), priority=priority)
        try:
            return future.result(timeout= self.config.server.timeout)
        except:
            raise TimeoutError('TimeOutError')

    def blacklist(self,pubkey:str) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
        Currently, this is not turned on.
        """
        uid =self.metagraph.hotkeys.index(pubkey)
        if self.metagraph.S[uid].item() < self.config.server.blacklist:
            return True
        else:
            return False

    def check(self):
        r"""Checks the server settings
        """
        assert self.tokenizer.name_or_path == self.pre_model.name_or_path, 'incorrect model ({}) and tokenizer ({})'.format(self.pre_model.name_or_path,self.tokenizer.name_or_path)
        if self.interpolate == False:
            assert self.mapping_function != None, 'Incorrect Settings; needs atleast one mapping function for sequence length changes'

    @staticmethod
    def config ():
        parser = argparse.ArgumentParser()
        parser.add_argument('--server.learning_rate', type=float, help='Training initial learning rate.', default=0.1)
        parser.add_argument('--server.momentum', type=float, help='optimizer momentum.', default=0.8)
        parser.add_argument('--server.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0)
        parser.add_argument('--server.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--server.model_name', type=str, help='pretrained model from hugging face',default='gpt2')
        parser.add_argument('--server.pretrained', type=bool, help='if the model should be pretrained',default='True')
        parser.add_argument('--server.padding', type=bool, help='To pad out final dimensions',default='True')
        parser.add_argument('--server.interpolate', type=bool, help='To interpolate between sentence length',default='True')
        parser.add_argument('--server.inter_degree', type=str, help='Interpolate algorithm (nearest | linear | bilinear | bicubic | trilinear | area)', default='nearest')
        parser.add_argument('--server.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='template_server')
        parser.add_argument('--server.checking', type=bool, help='To check if server settings are correct',default='True')
        parser.add_argument('--server.timeout', type=int, help='Number of seconds to wait for axon request', default=10)
        parser.add_argument('--server.blacklist', type=float, help='Amount of stake (tao) in order not to get blacklisted', default=0)


        bittensor.wallet.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.prioritythreadpool.add_args( parser )

        return bittensor.config( parser )

def main( config ):

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config)

    # Load/Create our bittensor wallet.
    wallet = bittensor.wallet( config = config ).create()

    # Load/Sync/Save our metagraph.
    metagraph = bittensor.metagraph ( 
        subtensor = subtensor
    ).load().sync().save()

    # Instantiate the model we are going to serve on the network.
    # Miner training device.
    mutex = Lock()
    gp_server = server(config=config)
    
    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": gp_server.parameters()} ],
        lr = config.server.learning_rate,
        momentum = config.server.momentum,
    )

    # Create our axon server and subscribe it to the network.
    gp_server.start(wallet,optimizer,mutex,metagraph)
    

    # Training Data
    dataload = bittensor.dataloader()
    full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.server.name ))
    bittensor.logging( config = config,logging_dir = full_path)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # --- Init Wandb.
    bittensor.wandb(
        config = config,
        cold_pubkey = wallet.coldkeypub,
        hot_pubkey = wallet.hotkey.public_key,
        root_dir = full_path
    )
    chain_weights =torch.zeros(metagraph.n)
    try:
        # --- Run 
        for epoch in range(10000):
            epoch_loss = 0
            epoch_batches = dataload.dataloader(epoch_length=10)
            for iteration, inputs in enumerate(epoch_batches):

                mutex.acquire()
                loss, _ = gp_server( inputs )
                loss.backward()
                clip_grad_norm_(gp_server.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                mutex.release()

                epoch_loss += loss.item()

            uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
            wandb_data = {
                'Epoch': epoch,
                'loss': epoch_loss/10,
                'stake': metagraph.S[ uid ].item(),
                'rank': metagraph.R[ uid ].item(),
                'incentive': metagraph.I[ uid ].item(),
            } 
            gp_server.metagraph.sync().save()
            wandb.log( wandb_data )
            logger.info(wandb_data)
            chain_weights[uid] = 1 

            try: 
                did_set = subtensor.timeout_set_weights(
                    timeout=10,
                    uids=metagraph.uids,
                    weights = chain_weights,
                    wait_for_inclusion = True,
                    wallet = wallet,
                )
            except Exception as e:
                logger.error('Failure setting weights on chain with error: {}', e)

    except KeyboardInterrupt:
        # --- User ended session ----
        gp_server.axon.stop()


if __name__ == "__main__":
    main( server.config() )