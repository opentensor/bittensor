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
    $ python miners/text/template_client.py

"""
import bittensor
import sys
import torch
import time
import wandb
import pandas
import datetime
from threading import Lock
from loguru import logger; logger = logger.opt(colors=True)
from datetime import datetime,timedelta
import torch.nn.functional as F

def serve( 
        config, 
        model,
        subtensor = None,
        wallet = None,
        axon= None,
        metagraph = None,
    ):
    config.to_defaults()
    model= model.to(model.device)

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config) if subtensor == None else subtensor

    # Load/Create our bittensor wallet.
    if wallet == None:
        wallet = bittensor.wallet( config = config ).create().register(subtensor=subtensor) 
    else:
        wallet.register(subtensor=subtensor)


    # Load/Sync/Save our metagraph.
    if metagraph == None:
        metagraph = bittensor.metagraph ( 
            subtensor = subtensor
        )
    
    metagraph.load().sync().save()

    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": model.parameters()} ],
        lr = config.neuron.learning_rate,
        momentum = config.neuron.momentum,
    )
    mutex = Lock()

    timecheck = {}
    n_topk_peer_weights = subtensor.min_allowed_weights

    def forward_generate( inputs_x:torch.FloatTensor, synapse):
        output = model.pre_model.generate(
            input_ids=inputs_x, 
            max_length=synapse.num_to_generate, 
            num_beams=synapse.num_beams, 
            no_repeat_ngram_size=synapse.no_repeat_ngram_size,
            early_stopping = synapse.early_stopping,
            do_sample=synapse.do_sample, 
            top_p=synapse.top_p, 
            num_return_sequences=synapse.num_return_sequences,
        )
        return output

    def forward_hidden_state(inputs_x, synapse):
        output = model.pre_model(inputs_x, output_hidden_states=True).hidden_states[-1]
        padding_r = (1024-output.size(2))
        encoded_hidden = F.pad(output, (0, padding_r),  "constant", 0)
        return encoded_hidden

    def forward_casual_lm(inputs_x, synapse):
        output = model.encode_forward_causallm(inputs_x.to(model.device))
        return output
    
    def optimizer_step():
        optimizer.step()
        optimizer.zero_grad()

    def backward_causal_lm(inputs_x, grads_dy):
        r"""Single threaded backwards function that is called when the axon receives
            a backwards request from other peers.
            Updates the server parameters with gradients through the chain.             
        """
        if config.neuron.training:
            with mutex:
                with torch.enable_grad():
                    with torch.autograd.set_detect_anomaly(True):
                        outputs_y = model.encode_forward_causallm(inputs_x.to(model.device))
                        torch.autograd.backward (
                            tensors = [ outputs_y ],
                            grad_tensors = [ grads_dy ]
                            )
                        optimizer.step()
                        optimizer.zero_grad()

    def backward_text ( inputs_x, grads_dy ):
        r"""Single threaded backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.
        """
        if config.neuron.training:
            with mutex:
                with torch.enable_grad():
                    with torch.autograd.set_detect_anomaly(True):
                        outputs_y = model.encode_forward( inputs_x )
                        torch.autograd.backward (
                            tensors = [ outputs_y ],
                            grad_tensors = [ grads_dy ]
                            )
                        optimizer.step()
                        optimizer.zero_grad()

    def blacklist(pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """
        # Check for registrations

        def registration_check():
            # If we allow non-registered requests return False = not blacklisted.
            is_registered = pubkey in metagraph.hotkeys
            if not is_registered:
                if config.neuron.blacklist_allow_non_registered:
                    
                    return False
                raise Exception('Registration blacklist')

        # Check for stake
        def stake_check() -> bool:
                
            # Check stake.
            uid = metagraph.hotkeys.index(pubkey)
            if metagraph.S[uid].item() < config.neuron.blacklist.stake:
                raise Exception('Stake blacklist')
            return False

        def validator_check():

            uid = metagraph.hotkeys.index(pubkey)
            if (metagraph.W[uid] >0).sum() >= n_topk_peer_weights:
                return False

            raise Exception('Validator blacklist')


        # Check for time
        def time_check():
            current_time = datetime.now()
            if pubkey in timecheck.keys():
                prev_time = timecheck[pubkey]
                if current_time - prev_time >= timedelta(seconds=config.neuron.blacklist.time):
                    timecheck[pubkey] = current_time
                    return False
                else:
                    timecheck[pubkey] = current_time
                    raise Exception('blacklist')
            else:
                timecheck[pubkey] = current_time
                return False

        # Black list or not
        try:
            #registration_check()

            #stake_check()

            #validator_check()
            
            return False

        except Exception as e:
            return True


    # Create our axon server and subscribe it to the network.
    if axon == None:
        axon = bittensor.axon (
            config = config,
            wallet = wallet,
            synapse_last_hidden = forward_hidden_state,
            synapse_causal_lm = forward_casual_lm,
            synapse_seq_2_seq = forward_generate,
            blacklist = blacklist,
        ).start().serve(subtensor=subtensor)
    
    axon.optimizer_step = optimizer_step
    
    if config.wandb.api_key != 'default':
        # --- Init Wandb.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.neuron.full_path
        )

    last_set_block = subtensor.get_current_block()


    # --- Run Forever.
    while True:
        
        current_block = subtensor.get_current_block()
        end_block = current_block + config.neuron.blocks_per_epoch
        while end_block >= current_block:
            time.sleep( bittensor.__blocktime__ )
            current_block = subtensor.get_current_block()

        nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
        uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        wandb_data = {
            'stake': nn.stake,
            'rank': nn.rank,
            'trust': nn.trust,
            'consensus': nn.consensus,
            'incentive': nn.incentive,
            'emission': nn.emission,
        }
        bittensor.__console__.print('[green]Current Status:[/green]', wandb_data)
        if config.wandb.api_key != 'default':

            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                axon.to_dataframe( metagraph = metagraph ),
            ], axis = 1)
            df['uid'] = df.index
            wandb_info_axon = axon.to_wandb()                
            wandb.log( { **wandb_data, **wandb_info_axon }, step = current_block )
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

        if current_block - last_set_block > config.neuron.blocks_per_set_weights:
            try: 
                last_set_block = current_block
                # Set self weights to maintain activity.
                # --- query the chain for the most current number of peers on the network
                chain_weights = torch.zeros(subtensor.n)
                chain_weights [ uid ] = 1 
                did_set = subtensor.set_weights(
                    uids=torch.arange(0,subtensor.n),
                    weights = chain_weights,
                    wait_for_inclusion = False,
                    wallet = wallet,
                )
                
                metagraph.sync()
                if did_set:
                    logger.success('Successfully set weights on the chain')
                else:
                    logger.error('Failed to set weights on chain. (Timeout)')
            except Exception as e:
                logger.error('Failure setting weights on chain with error: {}', e)
