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
""" Advanced server neuron.

Example:
    $ python miners/text/advanced_server/main.py

"""
from time import time
import bittensor
import torch
import wandb
import pandas
import datetime
import traceback
import sys
import os

from loguru import logger; logger = logger.opt(colors=True)
from torch.nn.utils import clip_grad_norm_
from datetime import datetime,timedelta
from threading import Lock
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def serve( 
    config, 
    gp_server= None, 
    subtensor = None,
    wallet = None, 
    metagraph = None,
    axon = None
):
    config.to_defaults()

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
        ).load().sync().save()
    else: 
        metagraph.load().sync().save()

    # Instantiate the model we are going to serve on the network.
    # Creating a threading lock for updates to the model
    mutex = Lock()
    gp_server = gp_server.to(gp_server.device)
    
    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": gp_server.parameters()} ],
        lr = config.neuron.learning_rate,
        momentum = config.neuron.momentum,
    )
    bittensor.tokenizer() 
    timecheck = {}

    n_topk_peer_weights = subtensor.min_allowed_weights
    # Define our forward function.
    def forward_text ( inputs_x ):
        r""" Forward function that is called when the axon recieves a forward request from other peers
            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """ 
        return gp_server.encode_forward( inputs_x.to(gp_server.device) )

    # Define our backward function.
    def backward_text (inputs_x, grads_dy ):
        r"""Backwards function that is called when the axon recieves a backwards request from other peers.
            Updates the server parameters with gradients through the chain.

            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                    
        """
        # -- normalized grads -- 
        grads_dy = grads_dy/(grads_dy.sum() + 0.00001)
        
        with mutex:
            outputs_y = gp_server.encode_forward( inputs_x.to(gp_server.device) )
            with torch.autograd.set_detect_anomaly(True):
                torch.autograd.backward (
                    tensors = [ outputs_y ],
                    grad_tensors = [ grads_dy.to(gp_server.device) ],
                    retain_graph=True
                )
            logger.info('Backwards axon gradient applied')

        gp_server.backward_gradients += inputs_x.size(0)
       
    def priority(pubkey:str, request_type:bittensor.proto.RequestType, inputs_x) -> float:
        r"""Calculates the priority on requests based on stake and size of input

            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """
        try:        
            uid = metagraph.hotkeys.index(pubkey)
            priority = metagraph.S[uid].item()/ sys.getsizeof(inputs_x)
        
        except:
            # zero priority for those who are not registered.
            priority =  0

        return priority

    def blacklist(pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """

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
            if request_type == bittensor.proto.RequestType.FORWARD:
                if metagraph.S[uid].item() < config.neuron.blacklist.stake.forward:
                    raise Exception('Stake blacklist')

                return False

            elif request_type == bittensor.proto.RequestType.BACKWARD:
                if metagraph.S[uid].item() < config.neuron.blacklist.stake.backward:
                    raise Exception('Stake blacklist')

                return False
        
        def validator_check():

            uid = metagraph.hotkeys.index(pubkey)
            if (metagraph.W[uid] >0).sum() ==n_topk_peer_weights:
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
                    raise Exception('Time blacklist')
            else:
                timecheck[pubkey] = current_time
                return False

        # Blacklist checks
        try:
            registration_check()

            stake_check()

            time_check()

            validator_check()
            
            return False

        #blacklisted
        except Exception as e:
            return True

    if axon == None: 
        # Create our axon server
        axon = bittensor.axon (
            config = config,
            wallet = wallet,
            forward_text = forward_text,
            backward_text = backward_text,
            blacklist = blacklist,
            priority = priority
        ) 

    # Training Data
    dataset = bittensor.dataset(config=config)

    # load our old model
    if not config.neuron.restart :
        gp_server.load(config.neuron.full_path)

    if config.wandb.api_key != 'default':
        # --- Init Wandb.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.neuron.full_path
        )

    nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)

    # --- last sync block 
    last_sync_block = subtensor.get_current_block()
    last_set_block = last_sync_block

    # -- Main Training loop --
    try:
        # -- download files from the mountain
        data = next(dataset)

        # --- creating our chain weights
        chain_weights = torch.zeros(metagraph.n)
        uid = nn.uid
        chain_weights[uid] = 1 

        # --  serve axon to the network.
        axon.start().serve(subtensor = subtensor)
        
        while True:
            
            # --- Check registration and optionally re-register
            nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
            if not wallet.is_registered( subtensor = subtensor ):
                wallet.register( subtensor = subtensor )
                axon.serve( subtensor = subtensor ) # Re-serve the erased axon data.
                nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
                

            # --- Run 
            current_block = subtensor.get_current_block()
            end_block = current_block + config.neuron.blocks_per_epoch
            interation = 0

            # --- Training step.
            while end_block >= current_block:
                if current_block != subtensor.get_current_block():
                    loss, _ = gp_server( next( dataset ).to(gp_server.device) )
                    if interation > 0 : 
                        losses += loss
                    else:
                        losses = loss
                    interation += 1
                    current_block = subtensor.get_current_block()

        
            #Custom learning rate
            if gp_server.backward_gradients > 0:
                optimizer.param_groups[0]['lr'] =  1/(gp_server.backward_gradients)
            else:
                optimizer.param_groups[0]['lr'] =  0.1
            
            # --- Update parameters
            if interation != 0 or gp_server.backward_gradients != 0:
                with mutex:
                    logger.info('Backpropagation Started')
                    if interation != 0:
                        losses.backward()
                    clip_grad_norm_(gp_server.parameters(), 1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info('Backpropagation Successful: Model updated')

            nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)

            gp_server.backward_gradients = 0
            # --- logging data
            wandb_data = {
                'block': end_block,
                'loss': losses.cpu().item()/interation,
                'stake': nn.stake,
                'rank': nn.rank,
                'incentive': nn.incentive,
                'trust': nn.trust,
                'consensus': nn.consensus,
                'incentive': nn.incentive,
                'dividends': nn.dividends,
                'emission':  nn.emission,
            } 
            bittensor.__console__.print('[green]Current Status:[/green]', wandb_data)

            # Add additional wandb data for axon, metagraph etc.
            if config.wandb.api_key != 'default':

                df = pandas.concat( [
                    bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                    bittensor.utils.indexed_values_to_dataframe( prefix = 's_i'.format(nn.uid), index = metagraph.uids, values = metagraph.S ),
                    axon.to_dataframe( metagraph = metagraph ),
                ], axis = 1)
                df['uid'] = df.index
                stats_data_table = wandb.Table( dataframe = df ) 
                wandb_info_axon = axon.to_wandb()                
                wandb.log( { **wandb_data, **wandb_info_axon }, step = current_block )
                wandb.log( { 'stats': stats_data_table }, step = current_block )
                wandb.log( { 'axon_query_times': wandb.plot.scatter( stats_data_table, "uid", "axon_query_time", title="Axon Query time by UID") } )
                wandb.log( { 'in_weights': wandb.plot.scatter( stats_data_table, "uid", 'w_i_{}'.format(nn.uid), title="Inward weights by UID") } )
                wandb.log( { 'stake': wandb.plot.scatter( stats_data_table, "uid", 's_i', title="Stake by UID") } )
                
            # Save the model
            gp_server.save(config.neuron.full_path)
            
            if current_block - last_set_block > config.neuron.blocks_per_set_weights:
                
                # --- Setting weights
                try: 
                    last_set_block = current_block
                    # Set self weights to maintain activity.
                    chain_weights = torch.zeros(metagraph.n)
                    chain_weights [ uid ] = 1 
                    did_set = subtensor.set_weights(
                        uids=metagraph.uids,
                        weights = chain_weights,
                        wait_for_inclusion = False,
                        wallet = wallet,
                    )
                    
                    if did_set:
                        logger.success('Successfully set weights on the chain')
                    else:
                        logger.error('Failed to set weights on chain. (Timeout)')
                except Exception as e:
                    logger.error('Failure setting weights on chain with error: {}', e)


            if current_block - last_sync_block > config.neuron.metagraph_sync:
                metagraph.sync()
                last_sync_block = current_block


    except KeyboardInterrupt:
        # --- User ended session ----
        axon.stop()
        dataset.close()
        
    except Exception as e:
        # --- Unknown error ----
        logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())

