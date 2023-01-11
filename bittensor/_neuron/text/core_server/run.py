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
import time
import datetime
from threading import Lock
from datetime import datetime,timedelta
from loguru import logger; logger = logger.opt(colors=True)
from torch.nn.utils.rnn import pad_sequence

import wandb
import pandas
# Prometheus
from prometheus_client import Counter, Gauge, Histogram, Summary, Info, CollectorRegistry
# Torch
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

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

    # Set neuron.netuid to netuid
    if config.neuron.get('netuid') == None:
        if config.get('netuid') != None:
            config.neuron.netuid = config.netuid
        else:
            raise ValueError("config.neuron.netuid or config.netuid must be set.")

    config.neuron.netuid = config.netuid

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config) if subtensor == None else subtensor

    # Load/Create our bittensor wallet.
    if wallet == None:
        wallet = bittensor.wallet( config = config ).create().reregister(subtensor=subtensor, netuid = config.neuron.netuid) 
    else:
        wallet.reregister(subtensor=subtensor, netuid = config.neuron.netuid)

    # Load/Sync/Save our metagraph.
    if metagraph == None:
        metagraph = bittensor.metagraph ( 
            netuid = config.neuron.netuid,
        )
    
    metagraph.load().sync(netuid= config.neuron.netuid).save()

    # Create our optimizer.
    optimizer = torch.optim.SGD(
        [ {"params": model.parameters()} ],
        lr = config.neuron.learning_rate,
        momentum = config.neuron.momentum,
    )
    mutex = Lock()

    # --- Setup prometheus summaries.
    # These will not be posted if the user passes --prometheus.level OFF
    registry = CollectorRegistry()
    prometheus_counters = Counter('neuron_counters', 'Counter sumamries for the running server-miner.', ['neuron_counters_name'], registry=registry)
    prometheus_guages = Gauge('neuron_guages', 'Guage sumamries for the running server-miner.', ['neuron_guages_name'], registry=registry)
    prometheus_info = Info('neuron_info', "Info sumamries for the running server-miner.", registry=registry)
    prometheus_guages.labels( 'model_size_params' ).set( sum(p.numel() for p in model.parameters()) )
    prometheus_guages.labels( 'model_size_bytes' ).set( sum(p.element_size() * p.nelement() for p in model.parameters()) )
    prometheus_info.info ({
        'type': "core_server",
        'uid': str(metagraph.hotkeys.index( wallet.hotkey.ss58_address )),
        'netuid': config.neuron.netuid,
        'network': config.subtensor.network,
        'coldkey': str(wallet.coldkeypub.ss58_address),
        'hotkey': str(wallet.hotkey.ss58_address),
    })

    timecheck_dicts = {bittensor.proto.RequestType.FORWARD:{}, bittensor.proto.RequestType.BACKWARD:{}}
    n_topk_peer_weights = subtensor.min_allowed_weights

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
            priority = metagraph.S[uid].item()
        
        except:
            # zero priority for those who are not registered.
            priority =  0

        return priority

    def forward_generate( inputs_x:torch.FloatTensor, synapse, model_output = None):
        tokens = model.token_remap(inputs_x.to(model.device))
        output = model.pre_model.generate(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            max_length=max(tokens['input_ids'].shape[1] + 1, synapse.num_to_generate),
            num_beams=synapse.num_beams,
            no_repeat_ngram_size=synapse.no_repeat_ngram_size,
            early_stopping = synapse.early_stopping,
            do_sample=synapse.do_sample,
            top_p=synapse.top_p,
            num_return_sequences=synapse.num_return_sequences,
            temperature = synapse.temperature,
            repetition_penalty = synapse.repetition_penalty,
            length_penalty = synapse.length_penalty,
            max_time = synapse.max_time,
            num_beam_groups = synapse.num_beam_groups,
        )
        raw_texts = [model.tokenizer.decode(out) for out in output]
        tokens = [model.std_tokenizer.encode(raw_text, return_tensors="pt")[:,:synapse.num_to_generate].view(-1) for raw_text in raw_texts]
        bittensor_output = pad_sequence(tokens, batch_first=True)
        return None, model_output, bittensor_output

    def forward_hidden_state(inputs_x:torch.FloatTensor, synapse, model_output = None):
        with mutex:
            message, model_output, hidden = model.encode_forward(inputs_x.to(model.device), model_output=model_output)
        return message, model_output, hidden

    def forward_casual_lm(inputs_x:torch.FloatTensor, synapse, model_output = None):
        with mutex:
            message, model_output, logits = model.encode_forward_causallm(inputs_x.to(model.device), model_output=model_output)
        return message, model_output, logits

    def forward_casual_lm_next(inputs_x: torch.FloatTensor, synapse, model_output=None):
        with mutex:
            message, model_output, topk_token_phrases = model.encode_forward_causallmnext(inputs_x.to(model.device),
                                                                                        topk=synapse.topk,
                                                                                        model_output=model_output)
        # topk_token_phrases: [sum_b(sum_k(len(phrase_k) + 1)_b)] contains topk token phrases and probabilities
        #   Compacted 1-D tensor >= batch_size * (2 * topk + 1)
        return message, model_output, topk_token_phrases

    def optimizer_step():
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

                prometheus_counters.labels("blacklisted.registration").inc()

                raise Exception('Registration blacklist')

        # Check for stake
        def stake_check() -> bool:
            # Check stake.
            uid = metagraph.hotkeys.index(pubkey)
            if metagraph.S[uid].item() < config.neuron.blacklist.stake:
                prometheus_counters.labels("blacklisted.stake").inc()

                raise Exception('Stake blacklist')
            return False

        # Check for time
        def time_check():
            current_time = datetime.now()
            # Only check if the request are forward requests
            timecheck = timecheck_dicts[request_type]
            if pubkey in timecheck.keys():
                prev_time = timecheck[pubkey]
                if current_time - prev_time >= timedelta(seconds=config.neuron.blacklist.time):
                    timecheck[pubkey] = current_time
                else:
                    timecheck[pubkey] = current_time
                    prometheus_counters.labels("blacklisted.time").inc()

                    raise Exception('Time blacklist')
            else:
                timecheck[pubkey] = current_time
        
            return False

        # Black list or not
        try:
            registration_check()
            time_check()
            stake_check()            
            return False
        except Exception as e:
            prometheus_counters.labels("blacklisted").inc()
            return True
    
    def synapse_check(synapse, hotkey):
        """
            Custom synapse function to protect certain synapse functions depending on the stake and weight.
            Certain synapses require more compute than others. For instance, TEXT_SEQ_2_SEQ requires a significantly
            more commitment by the server than a requeset for TEXT_CAUSAL_LM_NEXT.

            Args:
                synapse (:obj:`bittensor.proto.SynapseArgs`, `required`): 
                    The proto message that contains additional args for individual synapse functions
                hotkey (:obj:`torch.FloatTensor`, `required`):
                    The hotkey that sent the request

        """
        ## Uid that sent the request
        incoming_uid = metagraph.hotkeys.index(hotkey)
        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            
            if metagraph.S[incoming_uid] < config.neuron.lasthidden_stake:
                return False
            
        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:

            if metagraph.S[incoming_uid] < config.neuron.causallm_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:

            if metagraph.S[incoming_uid] < config.neuron.causallmnext_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:

            if (metagraph.S[incoming_uid] < config.neuron.seq2seq_stake) and (metagraph.S[incoming_uid,  uid]):
                return False     
        else:
            return False

        return True

    def backward_callback(inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
        """
            The default backward callback when no callback is attached: Is used to call specific synapse functions

            Args:
                inputs_x (:obj:`torch.FloatTensor`, `required`): 
                    The inputs that will be passed to the synapse functions
                grads_dy (:obj:`torch.FloatTensor`, `required`):
                    The gradients that will be passed to the synapse functions
                synapses (:obj: list of bittensor.proto.SynapseArgs, 'Optional')
                    The proto message that contains additional args for individual synapse functions

            Returns:
                response_tensors: (:obj: list of bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                response_codes: (:obj: list of bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                response_messages: (:obj: list of strings, `required`)
                    return message associated with synapse call
        """
        # --- initialize response variables --- 
        response_tensors = []
        response_codes = []
        response_messages = []
        
        if not config.neuron.remote_train:
            return response_tensors, response_codes, response_messages

        # --- calling attached synapses ---
        with mutex and torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
            for index, synapse in enumerate(synapses):
                try:
                    if synapse.synapse_type in axon.synapse_callbacks and axon.synapse_callbacks[synapse.synapse_type] != None:
                        message, model_output, response_tensor = axon.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                        grads_dy_norm = grads_dy[index]/(grads_dy[index].sum() + 0.00001)
                        torch.autograd.backward (
                            tensors = [ response_tensor ],
                            grad_tensors = [ grads_dy_norm ],
                            retain_graph=True
                        )
                        # Only consider loss from causal LM next.
                        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:
                            model.remote_losses.append(model_output.loss)
                            model.remote_losses = model.remote_losses[-config.neuron.num_remote_loss:] if len(model.remote_losses) > config.neuron.num_remote_loss else model.remote_losses
                        model.backward_gradients_count += inputs_x[index].size(0)
                        response_tensors.append(None)
                        response_codes.append(bittensor.proto.ReturnCode.Success)
                        response_messages.append('Success')
                        
                    else:
                        response_tensors.append(None)
                        response_codes.append(bittensor.proto.ReturnCode.NotImplemented)
                        response_messages.append('Not Implemented')
                except Exception as e:
                    # --- Exception Hit in Synapse ---
                    response_tensors.append(None)
                    response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                    response_messages.append(str(e))

        return response_tensors, response_codes, response_messages

    # Create our axon server and subscribe it to the network.
    if axon == None:
        axon = bittensor.axon(
            config = config,
            wallet = wallet,
            netuid = config.neuron.netuid,
            synapse_checks=synapse_check,
            synapse_last_hidden = forward_hidden_state if model.config.neuron.lasthidden else None,
            synapse_causal_lm = forward_casual_lm if model.config.neuron.causallm else None,
            synapse_causal_lm_next = forward_casual_lm_next if model.config.neuron.causallmnext else None,
            synapse_seq_2_seq = forward_generate if model.config.neuron.seq2seq else None ,
            blacklist = blacklist if not model.config.neuron.disable_blacklist else None,
            priority = priority if not model.config.neuron.disable_priority else None,
        ).start().serve(subtensor=subtensor)
    
    axon.optimizer_step = optimizer_step
    axon.attach_backward_callback(backward_callback)
    # Training Data
    if config.neuron.local_train:
        dataset = bittensor.dataset(config=config)
        dataset.set_data_size(10, 64)
        data = next(dataset)

    # load our old model
    if not config.neuron.restart :
        model.load(config.neuron.full_path)

    if config.wandb.api_key != 'default':
        # --- Init Wandb.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.neuron.full_path
        )

    last_set_block = subtensor.get_current_block()
    blocks_per_set_weights = subtensor.validator_epoch_length(config.neuron.netuid) if config.neuron.blocks_per_set_weights == -1 else config.neuron.blocks_per_set_weights

    # --- Run Forever.
    while True:
        iteration = 0
        local_data = {}
        nn = subtensor.get_neuron_for_pubkey_and_subnet(wallet.hotkey.ss58_address, netuid = config.neuron.netuid)
        uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
        current_block = subtensor.get_current_block()
        end_block = current_block + config.neuron.blocks_per_epoch
        if config.neuron.local_train:
            # --- Training step.
            while end_block >= current_block:
                if current_block != subtensor.get_current_block() and axon.priority_threadpool.is_empty:
                    with mutex:
                        logger.info(f'local training\titeration: {iteration}\tstart')
                        loss, _ = model( next(dataset).to(model.device) )
                        if iteration > 0 : 
                            losses += loss
                        else:
                            losses = loss
                        iteration += 1
                        current_block = subtensor.get_current_block()
                        logger.info(f'local training\titeration: {iteration}\tloss: {loss}')
                else:
                    time.sleep(1)
            
            if iteration != 0:
                (losses/iteration).backward()
        
        else:
            while end_block >= current_block:
                time.sleep(12)
                current_block = subtensor.get_current_block()

        # --- Update parameters
        if (config.neuron.local_train and iteration > 0) or (config.neuron.remote_train and model.backward_gradients_count > 0):
            # Custom learning rate
            if model.backward_gradients_count > 0:
                optimizer.param_groups[0]['lr'] =  0.1/(model.backward_gradients_count)
            else:
                optimizer.param_groups[0]['lr'] =  0.1

            logger.info('Optmization Started')
            with mutex:
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            logger.info('Optimization Successful: Model updated')

            if (config.neuron.local_train and iteration > 0):
                local_data = {'local/loss': losses.detach().item() / iteration}

                if local_data['local/loss'] < model.best_loss:
                    model.best_loss = local_data['local/loss']
                    model.save(config.neuron.full_path)

            # Save it only when it gives a low average loss over a large sample size (config.neuron.num_remote_loss), default to 20. 
            elif (config.neuron.remote_train and len(model.remote_losses) >= config.neuron.num_remote_loss):
                local_data = {'local/remote_loss': sum(model.remote_losses) / len(model.remote_losses)}

                if local_data['local/remote_loss'] < model.best_remote_loss:
                    model.best_remote_loss = local_data['local/remote_loss']
                    model.save(config.neuron.full_path)

                model.remote_losses = []

            model.backward_gradients_count = 0
            
        wandb_data = {            
            'stake': nn.stake,
            'rank': nn.rank,
            'trust': nn.trust,
            'consensus': nn.consensus,
            'incentive': nn.incentive,
            'emission': nn.emission,
        }
        
        if config.wandb.api_key != 'default':

            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                axon.to_dataframe( metagraph = metagraph ),
            ], axis = 1)
            df['uid'] = df.index
            wandb_info_axon = axon.to_wandb()                
            wandb.log( { **wandb_data, **wandb_info_axon, **local_data }, step = current_block )
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

        # === Prometheus logging.
        prometheus_guages.labels("stake").set( nn.stake )
        prometheus_guages.labels("rank").set( nn.rank )
        prometheus_guages.labels("trust").set( nn.trust )
        prometheus_guages.labels("consensus").set( nn.consensus )
        prometheus_guages.labels("incentive").set( nn.incentive )
        prometheus_guages.labels("emission").set( nn.emission )

        if current_block - last_set_block > blocks_per_set_weights:
            bittensor.__console__.print('[green]Current Status:[/green]', {**wandb_data, **local_data})
            metagraph.sync(netuid=config.neuron.netuid)
            last_set_block = current_block
            if not config.neuron.no_set_weights:
                try: 
                    bittensor.__console__.print('[green]Current Status:[/green]', {**wandb_data, **local_data})
                    # Set self weights to maintain activity.
                    # --- query the chain for the most current number of peers on the network
                    chain_weights = torch.zeros(subtensor.subnetwork_n( netuid = config.neuron.netuid ))
                    chain_weights [ uid ] = 1 
                    did_set = subtensor.set_weights(
                        netuid = config.neuron.netuid,
                        uids=torch.arange(0,subtensor.subnetwork_n( netuid = config.neuron.netuid )),
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
