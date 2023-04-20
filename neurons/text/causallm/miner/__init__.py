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

""" Template server.

Example:
    $ import neurons
    $ neurons.text.core_server.neuron().run()
"""

import bittensor
import os
import sys

from .nucleus_impl import server
from prometheus_client import Counter, Gauge, Histogram, Summary, Info, CollectorRegistry
from threading import Lock
from loguru import logger; logger = logger.opt(colors=True)
import time

from datetime import datetime,timedelta
import wandb
import pandas

# Torch
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_

from rich import print
from rich.console import Console
from rich.style import Style

class neuron:
    r"""
    Creates a bittensor neuron that specializes in the serving. The template server miner
    serves a NLP model from huggingface on the bittensor network. By default, the model does 
    not train itself and thus requires less memory to run. 

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            lasthidden (:obj:bool, `optional`):
                lasthidden synapse control
            causallm (:obj:bool, `optional`):
                causallm synapse control
            causallmnext (:obj:bool, `optional`):
                causallmnext synapse control
            seq2seq (:obj:bittensor.metagraph, `optional`):
                seq2seq synapse control
            synapse_list (:obj:list of int, `optional`):
      

    Examples::
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> server = bittensor.neuron.text.core_server.neuron(subtensor=subtensor)
            >>> server.run()
    """
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,
        model: 'bittensor.neurons.text.core_server.server' = None, 
        lasthidden = None,
        causallm = None,
        causallmnext = None,
        seq2seq = None,
        synapse_list = None,
        netuid = None,
        blacklist_hotkeys = None,
    ):
        if config is None:
            config = server.config()

        config = config; 

        config.netuid = netuid if netuid != None else config.netuid

        if synapse_list != None:
            config.neuron.lasthidden = False
            config.neuron.causallm = False
            config.neuron.causallmnext = False
            config.neuron.seq2seq = False

            if bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE in synapse_list:
                config.neuron.lasthidden = True
            
            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM in synapse_list:
                config.neuron.causallm = True

            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT in synapse_list:
                config.neuron.causallmnext = True

            if bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ in synapse_list:
                config.neuron.seq2seq = True

        config.neuron.lasthidden = lasthidden if lasthidden != None else config.neuron.lasthidden
        config.neuron.causallm = causallm if causallm != None else config.neuron.causallm
        config.neuron.causallmnext = causallmnext if causallmnext is not None else config.neuron.causallmnext
        config.neuron.seq2seq = seq2seq if seq2seq != None else config.neuron.seq2seq
        config.neuron.blacklist.hotkeys = blacklist_hotkeys if blacklist_hotkeys != None else config.neuron.blacklist.hotkeys

        self.check_config( config )
        self.config = config

        bittensor.logging (
            config = config,
            logging_dir = config.neuron.full_path,
        )

        # --- Setup prometheus summaries.
        # These will not be posted if the user passes --prometheus.level OFF
        registry = CollectorRegistry()
        self.prometheus_counters = Counter('neuron_counters', 'Counter sumamries for the running server-miner.', ['neuron_counters_name'], registry=registry)
        self.prometheus_guages = Gauge('neuron_guages', 'Guage sumamries for the running server-miner.', ['neuron_guages_name'], registry=registry)
        self.prometheus_info = Info('neuron_info', "Info sumamries for the running server-miner.", registry=registry)
        self.config.to_prometheus()

        if self.config.netuid == None and self.config.subtensor.network != 'nakamoto':
            subtensor = bittensor.subtensor(config = config) if subtensor == None else subtensor
            self.config.netuid = subtensor.get_subnets()[0]

        self.mutex = Lock()
        self.model = server(config = config).to(config.neuron.device) if model == None else model
        self.subtensor = bittensor.subtensor(config = config) if subtensor == None else subtensor
        self.wallet = bittensor.wallet( config = config ) if wallet == None else wallet
        self.metagraph = bittensor.metagraph ( config = config, netuid = self.config.netuid) if metagraph == None else metagraph
        self.timecheck_dicts = {bittensor.proto.RequestType.FORWARD:{}, bittensor.proto.RequestType.BACKWARD:{}}

        self.config.neuron.max_batch_size = self.subtensor.validator_batch_size(netuid=self.config.netuid) if self.config.neuron.max_batch_size == -1 else self.config.neuron.max_batch_size
        self.config.neuron.max_sequence_len = self.subtensor.validator_sequence_length(netuid=self.config.netuid) if self.config.neuron.max_sequence_len == -1 else self.config.neuron.max_sequence_len

        if axon == None:
            axon = bittensor.axon(
                config = config,
                wallet = wallet,
                netuid = self.config.netuid,
                synapse_checks=self.synapse_check,
                synapse_last_hidden = self.forward_hidden_state if self.model.config.neuron.lasthidden else None,
                synapse_causal_lm = self.forward_casual_lm if self.model.config.neuron.causallm else None,
                synapse_causal_lm_next = self.forward_casual_lm_next if self.model.config.neuron.causallmnext else None,
                synapse_seq_2_seq = self.forward_generate if self.model.config.neuron.seq2seq else None ,
                blacklist = self.blacklist if not self.model.config.neuron.disable_blacklist else None,
                priority = self.priority if not self.model.config.neuron.disable_priority else None,
            )
        self.axon = axon
        self.query_data = {}
        
        # Init prometheus.
        # By default we pick the prometheus port to be axon.port - 1000 so that we can match port to server.
        bittensor.prometheus ( 
            config = config,
            wallet = self.wallet,
            netuid = self.config.netuid,
            port = config.prometheus.port if config.axon.port == bittensor.defaults.axon.port else config.axon.port - 1000
        )

        # Verify subnet exists
        if self.config.subtensor.network != 'nakamoto' and not self.subtensor.subnet_exists( netuid = self.config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {self.config.netuid} does not exist[/red]")
            sys.exit(1)

    @classmethod
    def config(cls):
        return server.config()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.prometheus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name), config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    def run(
            self,
        ):
        self.config.to_defaults()

        # Load/Create our bittensor wallet.
        self.wallet.reregister(subtensor=self.subtensor, netuid = self.config.netuid)
        self.metagraph.sync(netuid = self.config.netuid, subtensor=self.subtensor).save()

        # Create our optimizer.
        optimizer = torch.optim.SGD(
            [ {"params": self.model.parameters()} ],
            lr = self.config.neuron.learning_rate,
            momentum = self.config.neuron.momentum,
        )

        self.prometheus_guages.labels( 'model_size_params' ).set( sum(p.numel() for p in self.model.parameters()) )
        self.prometheus_guages.labels( 'model_size_bytes' ).set( sum(p.element_size() * p.nelement() for p in self.model.parameters()) )
        self.prometheus_info.info ({
            'type': "core_server",
            'uid': str(self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )),
            'netuid': self.config.netuid,
            'network': self.config.subtensor.network,
            'coldkey': str(self.wallet.coldkeypub.ss58_address),
            'hotkey': str(self.wallet.hotkey.ss58_address),
        })

        # Create our axon server and subscribe it to the network.
        self.axon.start().serve(subtensor=self.subtensor)
        self.axon.attach_backward_callback(self.backward_callback)


        # Training Data
        if self.config.neuron.local_train:
            self.dataset = bittensor.dataset(config=self.config)
            self.dataset.set_data_size(10, 64)
            data = next(self.dataset)

        # load our old model
        if not self.config.neuron.restart :
            self.model.load(self.config.neuron.full_path)

        if self.config.wandb.api_key != 'default':
            # --- Init Wandb.
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )

        last_set_block = self.subtensor.get_current_block()
        blocks_per_set_weights = self.get_blocks_per_set_weights()
        epoch_starting_successes = self.axon.stats.total_successes
        epoch_starting_requests = self.axon.stats.total_requests
        # --- Run Forever.
        while True:
            iteration = 0
            local_data = {}
            self.query_data = {}
            nn = self.get_neuron()
            uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
            current_block = self.subtensor.get_current_block()
            end_block = current_block + self.config.neuron.blocks_per_epoch

            if self.config.neuron.local_train:
                # --- Training step.
                while end_block >= current_block:
                    if current_block != self.subtensor.get_current_block() and self.axon.priority_threadpool.is_empty:
                        with self.mutex:
                            logger.info(f'local training\titeration: {iteration}\tstart')
                            loss, _ = self.model( next(self.dataset).to(self.model.device) )
                            if iteration > 0 : 
                                losses += loss
                            else:
                                losses = loss
                            iteration += 1
                            current_block = self.subtensor.get_current_block()
                            logger.info(f'local training\titeration: {iteration}\tloss: {loss}')
                    else:
                        time.sleep(1)
                
                if iteration != 0:
                    (losses/iteration).backward()
            
            else:
                while end_block > current_block:
                    time.sleep(12)
                    current_block = self.subtensor.get_current_block()

            # --- Update parameters
            if (self.config.neuron.local_train and iteration > 0) or (self.config.neuron.remote_train and self.model.backward_gradients_count > 0):
                # Custom learning rate
                if self.model.backward_gradients_count > 0:
                    optimizer.param_groups[0]['lr'] =  0.1/(self.model.backward_gradients_count)
                else:
                    optimizer.param_groups[0]['lr'] =  0.1

                logger.info('Optmization Started')
                with self.mutex:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                logger.info('Optimization Successful: Model updated')

                if (self.config.neuron.local_train and iteration > 0):
                    local_data = {'local/loss': losses.detach().item() / iteration}

                    if local_data['local/loss'] < self.model.best_loss:
                        self.model.best_loss = local_data['local/loss']
                        self.model.save(self.config.neuron.full_path)

                # Save it only when it gives a low average loss over a large sample size (config.neuron.num_remote_loss), default to 20. 
                elif (self.config.neuron.remote_train and len(self.model.remote_losses) >= self.config.neuron.num_remote_loss):
                    local_data = {'local/remote_loss': sum(self.model.remote_losses) / len(self.model.remote_losses)}

                    if local_data['local/remote_loss'] < self.model.best_remote_loss:
                        self.model.best_remote_loss = local_data['local/remote_loss']
                        self.model.save(self.config.neuron.full_path)

                    self.model.remote_losses = []

                self.model.backward_gradients_count = 0
                
            data = {            
                'stake': nn.stake,
                'rank': nn.rank,
                'trust': nn.trust,
                'consensus': nn.consensus,
                'incentive': nn.incentive,
                'emission': nn.emission,
            }
            
            if self.config.wandb.api_key != 'default':

                df = pandas.concat( [
                    bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = self.metagraph.uids, values = self.metagraph.W[:, uid] ),
                    self.axonaxon.to_dataframe( metagraph = self.metagraph ),
                ], axis = 1)
                df['uid'] = df.index
                wandb_info_axon = self.axon.to_wandb()                
                wandb.log( { **data, **wandb_info_axon, **local_data }, step = current_block )
                wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

            # === Prometheus logging.
            self.prometheus_guages.labels("stake").set( nn.stake )
            self.prometheus_guages.labels("rank").set( nn.rank )
            self.prometheus_guages.labels("trust").set( nn.trust )
            self.prometheus_guages.labels("consensus").set( nn.consensus )
            self.prometheus_guages.labels("validator_trust").set( nn.validator_trust )
            self.prometheus_guages.labels("incentive").set( nn.incentive )
            self.prometheus_guages.labels("emission").set( nn.emission )

            print(f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                f"{f'[magenta dim not bold]#{current_block}[/magenta dim not bold]'.center(16 + len('[magenta dim not bold][/magenta dim not bold]'))} | "
                f'[green not bold]{current_block - last_set_block}[/green not bold]/'
                f'[white not bold]{blocks_per_set_weights}[/white not bold] [dim]blocks/epoch[/dim] | '
                f'[bright_green not bold]{self.axon.stats.total_successes - epoch_starting_successes}[/bright_green not bold]'
                f'[white]/{self.axon.stats.total_requests - epoch_starting_requests}[/white] '
                f'[dim]Epoch Success [/dim]|'
                f'[dim][green] {self.axon.stats.total_successes}[/green]'
                f'[white]/{self.axon.stats.total_requests}[/white]'
                f' Total Success [/dim]|'
                f'[dim white not bold][yellow] {self.axon.stats.total_codes[bittensor.proto.ReturnCode.Name(2)]}[/yellow]'
                f'[white]/{self.axon.stats.total_requests}[/white]'
                f' Timeout [/dim white not bold]|'
                f'[dim white not bold][red] {self.axon.stats.total_requests - self.axon.stats.total_successes - self.axon.stats.total_codes[bittensor.proto.ReturnCode.Name(2)]}[/red]'
                f'[white]/{self.axon.stats.total_requests}[/white]'
                f' Error [/dim white not bold]|')


            if current_block - last_set_block > blocks_per_set_weights:
                self.metagraph.sync(netuid=self.config.netuid, subtensor = self.subtensor)
                last_set_block = current_block
                epoch_starting_successes = self.axon.stats.total_successes
                epoch_starting_requests = self.axon.stats.total_requests

                print(f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                    f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
                    f'[dim white not bold] [green]{str(nn.stake):.4}[/green] Stake [/dim white not bold]'
                    f'[dim white not bold]| [yellow]{nn.trust:.3}[/yellow] Trust [/dim white not bold]'
                    f'[dim white not bold]| [green]{nn.incentive:.3}[/green] Incentive [/dim white not bold]')

                if not self.config.neuron.no_set_weights:
                    try: 
                        # Set self weights to maintain activity.
                        # --- query the chain for the most current number of peers on the network
                        chain_weights = torch.zeros(self.get_neuron_num())
                        chain_weights [ uid ] = 1 
                        did_set = self.subtensor.set_weights(
                            uids=torch.arange(0,len(chain_weights)),
                            netuid = self.config.netuid,
                            weights = chain_weights,
                            wait_for_inclusion = False,
                            wallet = self.wallet,
                        )
                        if did_set:
                            logger.success('Successfully set weights on the chain')
                        else:
                            logger.error('Failed to set weights on chain. (Timeout)')
                        
                    except Exception as e:
                        logger.error('Failure setting weights on chain with error: {}', e)

    def synapse_check(self, synapse, hotkey, inputs_x=None):
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
        try:
            incoming_uid = self.metagraph.hotkeys.index(hotkey)
        except Exception as e:
            if self.config.neuron.blacklist_allow_non_registered:
                return False
            return True

        batch_size, sequence_len  =  inputs_x[0].size()
        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            if self.metagraph.S[incoming_uid] < self.config.neuron.lasthidden_stake \
                or (batch_size > self.config.neuron.max_batch_size) \
                or (sequence_len > self.config.neuron.max_sequence_len):
                return False
            
        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:
            if (self.metagraph.S[incoming_uid] < self.config.neuron.causallm_stake) \
                or (batch_size > self.config.neuron.max_batch_size) \
                or (sequence_len > self.config.neuron.max_sequence_len):
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:
            if (self.metagraph.S[incoming_uid] < self.config.neuron.causallmnext_stake) \
                or (batch_size > self.config.neuron.max_batch_size) \
                or (sequence_len > self.config.neuron.max_sequence_len):
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:
            if (self.metagraph.S[incoming_uid] < self.config.neuron.seq2seq_stake) \
                or (batch_size > self.config.neuron.max_batch_size) \
                or (sequence_len > self.config.neuron.max_sequence_len) \
                or (self.metagraph.W[incoming_uid,  self.uid]):
                return False     
        else:
            raise Exception('Unknown Synapse')

        return True

    def backward_callback(self, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
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
        
        if not self.config.neuron.remote_train:
            return response_tensors, response_codes, response_messages

        # --- calling attached synapses ---
        with self.mutex and torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
            for index, synapse in enumerate(synapses):
                try:
                    if synapse.synapse_type in self.axon.synapse_callbacks and self.axon.synapse_callbacks[synapse.synapse_type] != None:
                        message, model_output, response_tensor = self.axon.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                        grads_dy_norm = grads_dy[index]/(grads_dy[index].sum() + 0.00001)
                        torch.autograd.backward (
                            tensors = [ response_tensor ],
                            grad_tensors = [ grads_dy_norm ],
                            retain_graph=True
                        )
                        # Only consider loss from causal LM next.
                        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:
                            self.model.remote_losses.append(model_output.loss)
                            self.model.remote_losses = self.model.remote_losses[-self.config.neuron.num_remote_loss:] if len(self.model.remote_losses) > self.config.neuron.num_remote_loss else self.model.remote_losses
                        self.model.backward_gradients_count += inputs_x[index].size(0)
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


    def priority(self, pubkey:str, request_type:bittensor.proto.RequestType, inputs_x) -> float:
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
            uid = self.metagraph.hotkeys.index(pubkey)
            priority = self.metagraph.S[uid].item()
        
        except:
            # zero priority for those who are not registered.
            priority =  0

        return priority

    def forward_generate(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        tokens = self.model.token_remap(inputs_x.to(self.model.device))
        output = self.model.pre_model.generate(
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
        raw_texts = [self.model.tokenizer.decode(out) for out in output]
        tokens = [self.model.std_tokenizer.encode(raw_text, return_tensors="pt")[:,:synapse.num_to_generate].view(-1) for raw_text in raw_texts]
        bittensor_output = pad_sequence(tokens, batch_first=True)
        return None, model_output, bittensor_output

    def forward_hidden_state(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        with self.mutex:
            message, model_output, hidden = self.model.encode_forward(inputs_x.to(self.model.device), model_output=model_output)
        return message, model_output, hidden

    def forward_casual_lm(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        with self.mutex:
            message, model_output, logits = self.model.encode_forward_causallm(inputs_x.to(self.model.device), model_output=model_output)
        return message, model_output, logits

    def forward_casual_lm_next(self,inputs_x: torch.FloatTensor, synapse, model_output=None):
        with self.mutex:
            message, model_output, topk_token_phrases = self.model.encode_forward_causallmnext(inputs_x.to(self.model.device),
                                                                                        topk=synapse.topk,
                                                                                        model_output=model_output)
        # topk_token_phrases: [sum_b(sum_k(len(phrase_k) + 1)_b)] contains topk token phrases and probabilities
        #   Compacted 1-D tensor >= batch_size * (2 * topk + 1)
        return message, model_output, topk_token_phrases


    def blacklist(self, pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
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
            is_registered = pubkey in self.metagraph.hotkeys
            if not is_registered:
                if self.config.neuron.blacklist_allow_non_registered:
                    return False

                self.prometheus_counters.labels("blacklisted.registration").inc()

                raise Exception('Registration blacklist')

        # Check for stake
        def stake_check() -> bool:
            # Check stake.
            uid = self.metagraph.hotkeys.index(pubkey)
            if self.metagraph.S[uid].item() < self.config.neuron.blacklist.stake:
                self.prometheus_counters.labels("blacklisted.stake").inc()

                raise Exception('Stake blacklist')
            return False

        # Check for time
        def time_check():
            current_time = datetime.now()
            # Only check if the request are forward requests
            timecheck = self.timecheck_dicts[request_type]
            if pubkey in timecheck.keys():
                prev_time = timecheck[pubkey]
                if current_time - prev_time >= timedelta(seconds=self.config.neuron.blacklist.time):
                    timecheck[pubkey] = current_time
                else:
                    timecheck[pubkey] = current_time
                    self.prometheus_counters.labels("blacklisted.time").inc()

                    raise Exception('Time blacklist')
            else:
                timecheck[pubkey] = current_time
        
            return False

        # Check for hotkeys
        def hotkey_check():
            # Only check if the request are forward requests
            if (pubkey in self.config.neuron.blacklist.hotkeys):
                raise Exception('Hotkey blacklist')
            return False
        
        # Black list or not
        try:
            registration_check()
            time_check()
            stake_check()      
            hotkey_check()      
            return False, None
        except Exception as error:
            self.prometheus_counters.labels("blacklisted").inc()
            return True, error

    def get_neuron(self):
        if self.subtensor.network == 'nakamoto':
            nn = self.subtensor.neuron_for_pubkey(self.wallet.hotkey.ss58_address)
        else:
            nn = self.subtensor.get_neuron_for_pubkey_and_subnet(self.wallet.hotkey.ss58_address, netuid = self.config.netuid)
        return nn

    def get_neuron_num(self):
        if self.subtensor.network == 'nakamoto':
            n = self.subtensor.n()
        else:
            n = self.subtensor.subnetwork_n( netuid = self.config.netuid)
        return n
    
    def get_blocks_per_set_weights(self):
        blocks_per_set_weights = self.config.neuron.blocks_per_set_weights
        if blocks_per_set_weights == -1:
            if self.subtensor.network == 'nakamoto':
                blocks_per_set_weights = self.subtensor.validator_epoch_length
            else:
                blocks_per_set_weights = self.subtensor.validator_epoch_length(self.config.netuid)
        
        return blocks_per_set_weights