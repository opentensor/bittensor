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

""" Main Validator script.

Example:
    $ python3 bittensor/_neurons/text/core_validator/main.py ...
"""

import argparse
import time
import bittensor
import torch
import os
import math
import random
import sys
from rich import print
from rich.console import Console
from rich.traceback import install
from typing import Dict, Any, Tuple, List
import bittensor.utils.networking as net
from types import SimpleNamespace

from bittensor._neuron.text.log_utilities import ValidatorLogger
from bittensor._neuron.text.core_validator.model import nucleus 

from loguru import logger
from threading import Lock
import bittensor.utils.networking as net

logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

class neuron:
    def __init__( 
        self, 
        config: 'bittensor.Config' = None,
        wallet: 'bittensor.Wallet' = None,
        subtensor: 'bittensor.Subtensor' = None,
        metagraph: 'bittensor.Metagraph' = None,
        dataset: 'bittensor.dataset' = None,
    ):
        # === Set up Config ===
        self.config = neuron.config() if config == None else config
        neuron.check_config( self.config )
        self.config.to_defaults()
         
        # === Create Bittensor objects ===
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.vlogger = ValidatorLogger( config = self.config )
        self.subtensor = bittensor.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.wallet = bittensor.wallet ( config = self.config ) if wallet == None else wallet
        self.metagraph = bittensor.metagraph ( config = self.config ) if metagraph == None else metagraph
        self.device = torch.device ( device = self.config.neuron.device )    
        self.nucleus = nucleus ( config = self.config, device = self.device).to( self.device )
        self.dataset = bittensor.dataset(config=self.config, 
                                          batch_size=self.subtensor.validator_batch_size(self.config.netuid),
                                          block_size=self.subtensor.validator_sequence_length(self.config.netuid) + self.config.neuron.validation_len + self.subtensor.validator_prune_len(netuid=self.config.netuid)
                        )if dataset is None else dataset
        
        self.loss = None
        self.loss_agg_mutex = Lock()

        # === Neuron statistics variables ===
        self.neuron_stats = {}  # neuron statistics dict of dicts: [uid] -> {'stat1': val1, 'stat2': val2, ...}
        self.neuron_hotkeys = []  # keep neuron hotkeys to compare and check for changes after metagraph.sync()
        self.alpha = 0.1  # EMA coefficient in [0, 1], higher alpha discounts older observations faster
        self.weight_key = 'shapley_values_nxt'  # stat key + ! to calculate neuron weights with
        self.synapse_keys = ['shapley_values_nxt']
        
        # === Load last saved validator values from the file system
        if not self.config.neuron.restart:
            self.load()
        
    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        nucleus.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.prometheus.check_config( config )
        
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'

        # check netuid
        subtensor = bittensor.subtensor(config)
        config.netuid = subtensor.get_subnets()[0] if config.netuid == None else config.netuid
        if not subtensor.subnet_exists( netuid = config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {config.netuid} does not exist[/red]")
            sys.exit(1)
    
        # check scaling law power
        config.nucleus.scaling_law_power = subtensor.scaling_law_power(netuid=config.netuid) if config.neuron.scaling_law_power == -1 else config.neuron.scaling_law_power
        config.nucleus.synergy_scaling_law_power = subtensor.synergy_scaling_law_power(netuid=config.netuid) if config.neuron.synergy_scaling_law_power == -1 else config.neuron.synergy_scaling_law_power
        
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_validator')
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.1 )
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8 )
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch, -1 value means we use the chain value.', default = -1 )
        parser.add_argument('--neuron.epochs_until_reset', type=int, help='Number of epochs before weights are reset.', default = -1 )
        parser.add_argument('--neuron.prune_len', type=int, help='Number of tokens to prune from each validation input sequence.  (default value: -1, pulling from subtensor directly)', default=-1)
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0 )
        parser.add_argument('--neuron.track_hotkey_changes', action='store_true', help='If True, track hotkey changes.', default=False)
        parser.add_argument('--neuron.restart', action='store_true', help='If True, reset neuron_stats and validate anew.', default=False)
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )
        parser.add_argument('--neuron.wait_for_finalization', action='store_true', help='''when setting weights the miner waits for trnasaction finalization.''', default=False)
        parser.add_argument('--neuron.forward_num', type=int, help='''How much forward request before a backward call.''', default=3)
        parser.add_argument('--neuron.validation_synapse', type=str, help='''Synapse used for validation.''', default='TextCausalLMNext', choices = ['TextCausalLMNext', 'TextCausalLM'])
        parser.add_argument('--neuron.exclude_quantile', type=float, help='Exclude the lowest quantile from weight setting. (default value: -1, pulling from subtensor directly)', default=-1)
        parser.add_argument('--neuron.topk', type=int, help='the number of peers queried during each remote forward call', default = 20 )
        parser.add_argument('--neuron.validation_len', type=int, help='Number of tokens to holdout for phrase validation beyond sequence context.', default=8)
        parser.add_argument('--neuron.scaling_law_power', type=float, help='Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5. (default value: -1, pulling from subtensor directly)', default=-1)
        parser.add_argument('--neuron.synergy_scaling_law_power', type=float, help='Power for synergy modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5. (default value: -1, pulling from subtensor directly)', default=-1)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )    
        
        # Netuid Arg
        parser.add_argument('--netuid', type=int , help='Subnet netuid', default=1)

        bittensor.wallet.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.axon.add_args( parser )
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )
    
    def __str__(self) -> str:
        return (f'[bold]UID {self.uid}[/bold] \[{net.get_external_ip}] '
                f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/'
                f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])')

    def __del__(self):
        if getattr(self, 'dataset', None) is not None:
            self.dataset.close()
        
        if getattr(self, 'dendrites', None) is not None:
            for dendrite in self.dendrites:
                dendrite.__del__()
    
    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        r""" Close down neuron.
        """
        print(exc_type, exc_value, exc_traceback)
        self.__del__()

    def __enter__(self):
        r""" Sanity checks and begin validator.
        """
        # === Wallet ===
        # Connects wallet to network. 
        self.wallet.create()
        self.wallet.reregister( subtensor = self.subtensor, netuid=self.config.netuid )

        # === UID ===
        # Get our uid from the chain. 
        # At this point we should have a uid because we are already registered.
        # TODO
        self.uid = 1# self.wallet.get_uid( subtensor = self.subtensor, netuid=self.config.netuid )    

    def save(self, path: str = None):
        r""" Save validated hotkeys and neuron_stats to filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path

            state_dict = {
                'neuron_stats': self.neuron_stats,
                'neuron_hotkeys': self.neuron_hotkeys
            }

            if self.config.neuron.track_hotkey_changes:
                state_dict['neuron_changes'] = self.neuron_changes

            torch.save(state_dict, f'{path}/model.torch')
            bittensor.logging.success(prefix='Saved model', sufix=f'<blue>{path}/model.torch</blue>')

        except Exception as e:
            logger.warning(f'Failed to save model with error: {e}')

    def load(self, path: str = None):
        r""" Load validated hotkeys and neuron_stats from filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path
            state_dict = torch.load(f'{path}/model.torch')

            self.neuron_stats = state_dict['neuron_stats']
            self.neuron_hotkeys = state_dict['neuron_hotkeys']

            if 'neuron_changes' in state_dict and self.config.neuron.track_hotkey_changes:
                self.neuron_changes = state_dict['neuron_changes']

            bittensor.logging.success(prefix='Reloaded model', sufix=f'<blue>{path}/model.torch</blue>')

        except Exception as e:
            logger.warning(f'Failed to load model with error: {e}')


    def init_dendrites(self):
        r""" Get dendrite per uid.
        """
        def endpoint_obj(i):
            return bittensor.endpoint(
                version = bittensor.__version_as_int__,
                uid = i,
                ip = '127.0.0.1',
                ip_type = 4,
                port = 5600 + i,
                hotkey = self.wallet.hotkey.ss58_address,
                coldkey = self.wallet.coldkeypub.ss58_address,
                modality = 0
            )    
        
        endpoints = [ endpoint_obj(i) for i in range(10)]
        
        self.dendrites = {ep.uid: bittensor.text_last_hidden_state( endpoint = ep, wallet = self.wallet ) for ep in endpoints} # TODO
        # self.dendrites = {ep.uid: bittensor.text_last_hidden_state( endpoint = ep, wallet = self.wallet ) for ep in self.metagraph.endpoint_objs}
        self.dendrites_order = list(self.dendrites.keys())
        random.shuffle(self.dendrites_order)

    def metagraph_sync(self):
        r""" Syncing metagraph together with other metagraph-size related objects
        """
        old_hotkeys = self.neuron_hotkeys + [] if self.neuron_hotkeys else self.metagraph.hotkeys
        self.metagraph.sync( subtensor=self.subtensor, netuid=self.config.netuid)
        self.neuron_hotkeys = self.metagraph.hotkeys

        changed_hotkeys = []
        # === Reset neuron stats if uid got replaced
        for uid, old_hotkey in enumerate(old_hotkeys):
            if old_hotkey != self.neuron_hotkeys[uid]:
                if self.config.neuron.track_hotkey_changes:
                    block = self.subtensor.block
                    self.neuron_changes.setdefault(uid, {})  # [uid] -> dict() of blocks
                    self.neuron_changes[uid][block] = {'new_hotkey': self.neuron_hotkeys[uid], 'old_hotkey': old_hotkey}
                    if uid in self.neuron_stats:
                        self.neuron_changes[uid][block]['old_stats'] = self.neuron_stats[uid]

                if uid in self.neuron_stats:
                    del self.neuron_stats[uid]
                    del self.dendrites[uid]
                    self.dendrites[uid] = bittensor.text_last_hidden_state( endpoint = self.metagraph.endpoint_objs, wallet = self.wallet )
                    changed_hotkeys += [uid]
        
        if len(changed_hotkeys):
            logger.info(f"Hotkeys changed: {changed_hotkeys}")
            self.save()  # save neuron_stats, neuron_hotkeys, and neuron_changes to filesystem

    def run(self):
        r""" Keep running running the epochs. 
        """
        self.epoch = 0
        self.global_step = 0 
        # self.metagraph_sync() # TODO
        with self:
            while True:
                self.run_epoch()

    def run_epoch(self):
        r""" Run validation steps and do weight setting and logging at the end of epoch.  
        """
        self.init_dendrites()
        
        epoch_status, epoch_params = self.init_epoch()
        
        while (self.subtensor.block < epoch_params.start_block + epoch_params.blocks_per_epoch or
               time.time() - epoch_status.start_time < epoch_params.blocks_per_epoch * bittensor.__blocktime__):

            logger.info(f'Run epoch {self.epoch} (step {epoch_status.step}) while '
                        f'({self.subtensor.block} < {epoch_params.start_block + epoch_params.blocks_per_epoch} '
                        f'= {epoch_params.start_block} + {epoch_params.blocks_per_epoch}) or '
                        f'({time.time() - epoch_status.start_time:.2f} < {epoch_params.blocks_per_epoch * bittensor.__blocktime__})')

            self.step(epoch_status, epoch_params)
            self.global_step += 1

        # self.metagraph_sync()  # Reset metagraph. # TODO

        # === Set neuron weights to chain ===
        sample_uids, sample_weights = self.calculate_weights(epoch_params)
        self.subtensor.set_weights(
            uids=sample_uids.detach().to('cpu'),
            weights=sample_weights.detach().to('cpu'),
            netuid = self.config.netuid,
            wallet=self.wallet,
            version_key=1,
            wait_for_finalization=self.config.neuron.wait_for_finalization,
        )

        # === End of epoch status logging. ===        
        # self.vlogger.epoch_log(
        #     # metagraph = self.metagraph, # TODO 
        #     netuid = self.config.netuid, 
        #     subtensor = self.subtensor, 
        #     neuron_stats = self.neuron_stats, 
        #     epoch_status = epoch_status,  
        #     debug = self.config.logging.debug or self.config.logging.trace, 
        #     sample_uids = sample_uids,
        #     sample_weights = sample_weights,
        # )
        
        # === Save status. ===
        if epoch_status.step % 25 == 1:
            self.save() 
        
        # === Iterate epochs. ===
        self.epoch += 1
        
    def step(self, epoch_status: Dict[str, Any] , epoch_params: Dict[str, Any] ):
        r""" Run a nucleus forward step, with step log at the end.
        """
        start_time = time.time()


        
        # === Init step statuss. === 
        stats = {}
        step_status = SimpleNamespace(
            responsive_uids = set(),
            queried_uids = set(),
            current_block = self.subtensor.block,
            step_time = None,
            forward_time = None,
            base_loss_time = None,
            shapley_time = None,
            syn_loss_diff = None
        )

        dendrite_flag = epoch_status.step * epoch_params.topk % len(self.dendrites)
        if dendrite_flag + epoch_params.topk > len(self.dendrites):
            random.shuffle(self.dendrites_order)

        # === Forward ===
        # Forwards inputs through the network and returns the loss
        # and endpoint scores using shapely approximation of salience.
        
        stats, step_status = self.nucleus( 
            stats = stats,
            step_status = step_status,
            text_input = next(self.dataset), 
            dendrites = [self.dendrites[uid] for uid in self.dendrites_order[ dendrite_flag: dendrite_flag + epoch_params.topk]],
            validation_len = epoch_params.validation_len
        )

        # === Get scoring. ===
        stats, step_status = self.shapley_synergy(stats, step_status, epoch_params, '_nxt')
        
        # === Stats update ===
        # Updates moving averages and history.
        step_status.responsive_uids, step_status.queried_uids = self.neuron_stats_update(stats, epoch_params)
        epoch_status.responsive_uids |= set(step_status.responsive_uids)
        epoch_status.queried_uids |= set(step_status.queried_uids)
        epoch_status.step += 1
        
        step_status.current_block = self.subtensor.block,
        step_status.step_time = time.time() - start_time
        
        # === End of step logging. ===
        # self.vlogger.step_log( 
        #     uid = self.uid, 
        #     wallet = self.wallet, 
        #     metagraph = self.metagraph, 
        #     netuid = self.config.netuid, 
        #     subtensor = self.subtensor, 
        #     stats = stats,
        #     neuron_stats = self.neuron_stats, 
        #     step_status = step_status,  
        #     epoch_status = epoch_status,  
        #     epoch_params = epoch_params,  
        #     debug = self.config.logging.debug or self.config.logging.trace, 
        #     synapse_keys = self.synapse_keys
        # )

    def init_epoch(self) -> Tuple[Dict, Dict]:
        r""" Init epoch params and reset status. 
        - Update epoch params according to subtensor.
        - Reset epoch status.
        """
        epoch_params = SimpleNamespace(
            batch_size = self.subtensor.validator_batch_size(netuid=self.config.netuid),
            sequence_length = self.subtensor.validator_sequence_length(netuid=self.config.netuid),
            prune_len = self.subtensor.validator_prune_len(netuid=self.config.netuid),
            logits_divergence = self.subtensor.validator_logits_divergence(netuid=self.config.netuid),
            min_allowed_weights = self.subtensor.min_allowed_weights(netuid=self.config.netuid),
            max_weight_limit = self.subtensor.max_weight_limit(netuid=self.config.netuid),
            scaling_law_power = self.subtensor.scaling_law_power(netuid=self.config.netuid),
            synergy_scaling_law_power = self.subtensor.synergy_scaling_law_power(netuid=self.config.netuid),
            current_block = self.subtensor.block,
            epochs_until_reset = self.subtensor.validator_epochs_per_reset(self.config.netuid) if self.config.neuron.epochs_until_reset == -1 else self.config.neuron.epochs_until_reset,
            blocks_per_epoch = self.subtensor.validator_epoch_length(self.config.netuid),
            start_block = self.subtensor.block,
            topk = self.config.neuron.topk,
            validation_len = self.config.neuron.validation_len,
            exclude_quantile = self.config.neuron.exclude_quantile 
        )

        epoch_status = SimpleNamespace(
            step = 0,
            start_time = time.time(),
            responsive_uids = set(),
            queried_uids = set(),
        )

        # === Update dataset size ===
        if (epoch_params.batch_size != self.dataset.batch_size) or (epoch_params.sequence_length + epoch_params.validation_len + epoch_params.prune_len != self.dataset.block_size):
            self.dataset.set_data_size(epoch_params.batch_size, epoch_params.sequence_length + epoch_params.validation_len + epoch_params.prune_len)

        return epoch_status, epoch_params

    def scaling_law_loss_to_params(self, loss: torch.tensor) -> torch.tensor:
        r""" (OpenAI scaling laws) Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv:2001.08361 (2020)
        """
        num_params = torch.exp(torch.log(torch.tensor(8.8e13)) -
                            torch.log(torch.clamp(loss, 1.69)) / 0.076)  # loss lower bound 1.69 is entropy of natural text
        return num_params
    
    def shapley_synergy(self, stats: Dict[str, Any], step_status: Dict[str, Any], epoch_params: Dict[str, Any], ext: str) -> Tuple[Dict, Dict]:
        r"""
        Calculates Shapley synergy for coalition size 2, measured performance above expected performance.
        Measured in effective number of model parameters, just like base Shapley values.
            Args:
                stats (:obj:`Dict`, `required`):
                    Statistics per endpoint for this batch.
                synergy (:obj:`Callable`, `required`)
                    Function to calculate measured loss.
                ext (:obj:`str`, `optional`):
                    Extension to parameter string for stats key.
                target (:obj:`torch.Tensor`, `optional`):
                    Target to measure loss against.
                scaling_law_power (:obj:`float`, `optional`):
                    Power for modified scaling law, powered down to improve dynamic range, e.g. 3 → 6 nats for 0.5.

            Returns:
                syn_loss_diff (:obj:`Dict`, `required`):
                    Dictionary table of pairwise synergies as loss reductions, with direct loss on diagonal.
        """
        def _synergy(first, second):
            # average first + second probabilities per batch item, convert to loss
            measured_loss = -torch.log((torch.exp(-first['losses_nxt']) +
                                        torch.exp(-second['losses_nxt'])) / 2 + 1e-40).mean()
            return measured_loss
        # === Shapley synergy approximation ===
        # Shapley values - second level - coalition size 2
        # Synergy = measured performance above expected performance
        # Measured in effective number of model parameters, just like base Shapley values.
        syn_loss_diff = {}  # expected_loss - measured_loss (where > 0)
        responsives = [uid for uid, stat in stats.items() if 'loss' + ext in stat]
        start_time = time.time()
        for _first, first in stats.items():
                
            if 'loss' + ext not in first:
                continue

            if 'synergy_loss_diff' + ext not in first:
                first['synergy_loss_diff' + ext] = 0

            if 'synergy' + ext not in first:
                first['synergy' + ext] = 0
                
            first_diff = syn_loss_diff.setdefault(_first, {})
            first_diff[_first] = first['loss' + ext]  # diagonal keeps direct loss

            for _second, second in stats.items():
                if 'loss' + ext not in second or _second <= _first:
                    continue
                
                if 'synergy_loss_diff'+ext not in second:
                    second['synergy_loss_diff' + ext] = 0
                
                if 'synergy' + ext not in second:
                    second['synergy' + ext] = 0
                
                second_diff = syn_loss_diff.setdefault(_second, {})

                with torch.no_grad():
                    expected_loss = torch.min(first['loss' + ext], second['loss' + ext])  # expecting min loss

                    measured_loss = _synergy(first, second)

                    loss_diff_share = torch.clamp(expected_loss - measured_loss, 0) / 2  # record direct loss diff
                    loss_diff_share /= len(responsives)  # average over responsives
                    first['synergy_loss_diff' + ext] += loss_diff_share
                    second['synergy_loss_diff' + ext] += loss_diff_share

                    # pairwise loss reduction of expected to measured loss due to synergy between first and second
                    first_diff[_second] = loss_diff_share
                    second_diff[_first] = loss_diff_share

                    measured_params = self.scaling_law_loss_to_params(measured_loss)
                    expected_params = self.scaling_law_loss_to_params(expected_loss)

                    # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
                    pow_measured_params = torch.pow(measured_params, epoch_params.scaling_law_power)
                    pow_expected_params = torch.pow(expected_params, epoch_params.scaling_law_power)

                    synergy_share = torch.clamp(pow_measured_params - pow_expected_params, 0) / 2
                    synergy_share /= len(responsives)  # average over responsives
                    first['synergy' + ext] += synergy_share  # share synergy amongst coalition members
                    second['synergy' + ext] += synergy_share
        
        step_status.shapley_time = time.time() - start_time
        step_status.syn_loss_diff = syn_loss_diff
        return stats, step_status
    
    def neuron_stats_update(self, neuron_stats: Dict[int, Dict[str, Any]], epoch_params: Dict[str, Any]) -> Tuple[List, List]:
        r""" Updates self.neuron_stats with new individual dictionaries per uid.
        """
        responsive_uids = []
        print(neuron_stats)
        for _uid, _stats in neuron_stats.items():
            stats = self.neuron_stats.setdefault(_uid, {})

            # === EMA normal update ===
            # If synapse responsive push available values into EMA for normal update.
            # Normal EMA values provide a view on neuron performance if fully responsive.
            for key in _stats:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                if math.isnan(_stats[key]):
                    continue
                if key in stats:
                    stats[key] = (1 - self.alpha) * stats[key] + self.alpha * _stats[key]  # update EMA
                else:
                    stats.setdefault(key, _stats[key])

            # === Extra stats computation ===
            # Compute values on EMA stats, such as the scaling law on EMA loss.
            # Required for values that need to be computed on longer-term stats.
            extra_stats = {}
            if 'loss_nxt' in _stats and 'loss_nxt' in stats:  # elif neuron not responsive then omit
                # estimate the effective number of model parameters from EMA loss
                _num_params = self.scaling_law_loss_to_params(torch.tensor(stats['loss_nxt']))

                # powered down number of params, e.g. dynamic range 3 → 6 nats for scaling_law_power=0.5
                _pow_num_params = torch.pow(_num_params, epoch_params.scaling_law_power)

                extra_stats.update({'est_params_nxt': _num_params.item(), 'base_params_nxt': _pow_num_params.item()})

                if 'synergy_nxt' in stats:
                    extra_stats['shapley_values_nxt'] = extra_stats['base_params_nxt'] + stats['synergy_nxt']

                if 'logits_excess_nxt' in stats:
                    # penalize by logits divergence excess
                    extra_stats['shapley_values_nxt'] /= 1 + epoch_params.logits_divergence * stats['logits_excess_nxt']

            # === EMA zeroing update ===
            # Push zero into EMA for synapse_keys to exponentially decay weighting keys if neuron non-responsive
            if 'updates!' in stats:
                stats['updates!'] += 1  # increment number of EMA zeroing updates
            else:
                stats.setdefault('updates!', 1)  # number of EMA zeroing updates init to zero

            for key in self.synapse_keys:
                zkey = key + '!'  # zeroing key
                stats.setdefault(zkey, 0.)  # initialize zkey val to zero to gradually increase with observations
                if key in _stats and not math.isnan(_stats[key]):
                    responsive_uids += [_uid]
                    stats[zkey] = (1 - self.alpha) * stats[zkey] + self.alpha * _stats[key]
                elif key in extra_stats and not math.isnan(extra_stats[key]):
                    responsive_uids += [_uid]
                    stats[zkey] = (1 - self.alpha) * stats[zkey] + self.alpha * extra_stats[key]
                else:
                    stats[zkey] = (1 - self.alpha) * stats[zkey]  # + self.alpha * 0

            # === EMA normal update ===
            # If synapse responsive push available values into EMA for normal update.
            # Normal EMA values provide a view on neuron performance if fully responsive.
            for key in self.synapse_keys:
                if key in _stats or key in extra_stats:
                    updates = 'updates_' + key
                    if updates in stats:
                        stats[updates] += 1  # increment number of normal EMA updates made
                    else:
                        stats.setdefault(updates, 1)  # add updates fields for new uid entries

            for key in extra_stats:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                if math.isnan(extra_stats[key]):
                    continue
                if key in stats:
                    stats[key] = (1 - self.alpha) * stats[key] + self.alpha * extra_stats[key]  # update EMA
                else:
                    stats.setdefault(key, extra_stats[key])

        return responsive_uids, list(neuron_stats.keys())  # responsive_uids, queried_uids
    
    def calculate_weights(self, epoch_params: Dict[str, Any]) -> Tuple[torch.tensor, torch.tensor]:
        r""" Calculates neuron set-weights from weight_key mapped values. Defines weight_key as the neuron stats key
        used to obtain the mapped stat value (typically a Shapley value) that the final set-weights are calculated from.
        """

        weight_key = self.weight_key + '!'  # use zeroing key to penalize non-responsive neurons

        min_allowed_weights = self.subtensor.min_allowed_weights(netuid=self.config.netuid) 
        max_weight_limit = self.subtensor.max_weight_limit(netuid=self.config.netuid)


        # === Populate neuron weights ===
        neuron_weights = torch.zeros_like(self.metagraph.total_stake)  # allow unevaluated UIDs for min_allowed_weights
        for uid in self.neuron_stats:
            if weight_key in self.neuron_stats[uid]:
                neuron_weights[uid] = torch.tensor([self.neuron_stats[uid][weight_key]])

        # === Filter to non-zero weights ===
        sample_uids = torch.argwhere(neuron_weights > 0).squeeze(dim=1)  # find uids with non-zero weight
        sample_weights = neuron_weights[sample_uids]  # filter to non-zero weights

        # === If no uids responds, return ===
        if len(sample_uids) == 0:
            return sample_uids, sample_weights

        # === Exclude lowest quantile from weight setting ===
        max_exclude = (len(sample_weights) - min_allowed_weights) / len(sample_weights)  # max excludable weight quantile
        
        quantile = self.subtensor.validator_exclude_quantile(netuid=self.config.netuid) if epoch_params.exclude_quantile == -1 else epoch_params.exclude_quantile 
        if 0 < max_exclude:
            exclude_quantile = min([quantile , max_exclude])  # reduce quantile to meet min_allowed_weights
            lowest_quantile = sample_weights.quantile(exclude_quantile)  # find lowest quantile threshold
            sample_uids = sample_uids[lowest_quantile <= sample_weights]  # exclude uids with weights below quantile
            sample_weights = sample_weights[lowest_quantile <= sample_weights]  # exclude weights below quantile

            logger.info(f'Exclude {exclude_quantile} quantile ({lowest_quantile}) | '
                        f'{len(sample_weights)} Shapley values | min:{sample_weights.min()} max:{sample_weights.max()}')

        # === Normalize and apply max_weight_limit ===
        sample_weights = bittensor.utils.weight_utils.normalize_max_weight(x=sample_weights,
                                                                             limit=max_weight_limit)
        logger.info(f'{len(sample_weights)} normalize_max_weight | '
                    f'max:{sample_weights.max()}')

        return sample_uids, sample_weights
    
if __name__ == "__main__":
    bittensor.utils.version_checking()
    neuron().run()
