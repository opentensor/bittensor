
import os
import time
import copy
import torch
import argparse
import bittensor
from abc import ABC, abstractmethod
from rich import print
from datetime import datetime

class BasePromptingMiner(ABC):

    @abstractmethod
    def synapse(self) -> "bittensor.TextPromptingSynapse": 
        ...

    def __init__(
        self,
        config: "bittensor.Config" = None
    ):
        config = config if config != None else BasePromptingMiner.config()
        self.config = copy.deepcopy( config )
        self.check_config( self.config )
        self.config.to_defaults()
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.subtensor = bittensor.subtensor( self.config )
        self.wallet = bittensor.wallet( self.config )
        self.wallet.reregister(netuid = config.netuid, subtensor = self.subtensor )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.axon = bittensor.axon( 
            wallet = self.wallet,
            metagraph = self.metagraph,
            config = self.config,
        )
        self.axon.attach( self.synapse() )
        self.axon.start()
        self.axon.netuid = config.netuid
        self.axon.protocol = 4
        self.subtensor.serve_axon( self.axon )

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        return bittensor.config( parser )

    @classmethod
    def help( cls ):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print( cls.__new__.__doc__ )
        parser.print_help()

    @classmethod
    def check_config( cls, config: "bittensor.Config" ):
        bittensor.axon.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        full_path = os.path.expanduser(
            '{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name),
                                  config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ) )
        config.neuron.full_path = os.path.expanduser( full_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser ):
        print ('get super level config')
        parser.add_argument(
            '--netuid', 
            type = int, 
            help = 'Subnet netuid', 
            default = 41
        )
        parser.add_argument(
            '--neuron.name', 
            type = str,
            help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
            default = 'openai_prompting_miner'
        )
        parser.add_argument(
            '--neuron.blocks_per_epoch', 
            type = str, 
            help = 'Blocks until the miner sets weights on chain',
            default = 100
        )
        parser.add_argument(
            '--neuron.no_set_weights', 
            action = 'store_true', 
            help = 'If True, the model does not set weights.',
            default = False
        )
        parser.add_argument(
            '--neuron.max_batch_size', 
            type = int, 
            help = 'The maximum batch size for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.max_sequence_len', 
            type = int, 
            help = 'The maximum sequence length for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.blacklist.hotkeys', 
            type = str, 
            required = False, 
            nargs = '*', 
            action = 'store',
            help = 'To blacklist certain hotkeys', default=[]
        )
        bittensor.wallet.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.metagraph.add_args( parser )

    def run( self ):
        # --- Run Forever.
        last_update = self.subtensor.get_current_block()
        while True:

            # --- Wait until next epoch.
            current_block = self.subtensor.get_current_block()
            while (current_block - last_update) < self.config.neuron.blocks_per_epoch:
                time.sleep(bittensor.__blocktime__)
                current_block = self.subtensor.get_current_block()
            last_update = self.axon.get_current_block()

            # --- Update the metagraph with the latest network state.
            self.metagraph.sync(netuid=self.config.netuid, subtensor=self.subtensor)
            uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

            # --- Log performance.
            print(
                f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
                f'[dim white not bold] [green]{str(self.metagraph.S[uid].item()):.4}[/green] Stake [/dim white not bold]'
                f'[dim white not bold]| [yellow]{str(self.metagraph.trust[uid].item()) :.3}[/yellow] Trust [/dim white not bold]'
                f'[dim white not bold]| [green]{str(self.metagraph.incentive[uid].item()):.3}[/green] Incentive [/dim white not bold]')

            # --- Set weights.
            if not self.config.neuron.no_set_weights:
                try:
                    # --- query the chain for the most current number of peers on the network
                    chain_weights = torch.zeros(self.subtensor.subnetwork_n(netuid=self.config.netuid))
                    chain_weights[uid] = 1
                    did_set = self.subtensor.set_weights(
                        uids=torch.arange(0, len(chain_weights)),
                        netuid=self.config.netuid,
                        weights=chain_weights,
                        wait_for_inclusion=False,
                        wallet=self.wallet,
                        version_key=1
                    )
                except:
                    pass