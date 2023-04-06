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

# General.
import os
import json
import time
import torch
import argparse
import bittensor

from typing import List
from rich import print
from datetime import datetime

# Torch tooling.
from langchain.llms import AlephAlpha


# Check run config.
def check_config(config: 'bittensor.Config'):
    bittensor.logging.check_config(config)
    bittensor.wallet.check_config(config)
    bittensor.subtensor.check_config(config)
    bittensor.metagraph.check_config(config)
    bittensor.axon.check_config(config)
    full_path = os.path.expanduser(
        '{}/{}/{}/{}'.format(config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name),
                             config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name))
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path)



# Create run config.
def get_config():
    parser = argparse.ArgumentParser()


    # AlephAlpha arguements
    parser.add_argument('--client.api_key', type=str, help='AlephAlpha API key.', required=True)
    parser.add_argument('--client.model', type=str, help='Model name to use.', default='luminous-base')
    parser.add_argument('--client.maximum_tokens', type=int, help='The maximum number of tokens to be generated.', default=64)
    parser.add_argument('--client.temperature', type=float, help='A non-negative float that tunes the degree of randomness in generation.', default=0.0)
    parser.add_argument('--client.stop_sequences', type=list[str], help='Stop tokens.', default=['user: ', 'bot: ', 'system: '])
    parser.add_argument('--client.top_k', type=int, help='Number of most likely tokens to consider at each step.', default=0)
    parser.add_argument('--client.top_p', type=float, help='Total probability mass of tokens to consider at each step.', default=0.0)

    # Miner arguements
    parser.add_argument('--netuid', type=int, help='Subnet netuid', default=21)
    parser.add_argument('--neuron.name', type=str,
                        help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
                        default='alephalpha_prompting_miner')
    parser.add_argument('--neuron.blocks_per_epoch', type=str, help='Blocks until the miner sets weights on chain',
                        default=100)
    parser.add_argument('--neuron.no_set_weights', action='store_true', help='If True, the model does not set weights.',
                        default=False)
    parser.add_argument('--neuron.max_batch_size', type=int, help='The maximum batch size for forward requests.',
                        default=-1)
    parser.add_argument('--neuron.max_sequence_len', type=int, help='The maximum sequence length for forward requests.',
                        default=-1)
    parser.add_argument('--neuron.blacklist.hotkeys', type=str, required=False, nargs='*', action='store',
                        help='To blacklist certain hotkeys', default=[])


    bittensor.wallet.add_args(parser)
    bittensor.axon.add_args(parser)
    bittensor.subtensor.add_args(parser)
    bittensor.logging.add_args(parser)
    bittensor.metagraph.add_args(parser)
    return bittensor.config(parser)


# Main entry point for model serving.
def main():
    # --- Build, Check, Set and Print the run config.
    config = get_config()
    config.to_defaults()
    check_config(config)
    print(config)

    # --- Turn on logging.
    bittensor.logging(config=config, logging_dir=config.neuron.full_path)
    
    # --- Create our chain connection.
    subtensor = bittensor.subtensor(config)
    
    # --- Create our wallet and register it to the subnetwork.
    wallet = bittensor.wallet(config)
    wallet.register(netuid=config.netuid, subtensor=subtensor)
    
    # --- Create our network state cache
    metagraph = bittensor.metagraph(config=config, netuid=config.netuid, )
    metagraph.sync(netuid=config.netuid, subtensor=subtensor).save()
    uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    
    # --- Build /Load our model and set the device.
    with bittensor.__console__.status(f"Loading model {config.neuron.model_name} AI21 ..."):
        bittensor.logging.info('Loading', config.neuron.model_name )
        model = AlephAlpha(**config.client)

    # --- Build axon server and start it.tensor.loggi
    axon = bittensor.axon(
        wallet=wallet,
        metagraph=metagraph,
        config=config,
    )

    def _process_history(history: List[dict]) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += 'system: ' + message['content'] + '\n'

            if message['role'] == 'assistant':
                processed_history += 'assistant: ' + message['content'] + '\n'
            
            if message['role'] == 'user':
                processed_history += 'user: ' + message['content'] + '\n'
        return processed_history

    class Synapse(bittensor.TextPromptingSynapse):
        def _priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
            return 0.0

        def _blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> bool:
            return False

        def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str:
            pass

        def forward(self, messages: List[str]) -> str:
            bittensor.logging.info('messages', str(messages))
            history = _process_history(messages)
            bittensor.logging.info('history', str(history))
            resp = model(history)
            bittensor.logging.info('response', str(resp))
            return resp


    # with bittensor.__console__.status("Serving Axon on netuid:{} subtensor:{} ...".format( config.netuid, subtensor )):
    syn = Synapse()
    axon.attach(syn)
    axon.start()
    axon.netuid = config.netuid
    axon.protocol = 4
    subtensor.serve_axon( axon )  
    print (axon)

    # --- Run Forever.
    last_update = subtensor.get_current_block()
    while True:

        # --- Wait until next epoch.
        current_block = subtensor.get_current_block()
        while (current_block - last_update) < config.neuron.blocks_per_epoch:
            time.sleep(bittensor.__blocktime__)
            current_block = subtensor.get_current_block()
        last_update = subtensor.get_current_block()

        # --- Update the metagraph with the latest network state.
        metagraph.sync(netuid=config.netuid, subtensor=subtensor)
        uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)

        # --- Log performance.
        print(
            f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
            f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
            f'[dim white not bold] [green]{str(metagraph.S[uid].item()):.4}[/green] Stake [/dim white not bold]'
            f'[dim white not bold]| [yellow]{str(metagraph.trust[uid].item()) :.3}[/yellow] Trust [/dim white not bold]'
            f'[dim white not bold]| [green]{str(metagraph.incentive[uid].item()):.3}[/green] Incentive [/dim white not bold]')

        # --- Set weights.
        if not config.neuron.no_set_weights:
            try:
                # --- query the chain for the most current number of peers on the network
                chain_weights = torch.zeros(subtensor.subnetwork_n(netuid=config.netuid))
                chain_weights[uid] = 1
                did_set = subtensor.set_weights(
                    uids=torch.arange(0, len(chain_weights)),
                    netuid=config.netuid,
                    weights=chain_weights,
                    wait_for_inclusion=False,
                    wallet=wallet,
                    version_key=1
                )
            except:
                pass


if __name__ == "__main__":
    bittensor.utils.version_checking()
    main()