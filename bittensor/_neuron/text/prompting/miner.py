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
import time
import torch
import argparse
import bittensor
from rich import print
from datetime import datetime
import openai

# Torch tooling.
from torch.nn.utils.rnn import pad_sequence


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

    parser.add_argument('--api_key', type=str, help='openai api key')

    parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')

    # Miner arguements
    parser.add_argument('--neuron.name', type=str,
                        help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
                        default='core_server')
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

    # Netuid Arg
    parser.add_argument('--netuid', type=int, help='Subnet netuid', default=1)

    bittensor.wallet.add_args(parser)
    bittensor.axon.add_args(parser)
    bittensor.subtensor.add_args(parser)
    bittensor.logging.add_args(parser)
    bittensor.metagraph.add_args(parser)
    # bittensor.TextPromptingSynapse.add_args(parser)
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

    # --- Build /Load our model and set the device.
    openai.api_key = config.api_key

    # --- Build axon server and start it.
    axon = bittensor.axon(
        wallet=wallet,
        metagraph=metagraph,
        config=config,
    )
    axon.start()

    class Synapse(bittensor.TextPromptingSynapse):
        def _priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
            return 0.0

        def _blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> bool:
            return False

        def forward(self, messages: str) -> str:
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"},
                    {"role": "assistant", "content": "The World Series was played in Arlington, Texas."},
                    {"role": "user", "content": "Who was the MVP?"}
                ]
            )
            return resp['choices'][0]['message']['content']

    syn = Synapse()
    axon.attach(syn)

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