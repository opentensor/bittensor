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
import json

from typing import List



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

    # OpenAI  arguments
    parser.add_argument('--api_key', type=str, required=True, help='openai api key')
    parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
    parser.add_argument('--neuron.model_name', type=str, default='gpt-3.5-turbo', help="ID of the model to use.")
    parser.add_argument('--neuron.suffix', type=str, default=None, help="The suffix that comes after a completion of inserted text.")
    parser.add_argument('--neuron.max_tokens', type=int, default=256, help="The maximum number of tokens to generate in the completion.")
    parser.add_argument('--neuron.temperature', type=float, default=0.7, help="Sampling temperature to use, between 0 and 2.")
    parser.add_argument('--neuron.top_p', type=float, default=1, help="Nucleus sampling parameter, top_p probability mass.")
    parser.add_argument('--neuron.n', type=int, default=1, help="How many completions to generate for each prompt.")
    parser.add_argument('--neuron.stream', action='store_true', default=False, help="Whether to stream back partial progress.")
    parser.add_argument('--neuron.logprobs', type=int, default=None, help="Include the log probabilities on the logprobs most likely tokens.")
    parser.add_argument('--neuron.echo', action='store_true', default=False, help="Echo back the prompt in addition to the completion.")
    parser.add_argument('--neuron.stop', type=list[str], help='Up to 4 sequences where the API will stop generating further tokens.', default=['user: ', 'bot: ', 'system: '])
    parser.add_argument('--neuron.presence_penalty', type=float, default=0, help="Penalty for tokens based on their presence in the text so far.")
    parser.add_argument('--neuron.frequency_penalty', type=float, default=0, help="Penalty for tokens based on their frequency in the text so far.")
    parser.add_argument('--neuron.best_of', type=int, default=1, help="Generates best_of completions server-side and returns the 'best' one.")
    parser.add_argument('--neuron.logit_bias', type=json.loads, default=None, help="Modify the likelihood of specified tokens appearing in the completion.")
    parser.add_argument('--neuron.user', type=str, default=None, help="A unique identifier representing your end-user.")

    # Miner arguements
    parser.add_argument('--netuid', type=int, help='Subnet netuid', default=21)
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

    # --- Build /Load our model and set the device.
    openai.api_key = config.api_key

    class Synapse(bittensor.TextPromptingSynapse):
        def _priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
            return 0.0

        def _blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> bool:
            return False

        def forward(self, messages: List[str]) -> str:
            return openai.ChatCompletion.create(
                model=config.neuron.model_name,
                messages=messages,
                temperature=config.neuron.temperature,
                max_tokens=config.neuron.max_tokens,
                top_p=config.neuron.top_p,
                frequency_penalty=config.neuron.frequency_penalty,
                presence_penalty=config.neuron.presence_penalty,
                n=config.neuron.n,
                best_of=config.neuron.best_of,
                openai_api_key=config.neuron.api_key,
                batch_size=config.neuron.batch_size,
                request_timeout=config.neuron.request_timeout,
                logit_bias=config.neuron.logit_bias,
                max_retries=config.neuron.max_retries,
            )['choices'][0]['message']['content']
        

    # --- Build axon server and start it.
    axon = bittensor.axon(
        wallet=wallet,
        metagraph=metagraph,
        config=config,
    )
    syn = Synapse()
    axon.attach(syn)
    axon.start()
    axon.netuid = config.netuid
    axon.protocol = 4
    subtensor.serve_axon( axon )  
    

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