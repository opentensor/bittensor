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
import datetime
from threading import Lock

# Torch tooling.
from torch.nn.utils.rnn import pad_sequence

# Local model.
from model import PretrainedModel

# Check run config.
def check_config( config: 'bittensor.Config' ):
    bittensor.logging.check_config( config )
    bittensor.wallet.check_config( config )
    bittensor.subtensor.check_config( config )
    bittensor.metagraph.check_config( config )
    bittensor.axon.check_config( config )
    full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name), config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ))
    config.neuron.full_path = os.path.expanduser( full_path )
    if not os.path.exists( config.neuron.full_path ):
        os.makedirs( config.neuron.full_path )

# Create run config.
def get_config ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='If set, defaults are overridden by passed file.')
    
    # Miner arguements
    parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_server')
    parser.add_argument('--neuron.restart', action='store_true', help='If True, train the neuron from the beginning', default=False)
    parser.add_argument('--neuron.no_set_weights', action='store_true', help='If True, the model does not set weights.', default=False)
    parser.add_argument('--neuron.max_batch_size', type=int, help='The maximum batch size for forward requests.', default=-1)
    parser.add_argument('--neuron.max_sequence_len', type=int, help='The maximum sequence length for forward requests.', default=-1)
    parser.add_argument('--neuron.blacklist.hotkeys', type=str, required=False, nargs='*', action='store', help='To blacklist certain hotkeys', default=[])

    # Synapse Arguements
    parser.add_argument('--neuron.no_lasthidden', action='store_false', help='To turn off last hidden synapse', default=True)
    parser.add_argument('--neuron.no_seq2seq', action='store_false', help='To turn off seq2seq synapse', default=True)

    # Netuid Arg
    parser.add_argument('--netuid', type=int , help='Subnet netuid', default=1)

    bittensor.wallet.add_args( parser )
    bittensor.axon.add_args( parser )
    bittensor.subtensor.add_args( parser )
    bittensor.logging.add_args( parser )
    bittensor.metagraph.add_args( parser )
    bittensor.TextSeq2SeqSynapse.add_args( parser )
    bittensor.TextLastHiddenStateSynapse.add_args( parser )
    PretrainedModel.add_args( parser )
    return bittensor.config( parser )

# Main entry point for model serving.
def main( ):

    # --- Build, Check, Set and Print the run config.
    config = get_config()
    config.to_defaults()
    check_config( config )
    print (config)

    # --- Turn on logging.
    bittensor.logging ( config = config, logging_dir = config.neuron.full_path )

    # --- Create our chain connection.
    subtensor = bittensor.subtensor( config )

    # --- Create our wallet and register it to the subnetwork.
    wallet = bittensor.wallet( config )
    wallet.register( netuid = config.netuid, subtensor = subtensor )

    # --- Create our network state cache
    metagraph = bittensor.metagraph ( config = config, netuid = config.netuid, )
    metagraph.sync( netuid = config.netuid, subtensor = subtensor ).save()

    # --- Build /Load our model and set the device.
    call_mutex = Lock()
    model = PretrainedModel( config = config ).to( config.neuron.device )
    if not config.neuron.restart:
        model.load( config.neuron.full_path )

    # --- Build axon server and start it.
    axon = bittensor.axon( config = config, wallet = wallet )
    axon.start()

    # --- Build our TextSeq2Seq synapse.
    if config.neuron.seq2seq:
        class TS2SSynapse( bittensor.TextSeq2SeqSynapse ):
            def priority(self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> float:
                return 0.0
            
            def blacklist(self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> torch.FloatTensor:
                return False
            
            def forward(self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> 'bittensor.TextSeq2SeqBittensorCall':
                with call_mutex:
                    tokens = model.token_remap( forward_call.text_prompt.to( model.device ) )
                    output = model.pre_model.generate(
                        input_ids = tokens['input_ids'],
                        attention_mask = tokens['attention_mask'],
                        max_length = max(tokens['input_ids'].shape[1] + 1, forward_call.num_to_generate),
                        num_beams = forward_call.num_beams,
                        no_repeat_ngram_size = forward_call.no_repeat_ngram_size,
                        early_stopping = forward_call.early_stopping,
                        do_sample = forward_call.do_sample,
                        top_p = forward_call.top_p,
                        num_return_sequences = forward_call.num_return_sequences,
                        temperature = forward_call.temperature,
                        repetition_penalty = forward_call.repetition_penalty,
                        length_penalty = forward_call.length_penalty,
                        max_time = forward_call.max_time,
                        num_beam_groups = forward_call.num_beam_groups,
                    )
                    raw_texts = [ model.tokenizer.decode(out) for out in output ]
                    tokens = [ model.std_tokenizer.encode(raw_text, return_tensors="pt")[:,:forward_call.num_to_generate].view(-1) for raw_text in raw_texts ]
                    forward_call.generations = pad_sequence(tokens, batch_first=True)
                    return forward_call
                
    # Attach the synapse to the axon.
    synapse_s2s = TS2SSynapse( config = config, metagraph = metagraph )
    axon.attach( synapse = synapse_s2s )

        
    # --- Build our TextLastHiddenState synapse.
    if config.neuron.lasthidden_stake:
        class TLHSSynapse( bittensor.TextLastHiddenStateSynapse ):
            def priority(self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> float:
                return 0.0
            
            def blacklist(self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> torch.FloatTensor:
                return False
            
            def forward(self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> bittensor.TextLastHiddenStateBittensorCall:
                with call_mutex:
                    _, _, hidden = model.encode_forward( forward_call.text_inputs.to( model.device ) )
                    forward_call.hidden_states = hidden
                    return forward_call
        # Attach the synapse to the axon.
        synapse_tlhs = TLHSSynapse( config = config, metagraph = metagraph )
        axon.attach( synapse = synapse_tlhs )

    # --- Run Forever.
    block_per_epoch = config.neuron.blocks_per_epoch
    last_update = subtensor.get_current_block()
    uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    while True:        

        # --- Wait until next epoch.
        current_block = subtensor.get_current_block()
        while current_block - last_update >= block_per_epoch:
            time.sleep( bittensor.__blocktime__ )
            current_block = subtensor.get_current_block()

        # --- Set last update.
        last_update = subtensor.get_current_block()

        # --- Update the metagraph with the latest network state.
        metagraph.sync( netuid = config.netuid, subtensor = subtensor )
        uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )

        # --- Log performance.
        print(
            f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
            f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
            f'[dim white not bold] [green]{str( metagraph.stake[ uid ].item() ):.4}[/green] Stake [/dim white not bold]'
            f'[dim white not bold]| [yellow]{ str( metagraph.trust[ uid ].item() ) :.3}[/yellow] Trust [/dim white not bold]'
            f'[dim white not bold]| [green]{ str( metagraph.incentive[ uid ].item() ):.3}[/green] Incentive [/dim white not bold]')

        # --- Set weights.
        if not config.neuron.no_set_weights:
            try: 
                # --- query the chain for the most current number of peers on the network
                chain_weights = torch.zeros( subtensor.subnetwork_n( netuid = config.netuid) )
                chain_weights [ uid ] = 1 
                did_set = subtensor.set_weights(
                    uids = torch.arange( 0, len( chain_weights ) ),
                    netuid = config.netuid,
                    weights = chain_weights,
                    wait_for_inclusion = False,
                    wallet = wallet,
                    version_key = 1 
                )
            except:
                pass



if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( )