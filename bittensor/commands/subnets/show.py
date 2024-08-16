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

import argparse
import bittensor as bt
console = bittensor.__console__  # type: ignore

class ShowSubnet:

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        stake_parser = parser.add_parser("show", help="""Show Subnet Stake.""")
        stake_parser.add_argument("--netuid", dest="netuid", type=int, required=False)
        stake_parser.add_argument("--no_prompt", "--y", "-y", dest = 'no_prompt', required=False, action='store_true', help="""Specify this flag to delegate stake""" )
        bt.subtensor.add_args(stake_parser)
        
    @staticmethod
    def check_config(config: "bt.config"): pass
    
    @staticmethod
    def run(cli: "bt.cli"):
        config = cli.config.copy()
        subtensor = bt.subtensor(config=config, log_verbose=False)
        
        
    