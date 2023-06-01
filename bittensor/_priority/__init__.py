# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import math
import argparse
import bittensor

class priority:

    def __init__( self, config: "bittensor.Config" = None ):
        self.config = config or priority.config()

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser()
        priority.add_args(parser)
        return bittensor.config( parser )

    @classmethod
    def help(cls):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print( cls.__new__.__doc__ )
        parser.print_help()

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None ):
        prefix_str = "" if prefix is None else prefix + "."
        parser.add_argument(
            '--' + prefix_str + 'priority.default_priority',
            type = float,
            help = 'Default call priority in queue.',
            default = 0.0
        )
        parser.add_argument(
            '--' + prefix_str + 'priority.blacklisted_keys', 
            type = str, 
            required = False, 
            nargs = '*', 
            action = 'store',
            help = 'List of ss58 addresses which are always given -math.inf priority', default=[]
        )
        parser.add_argument(
            '--' + prefix_str + 'priority.whitelisted_keys', 
            type = str, 
            required = False, 
            nargs = '*', 
            action = 'store',
            help = 'List of ss58 addresses which are always given math.inf priority', default=[]
        )

    def priority( 
            self, 
            forward_call: "bittensor.SynapseCall",
            metagraph: "bittensor.Metagraph" = None,
        ) -> float:

        # Check for blacklisted keys which take priority over all other checks.
        src_hotkey = forward_call.src_hotkey
        if src_hotkey in self.config.priority.blacklisted_keys:
            return -math.inf
        
        # Check for whitelisted keys which take priority over all remaining checks.
        if src_hotkey in self.config.priority.whitelisted_keys:
            return math.inf 

        # Otherwise priority of requests is based on stake.
        if metagraph is not None:
            uid = metagraph.hotkeys.index( forward_call.src_hotkey )
            return metagraph.S[ uid ].item()
        
        # Default priority.
        return self.config.priority.default_priority
