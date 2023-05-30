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
import argparse
import bittensor
from typing import Union, Tuple

class blacklist:

    def __init__( self, config: "bittensor.Config" = None ):
        self.config = config or blacklist.config()

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser()
        blacklist.add_args(parser)
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
            '--' + prefix_str + 'blacklist.blacklisted_keys', 
            type = str, 
            required = False, 
            nargs = '*', 
            action = 'store',
            help = 'List of ss58 addresses which are always disallowed pass through.', default=[]
        )
        parser.add_argument(
            '--' + prefix_str + 'blacklist.whitelisted_keys', 
            type = str, 
            required = False, 
            nargs = '*', 
            action = 'store',
            help = 'List of ss58 addresses which are always allowed pass through.', default=[]
        )
        parser.add_argument(
            '--' + prefix_str + 'blacklist.allow_non_registered',
            action = 'store_true',
            help = 'If True, the miner will allow non-registered hotkeys to mine.',
            default = True
        )
        parser.add_argument(
            '--' + prefix_str + 'blacklist.min_allowed_stake',
            type = float,
            help = 'Minimum stake required to pass blacklist.',
            default = 0.0
        )
        parser.add_argument(
            '--' + prefix_str + 'blacklist.vpermit_required',
            action = 'store_true',
            help = 'If True, the miner will require a vpermit to pass blacklist.',
            default = False
        )

    def blacklist( 
            self, 
            forward_call: "bittensor.SynapseCall",
            metagraph: "bittensor.Metagraph" = None,
        ) -> Union[ Tuple[bool, str], bool ]:

        # Check for blacklisted keys which take priority over all other checks.
        src_hotkey = forward_call.src_hotkey
        if src_hotkey in self.config.blacklist.blacklisted_keys:
            return True, 'blacklisted key'

        # Check for whitelisted keys which take priority over all remaining checks.
        if src_hotkey in self.config.blacklist.whitelisted_keys:
            return False, 'whitelisted key'

        # Check if pubkey is registered.
        is_registered = False
        if metagraph is not None:
            is_registered = src_hotkey in metagraph.hotkeys

        if not is_registered and not self.config.blacklist.allow_non_registered:
            return True, 'pubkey not registered'

        # Check for stake amount.
        if is_registered and self.config.blacklist.min_allowed_stake > 0.0:
            uid = metagraph.hotkeys.index(src_hotkey)
            stake = metagraph.S[uid].item()
            if stake < self.config.blacklist.min_allowed_stake:
                return True, 'pubkey stake below min_allowed_stake'

        # Check for vpermit.
        if metagraph is not None and self.config.blacklist.vpermit_required and is_registered:
            uid = metagraph.hotkeys.index(src_hotkey)
            return metagraph.neurons[uid].validator_permit

        # All checks passed.
        return False, 'passed blacklist'