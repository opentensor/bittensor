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


import bittensor
from .commands import *

class CLI:
    """
    Implementation of the CLI class, which handles the coldkey, hotkey and money transfer 
    """
    def __init__(self, config: 'bittensor.Config' ):
        r""" Initialized a bittensor.CLI object.
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.cli.config()
        """
        # (d)efaults to True if config.no_version_checking is not set.
        if not config.get("no_version_checking", d=True):
            try:
                bittensor.utils.version_checking()
            except:
                raise RuntimeError("To avoid internet based version checking pass --no_version_checking while running the CLI.")
        self.config = config

    def run ( self ):
        """ Execute the command from config 
        """
        if self.config.command == "run":
            RunCommand.run( self )
        elif self.config.command == "transfer":
            TransferCommand.run( self )
        elif self.config.command == "register":
            RegisterCommand.run( self )
        elif self.config.command == "unstake":
            UnStakeCommand.run( self )
        elif self.config.command == "stake":
            StakeCommand.run( self )
        elif self.config.command == "overview":
            OverviewCommand.run( self )
        elif self.config.command == "list":
            ListCommand.run( self )
        elif self.config.command == "new_coldkey":
            NewColdkeyCommand.run( self )
        elif self.config.command == "new_hotkey":
            NewHotkeyCommand.run( self )
        elif self.config.command == "regen_coldkey":
            RegenColdkeyCommand.run( self )
        elif self.config.command == "regen_coldkeypub":
            RegenColdkeypubCommand.run( self )
        elif self.config.command == "regen_hotkey":
            RegenHotkeyCommand.run( self )
        elif self.config.command == "metagraph":
            MetagraphCommand.run( self )
        elif self.config.command == "weights":
            WeightsCommand.run( self )
        elif self.config.command == "inspect":
            InspectCommand.run( self )
        elif self.config.command == "help":
            HelpCommand.run( self )
        elif self.config.command == 'update':
            UpdateCommand.run( self )
        elif self.config.command == 'nominate':
            NominateCommand.run( self )
        elif self.config.command == 'delegate':
            DelegateStakeCommand.run( self )
        elif self.config.command == 'undelegate':
            DelegateUnstakeCommand.run( self )
        elif self.config.command == 'my_delegates':
            MyDelegatesCommand.run( self )
        elif self.config.command == 'list_delegates':
            ListDelegatesCommand.run( self )
        elif self.config.command == 'list_subnets':
            ListSubnetsCommand.run( self )
        elif self.config.command == 'recycle_register':
            RecycleRegisterCommand.run( self )
        
