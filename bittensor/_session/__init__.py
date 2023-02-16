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
import sys
import bittensor
from rich.prompt import Confirm, Prompt, PromptBase

class Session():

    def __init__(
        self,
        coldkey: str,
        hotkey: str,
        network: str,
    ):
        # Get the user's wallet.
        self.wallet = bittensor.wallet( 
            name = coldkey, 
            hotkey = hotkey 
        )
        if not self.wallet.coldkey_file.exists_on_device():
            if not Confirm.ask("[bold yellow]$[/bold yellow] You dont have a wallet named [bold white]\"{}\"[/bold white], would you like to create it?".format( coldkey )):
                sys.exit()
            self.wallet.create_new_coldkey()
        if not self.wallet.hotkey_file.exists_on_device():
            if not Confirm.ask("[bold yellow]$[/bold yellow] You dont have a wallet-hotkey named [bold white]\"{}\"[/bold white], would you like to create it?".format( hotkey )):
                sys.exit()
            self.wallet.create_new_hotkey()    

        # Get the user specified network.    
        self.subtensor = bittensor.subtensor( network = network )

        # Loading the metagraph.
        bittensor.turn_console_off()
        bittensor.__use_console__ = False
        self.metagraph = bittensor.metagraph( subtensor = self.subtensor ).load()
        if self.subtensor.block - self.metagraph.block.item() > 7200:
            self.metagraph.sync().save()
        self.dendrite = bittensor.dendrite( wallet = self.wallet )
        print ('bittensor.init\n\t{}\n\t{}'.format( self.wallet, self.subtensor ) )

    def default_uid( self ) -> int:
        return self.metagraph.I.sort()[1][-1].item()


