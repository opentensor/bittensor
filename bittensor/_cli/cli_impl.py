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
import argparse
import bittensor


from loguru import logger
logger = logger.opt(colors=True)

class CLI:
    def __init__(self, config: 'bittensor.Config', executor: 'bittensor.executor.Executor' ):
        r""" Initialized a bittensor.CLI object.
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.cli.config()
                executor (:obj:`bittensor.executor.executor`, `required`):
                    bittensor executor object, used to execute cli options.
        """
        self.config = config
        self.executor = executor

    def run ( self ):
        if self.config.command == "transfer":
            self.executor.transfer( amount_tao=self.config.amount, destination=self.config.dest)
        elif self.config.command == "unstake":
            if self.config.unstake_all:
                self.executor.unstake_all()
            else:
                self.executor.unstake( amount_tao =self.config.amount, uid=self.config.uid )
        elif self.config.command == "stake":
            self.executor.stake( amount_tao=self.config.amount, uid=self.config.uid )
        elif self.config.command == "overview":
            self.executor.overview()
        elif self.config.command == "new_coldkey":
            self.executor.create_new_coldkey( n_words=self.config.n_words, use_password=self.config.use_password )
        elif self.config.command == "new_hotkey":
            self.executor.create_new_hotkey( n_words=self.config.n_words, use_password=self.config.use_password )
        elif self.config.command == "regen_coldkey":
            self.executor.regenerate_coldkey( mnemonic=self.config.mnemonic, use_password=self.config.use_password )
        elif self.config.command == "regen_hotkey":
            self.executor.regenerate_hotkey( mnemonic=self.config.mnemonic, use_password=self.config.use_password )
        else:
            logger.critical("The command {} not implemented".format( self.config.command ))
            quit()
