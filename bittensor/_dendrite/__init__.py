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

from concurrent.futures.thread import ThreadPoolExecutor
from os import name
import bittensor
import argparse
import copy

from . import dendrite_impl

class dendrite:

    def __new__(
            cls, 
            wallet: 'bittensor.Wallet' = None,
            receptor_pool: 'bittensor.ReceptorPool' = None,
        ) -> 'bittensor.Dendrite':
        r""" Creates a new Dendrite object from passed arguments.
            Args:
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                receptor_pool (:obj:`bittensor.ReceptorPool`, `optional`):
                    bittensor receptor pool, maintains a pool of active TCP connections.
        """
        if wallet == None:
            wallet = bittensor.wallet()
        if receptor_pool == None:
            receptor_pool = bittensor.receptor_pool( wallet = wallet )  
        return dendrite_impl.Dendrite ( 
            wallet = wallet, 
            receptor_pool = receptor_pool 
        )
