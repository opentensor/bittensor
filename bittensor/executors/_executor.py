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
import time
import torch

from munch import Munch
from tqdm import tqdm
from rich.align import Align
from rich.console import Console
from rich.table import Table

import bittensor
import bittensor.utils.codes as code_utils
from bittensor.utils.balance import Balance

from loguru import logger
logger = logger.opt(colors=True)

class Executor:

    def __init__( 
            self, 
            config: 'Munch', 
<<<<<<< HEAD:bittensor/executors/_executor.py
            wallet: 'bittensor.wallet',
=======
            wallet: 'bittensor.wallet.Wallet',
>>>>>>> 0b16b8c0885d06605ba1434900d514b4685eec4e:bittensor/executor.py
            subtensor: 'bittensor.subtensor.Subtensor',
            metagraph: 'bittensor.metagraph.Metagraph',
            axon: 'bittensor.axon.Axon',
            dendrite: 'bittensor.dendrite.Dendrite'
        ):
        r""" Creates a new Executor object for interfacing with the bittensor API.
            Args:
                config (:obj:`Munch`, `required`): 
                    bittensor.executor.default_config()
<<<<<<< HEAD:bittensor/executors/_executor.py
                wallet (:obj:`bittensor.wallet`, `required`):
=======
                wallet (:obj:`bittensor.wallet.Wallet`, `required`):
>>>>>>> 0b16b8c0885d06605ba1434900d514b4685eec4e:bittensor/executor.py
                    bittensor wallet with hotkey and coldkeypub.
                subtensor (:obj:`bittensor.subtensor.Subtensor`, `required`):
                    Bittensor subtensor chain connection.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `required`):
                    Bittensor metagraph chain state.
                axon (:obj:`bittensor.axon.Axon`, `required`):
                    Bittensor axon server.
                dendrite (:obj:`bittensor.dendrite.Dendrite`, `required`):
                    Bittensor dendrite client.
        """
        self.config = config
        self.wallet = wallet
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.axon = axon
        self.dendrite = dendrite

    def regenerate_coldkey ( self, mnemonic: str, use_password: bool, overwrite: bool = False ):
        r""" Regenerates a colkey under this wallet.
        """
        self.wallet.regenerate_coldkey( mnemonic = mnemonic, use_password = use_password, overwrite = overwrite )

    def regenerate_hotkey ( self, mnemonic: str, use_password: bool, overwrite: bool = False):
        r""" Regenerates a hotkey under this wallet.
        """
        self.wallet.regenerate_hotkey( mnemonic = mnemonic, use_password = use_password, overwrite = overwrite )

    def create_new_coldkey ( self, n_words: int, use_password: bool, overwrite: bool = False ):
        r""" Creates a new coldkey under this wallet.
        """
        self.wallet.create_new_coldkey( n_words = n_words, use_password = use_password, overwrite = overwrite)   

    def create_new_hotkey ( self, n_words: int, use_password: bool, overwrite: bool = False ):  
        r""" Creates a new hotkey under this wallet.
        """
        self.wallet.create_new_hotkey( n_words = n_words, use_password = use_password, overwrite = overwrite )  

    def overview ( self ): 
        r""" Prints an overview for the wallet's colkey.
        """
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        self.metagraph.load()
        self.metagraph.sync()
        self.metagraph.save()
        balance = self.subtensor.get_balance( self.wallet.coldkeypub )

        owned_neurons = [] 
        neuron_endpoints = self.metagraph.neuron_endpoints
        for uid, cold in enumerate(self.metagraph.coldkeys):
            if cold == self.wallet.coldkeypub:
                owned_neurons.append( neuron_endpoints[uid] )

        TABLE_DATA = []

        total_stake = 0
        total_rank = 0
        total_incentive = 0
        total_success = 0
        total_time = 0.0
        logger.info('\nRunning queries ...')
        for neuron in tqdm(owned_neurons):

            # Make query and get response.
            if self.wallet.has_hotkey:
                start_time = time.time()
                result, code = self.dendrite.forward_text( neurons = [neuron], x = [torch.zeros((1,1), dtype=torch.int64)] )
                end_time = time.time()
                code_to_string = code_utils.code_to_string(code.item())
                code_color = code_utils.code_to_color(code.item()) 
                code_str =  '[' + str(code_color) + ']' + code_to_string 
                query_time = '[' + str(code_color) + ']' + "" + '{:.3}'.format(end_time - start_time) + "s"

                if code.item() == 0:
                    total_success += 1
                    total_time += end_time - start_time
            else:
                code_str = '[N/A]'
                query_time = '[N/A]'

            uid = neuron.uid
            stake = self.metagraph.S[ uid ].item()
            rank = self.metagraph.R[ uid ].item()
            incentive = self.metagraph.I[ uid ].item()
            lastemit = int(self.metagraph.block - self.metagraph.lastemit[ uid ])
            lastemit = "[bold green]" + str(lastemit) if lastemit < 3000 else "[bold red]" + str(lastemit)
            row = [str(neuron.uid), neuron.ip + ':' + str(neuron.port), '{:.5}'.format(stake),'{:.5}'.format(rank),  '{:.5}'.format(incentive * 14400), str(lastemit), query_time, code_str, neuron.hotkey]
            TABLE_DATA.append(row)
            total_stake += stake
            total_rank += rank
            total_incentive += incentive * 14400
            
        total_neurons = len(owned_neurons)
        total_stake = '{:.7}'.format(total_stake)
        total_rank = '{:.7}'.format(total_rank)
        total_incentive = '{:.7}'.format(total_incentive)
        total_time = '{:.3}s'.format(total_time / total_neurons) if total_time != 0 else '0.0s'
        total_success = '[bold green]' + str(total_success) + '/[bold red]' +  str(total_neurons - total_success)
                
        console = Console()
        table = Table(show_footer=False)
        table_centered = Align.center(table)
        table.title = (
            "[bold white]Coldkey.pub:" + str(self.wallet.coldkeypub)
        )
        table.add_column("[overline white]UID",  str(total_neurons), footer_style = "overline white", style='yellow')
        table.add_column("[overline white]IP", justify='left', style='dim blue', no_wrap=True) 
        table.add_column("[overline white]STAKE (\u03C4)", str(total_stake), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]RANK (\u03C4)", str(total_rank), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]INCENTIVE (\u03C4/day)", str(total_incentive), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]LastEmit (blocks)", justify='right', no_wrap=True)
        table.add_column("[overline white]Query (sec)", str(total_time), footer_style = "overline white", justify='right', no_wrap=True)
        table.add_column("[overline white]Query (code)", str(total_success), footer_style = "overline white", justify='right', no_wrap=True)
        table.add_column("[overline white]HOTKEY", style='dim blue', no_wrap=False)
        table.show_footer = True
        table.caption = "[bold white]Coldkey Balance: [bold green]\u03C4" + str(balance.tao)

        console.clear()
        for row in TABLE_DATA:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        console = Console()
        console.print(table)


    def unstake_all ( self ):
        r""" Unstaked from all hotkeys associated with this wallet's coldkey.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        self.metagraph.load()
        self.metagraph.sync()
        self.metagraph.save()

        owned_neurons = [] 
        neuron_endpoints = self.metagraph.neuron_endpoints
        for uid, cold in enumerate(self.metagraph.coldkeys):
            if cold == self.wallet.coldkeypub:
                owned_neurons.append( neuron_endpoints[uid] )

        for neuron in owned_neurons:
            stake = self.metagraph.S[ neuron.uid ].item()
            result = self.subtensor.unstake( 
                wallet = self.wallet, 
                amount = Balance(stake), 
                hotkey_id = neuron.hotkey, 
                wait_for_finalization = True, 
                timeout = bittensor.__blocktime__ * 5 
            )
            if result:
                logger.success( "Unstaked: \u03C4{} from uid: {} to coldkey.pub: {}".format( stake, neuron.uid, self.wallet.coldkey.public_key ))
            else:
                logger.critical("Unstaking transaction failed")

    def unstake( self, amount_tao: int, uid: int ):
        r""" Unstaked token of amount to from uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        self.metagraph.load()
        self.metagraph.sync()
        self.metagraph.save()

        neuron = None 
        neuron_endpoints = self.metagraph.neuron_endpoints
        for neuron_uid, cold in enumerate(self.metagraph.coldkeys):
            if neuron_uid == uid:
                if cold != self.wallet.coldkeypub:
                    logger.critical("Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
                    quit()
                else:
                    neuron = neuron_endpoints[neuron_uid]
        if neuron == None:
            logger.critical("No Neuron with uid: {} associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
            quit()


        unstaking_balance = Balance.from_float( amount_tao )
        stake = self.subtensor.get_stake_for_uid(neuron.uid)
        if unstaking_balance > stake:
            logger.critical("Neuron with uid: {} does not have enough stake ({}) to be able to unstake {}".format( uid, stake, unstaking_balance))
            quit()

        logger.info("Requesting unstake of \u03C4{} from hotkey: {} to coldkey: {}".format(unstaking_balance.tao, neuron.hotkey, self.wallet.coldkey.public_key))
        logger.info("Waiting for finalization...")
        result = self.subtensor.unstake (
            wallet = self.wallet, 
            amount = unstaking_balance, 
            hotkey_id = neuron.hotkey, 
            wait_for_finalization = True, 
            timeout = bittensor.__blocktime__ * 5
        )
        if result:
            logger.success("Unstaked: \u03C4{} from uid:{} to coldkey.pub:{}".format(unstaking_balance.tao, neuron.uid, self.wallet.coldkey.public_key))
        else:
            logger.critical("Unstaking transaction failed")

    def stake( self, amount_tao: int, uid: int ):
        r""" Stakes token of amount to hotkey uid.
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        self.metagraph.load()
        self.metagraph.sync()
        self.metagraph.save()
        staking_balance = Balance.from_float( amount_tao )
        account_balance = self.subtensor.get_balance( self.wallet.coldkey.ss58_address )
        if account_balance < staking_balance:
            logger.critical("Not enough balance (\u03C4{}) to stake \u03C4{}".format(account_balance, staking_balance))
            quit()

        neuron = None 
        neuron_endpoints = self.metagraph.neuron_endpoints
        for neuron_uid, cold in enumerate(self.metagraph.coldkeys):
            if neuron_uid == uid:
                if cold != self.wallet.coldkeypub:
                    logger.critical("Neuron with uid: {} is not associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
                    quit()
                else:
                    neuron = neuron_endpoints[neuron_uid]
                    
        if neuron == None:
            logger.critical("No Neuron with uid: {} associated with coldkey.pub: {}".format( uid, self.wallet.coldkey.public_key))
            quit()

        logger.info("Adding stake of \u03C4{} from coldkey {} to hotkey {}".format( staking_balance.tao, self.wallet.coldkey.public_key, neuron.hotkey))
        logger.info("Waiting for finalization...")
        result = self.subtensor.add_stake ( 
            wallet = self.wallet, 
            amount = staking_balance, 
            hotkey_id = neuron.hotkey, 
            wait_for_finalization = True, 
            timeout = bittensor.__blocktime__ * 5
        )
        if result: 
            logger.success("Staked: \u03C4{} to uid: {} from coldkey.pub: {}".format( staking_balance.tao, uid, self.wallet.coldkey.public_key ))
        else:
            logger.critical("Stake transaction failed")

    def transfer( self, amount_tao: int, destination: str):
        r""" Transfers token of amount to dest.
            
        """
        self.wallet.assert_coldkey()
        self.wallet.assert_coldkeypub()
        self.subtensor.connect()
        transfer_balance = Balance.from_float( amount_tao )
        acount_balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
        if acount_balance < transfer_balance:
            logger.critical("Not enough balance (\u03C4{}) to transfer \u03C4{}".format(acount_balance, transfer_balance))
            quit()

        logger.info("Requesting transfer of \u03C4{}, from coldkey: {} to destination: {}".format(transfer_balance.tao, self.wallet.coldkey.public_key, destination))
        logger.info("Waiting for finalization...")
        result = self.subtensor.transfer( 
            wallet = self.wallet, 
            dest = destination, 
            amount = transfer_balance,  
            wait_for_finalization = True, 
            timeout = bittensor.__blocktime__ * 5 
        )
        if result:
            logger.success("Transfer finalized with amount: \u03C4{} to destination: {} from coldkey.pub: {}".format(transfer_balance.tao, destination, self.wallet.coldkey.public_key))
        else:
            logger.critical("Transfer failed")
 
