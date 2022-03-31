from types import SimpleNamespace
import bittensor

from substrateinterface.base import Keypair
import unittest
from unittest.mock import MagicMock
from substrateinterface.exceptions import SubstrateRequestException
from bittensor._subtensor.subtensor_mock import mock_subtensor
from unittest.mock import patch

class TestCli(unittest.TestCase):

    def setUp(self):
        mock_subtensor.kill_global_mock_process()
        self.config = TestCli.construct_config()
        # Mocked objects
        self.mock_neuron = TestCli._neuron_dict_to_namespace(
            dict({
                "version":1,
                "ip":0,
                "port":0,
                "ip_type":0,
                "uid":1,
                "modality":0,
                "hotkey":'some_hotkey',
                "coldkey":'some_coldkey',
                "active":0,
                "last_update":0,
                "priority":0,
                "stake":1000000000000.0,
                "rank":0.0,
                "trust":0.0,
                "consensus":0.0,
                "incentive":0.0,
                "dividends":0.0,
                "emission":0.0,
                "bonds":[],
                "weights":[],
                "is_null":False
            })
        )

    @staticmethod
    def construct_config():
        defaults = bittensor.Config()
        bittensor.subtensor.add_defaults( defaults )
        bittensor.dendrite.add_defaults( defaults )
        bittensor.axon.add_defaults( defaults )
        bittensor.wallet.add_defaults( defaults )
        bittensor.dataset.add_defaults( defaults )
        
        return defaults

    @staticmethod
    def _neuron_dict_to_namespace(neuron_dict) -> SimpleNamespace:
            if neuron_dict['hotkey'] == '5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM':
                return bittensor.subtensor._null_neuron()
            else:
                RAOPERTAO = 1000000000
                U64MAX = 18446744073709551615
                neuron = SimpleNamespace( **neuron_dict )
                neuron.stake = neuron.stake / RAOPERTAO
                neuron.rank = neuron.rank / U64MAX
                neuron.trust = neuron.trust / U64MAX
                neuron.consensus = neuron.consensus / U64MAX
                neuron.incentive = neuron.incentive / U64MAX
                neuron.dividends = neuron.dividends / U64MAX
                neuron.emission = neuron.emission / RAOPERTAO
                neuron.is_null = False
                return neuron

    @staticmethod
    def generate_wallet(coldkey : 'Keypair' = None, hotkey: 'Keypair' = None):
        wallet = bittensor.wallet(_mock=True)   

        if not coldkey:
            coldkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
        if not hotkey:
            hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())

        wallet.set_coldkey(coldkey, encrypt=False, overwrite=True)
        wallet.set_coldkeypub(coldkey, encrypt=False, overwrite=True)    
        wallet.set_hotkey(hotkey, encrypt=False, overwrite=True)
        
        return wallet

    def test_check_configs(self):
        commands = ["run", "transfer", "register", "unstake", 
        "stake", "overview", "new_coldkey", "new_hotkey", 
        "regen_coldkey", "regen_hotkey", "metagraph", "weights", 
        "set_weights", "inspect"]
        config = self.config
        config.no_prompt = True
        config.model = "template_miner"
        config.dest = "no_prompt"
        config.amount = 1
        config.mnemonic = "this is a mnemonic"
        config.uids = [1,2,3]
        config.weights = [0.25, 0.25, 0.25, 0.25]

        cli = bittensor.cli
        
        for cmd in commands:
            config.command = cmd
            cli.check_config(config)

    def test_overview( self ):
        bittensor.subtensor.register = MagicMock(return_value = True)  
        
        config = self.config
        config.command = "overview"
        config.subtensor._mock = True
        config.subtensor.network = "mock"
        config.no_prompt = True

        cli = bittensor.cli(config)
        cli.run()

    def test_register( self ):

        config = self.config
        config.subtensor._mock = True
        config.command = "register"
        config.subtensor.network = "mock"
        config.no_prompt = True

        with patch('bittensor.Subtensor.register', return_value=True):
            cli = bittensor.cli(config)
            cli.run()

    def test_stake( self ):
        wallet = TestCli.generate_wallet()
        bittensor.Subtensor.neuron_for_pubkey = MagicMock(return_value=self.mock_neuron)
        config = self.config
        config.subtensor.network = "mock"
        config.no_prompt = True
        config.subtensor._mock = True
        config.command = "stake"
        config.amount = 0.5
        config.stake_all = False
        config.no_password = True
        
        config.model = "template_miner"

        cli = bittensor.cli(config)
        cli.run()

    def test_new_coldkey( self ):
        
        config = self.config
        config.wallet.name = "new_coldkey_testwallet"

        config.command = "new_coldkey"
        config.amount = 1
        config.dest = "no_prompt"
        config.subtensor._mock = True
        config.model = "template_miner"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_new_hotkey( self ):
        
        #subtensor.register(wallet=wallet)    
        config = self.config
        config.wallet.name = "new_hotkey_testwallet"
        config.command = "new_hotkey"
        config.amount = 1
        config.subtensor.network = "mock"
        config.dest = "no_prompt"
        config.subtensor._mock = True
        config.model = "template_miner"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_hotkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_regen_coldkey( self ):
        config = self.config
        config.wallet.name = "regen_coldkey_testwallet"
        config.command = "regen_coldkey"
        config.amount = 1
        config.subtensor.network = "mock"
        config.dest = "no_prompt"
        config.subtensor._mock = True
        config.model = "template_miner"
        config.mnemonic = "faculty decade seven jelly gospel axis next radio grain radio remain gentle"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_coldkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_regen_hotkey( self ):
        config = self.config
        config.wallet.name = "regen_hotkey_testwallet"
        config.command = "regen_hotkey"
        config.amount = 1
        config.subtensor.network = "mock"
        config.subtensor._mock = True
        config.model = "template_miner"
        config.mnemonic = "faculty decade seven jelly gospel axis next radio grain radio remain gentle"
        config.n_words = 12
        config.use_password = False
        config.no_prompt = True
        config.overwrite_hotkey = True

        cli = bittensor.cli(config)
        cli.run()

    def test_metagraph( self ):    
        config = self.config
        config.wallet.name = "metagraph_testwallet"
        config.command = "metagraph"
        config.subtensor.network = "mock"
        config.no_prompt = True
        config.subtensor._mock = True

        cli = bittensor.cli(config)
        cli.run()

    def test_set_weights( self ):

        config = self.config
        config.wallet.name = "set_weights_testwallet"
        config.subtensor.network = "mock"
        config.no_prompt = True
        config.uids = [1, 2, 3, 4]
        config.weights = [0.25, 0.25, 0.25, 0.25]
        config.subtensor._mock = True
        config.n_words = 12
        config.use_password = False


        config.overwrite_hotkey = True

        # First create a new hotkey
        config.command = "new_hotkey"
        cli = bittensor.cli(config)
        cli.run()
        
        # Now set the weights
        config.command = "set_weights"
        cli.config = config
        cli.run()

    def test_inspect( self ):
        config = self.config
        config.wallet.name = "inspect_testwallet"
        config.subtensor.network = "mock"
        config.no_prompt = True
        config.subtensor._mock = True
        config.n_words = 12
        config.use_password = False
        config.overwrite_coldkey = True
        config.overwrite_hotkey = True

        # First create a new coldkey
        config.command = "new_coldkey"
        cli = bittensor.cli(config)
        cli.run()

        # Now let's give it a hotkey
        config.command = "new_hotkey"
        cli.config = config
        cli.run()

        # Now inspect it
        cli.config.command = "inspect"
        cli.config = config
        cli.run()

        cli.config.command = "list"
        cli.config = config
        cli.run()


if __name__ == "__main__":
    cli = TestCli()
    cli.setUp()
    cli.test_register()