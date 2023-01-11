from atexit import register
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from more_itertools import side_effect

import pytest

import bittensor
import torch
import torch.nn as nn
from bittensor._subtensor import subtensor
from bittensor._subtensor.subtensor_mock import mock_subtensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def test_set_fine_tuning_params():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 50; network_dim = 10; nlayers_1 = 4; nlayers_2 = 3; max_n = 5; nhead = 2
            self.embedding = torch.nn.Embedding( vocab_size,  network_dim )
            self.encoder_layers = TransformerEncoderLayer( network_dim, nhead )
            self.encoder = TransformerEncoder( self.encoder_layers, nlayers_1 )
            self.encoder2 = TransformerEncoder( self.encoder_layers, nlayers_2 )
            self.decoder = torch.nn.Linear( network_dim, vocab_size , bias=False)
          
    core_server = bittensor._neuron.text.core_server.server(pretrained=False)
    # test for the basic default gpt2 case
    assert core_server.set_fine_tuning_params() == (True, 'h.11')
    
    # test for the case when there are 2 modulelists
    core_server.pre_model = Model()
    assert core_server.set_fine_tuning_params() == (True, 'encoder2.layers.2')
    
    # test for user specification of the number of layers
    core_server.config.neuron.finetune.num_layers = 3
    assert core_server.set_fine_tuning_params() == (True, 'encoder2.layers.0')
    
    # test for user specification of the number of layers
    core_server.config.neuron.finetune.num_layers = 4
    assert core_server.set_fine_tuning_params() == (True, 'encoder.layers.0')
    
    # test for user specification of the number of layers set too large
    core_server.config.neuron.finetune.num_layers = 5
    assert core_server.set_fine_tuning_params() == (False, None)
    
    # test for user specification of the layer name
    core_server.config.neuron.finetune.layer_name = 'encoder2.layers.1'
    assert core_server.set_fine_tuning_params() == (True, 'encoder2.layers.1')
    
    # test for user specification of a non-existing layer name
    core_server.config.neuron.finetune.layer_name = 'non_existing_layer'
    assert core_server.set_fine_tuning_params() == (False, 'non_existing_layer')
    

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            vocab_size = 50; network_dim = 10; nlayers = 2; max_n = 5; nhead = 2
            self.decoder = torch.nn.Linear( network_dim, vocab_size , bias=False)
            
    # test for a non-existing modulelist
    core_server.pre_model = Model()
    core_server.config.neuron.finetune.layer_name = None
    assert core_server.set_fine_tuning_params() == (False, None) 

def test_coreserver_reregister_flag_false_exit():
    config = bittensor.Config()

    config.netuid = -1

    config.wallet = bittensor.Config()
    config.wallet.reregister = False # don't reregister the wallet

    config.subtensor = bittensor.Config()
    config.subtensor.register = bittensor.Config()
    config.subtensor.register.cuda = bittensor.Config()
    config.subtensor.register.cuda.use_cuda = False # don't use cuda on test
    # No need to specify the other config options as they are default to None

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_register = MagicMock(side_effect=exit_early)

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        model=MagicMock(),
        axon=MagicMock(),
        metagraph=MagicMock(),
        spec=bittensor.neurons.core_server.neuron,
        subtensor=MagicMock(
            network="mock",
            register=mock_register
        ),
        config=config,
    )

    with patch.object(
            mock_wallet,
            'is_registered', MagicMock(return_value=False), # mock the wallet as not registered
        ):
        
        # Should exit without calling register
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            # Should not raise MockException
            bittensor.neurons.core_server.neuron.run(
                self=mock_self_neuron
            )

        # Should not try to register the neuron
        mock_register.assert_not_called()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 0 # No error

def test_coreserver_reregister_flag_true():
    config = bittensor.Config()
    
    config.netuid = -1

    config.wallet = bittensor.Config()
    config.wallet.reregister = True # try to reregister the wallet

    config.subtensor = bittensor.Config()
    config.subtensor.register = bittensor.Config()
    config.subtensor.register.cuda = bittensor.Config()
    config.subtensor.register.cuda.use_cuda = False # don't use cuda on test
    # No need to specify the other config options as they are default to None

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_register = MagicMock(side_effect=exit_early)

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        model=MagicMock(),
        axon=MagicMock(),
        metagraph=MagicMock(),
        spec=bittensor.neurons.core_server.neuron,
        subtensor=MagicMock(
            network="mock",
            register=mock_register,
        ),
        config=config,
    )

    with patch.object(
            mock_wallet,
            'is_registered', MagicMock(return_value=False), # mock the wallet as not registered
        ):
        # Should not exit
        with pytest.raises(MockException):
            # Should raise MockException
            bittensor.neurons.core_server.neuron.run(
                self=mock_self_neuron
            )

        # Should try to register the neuron
        mock_register.assert_called_once()

def test_corevalidator_reregister_flag_false_exit():
    config = bittensor.Config()
    
    config.netuid = -1

    config.wallet = bittensor.Config()
    config.wallet.reregister = False # don't reregister the wallet

    config.subtensor = bittensor.Config()
    config.subtensor.register = bittensor.Config()
    config.subtensor.register.cuda = bittensor.Config()
    config.subtensor.register.cuda.use_cuda = False # don't use cuda on test
    # No need to specify the other config options as they are default to None

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_register = MagicMock(side_effect=exit_early)

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock",
            register=mock_register,
        ),
        config=config,
    )

    with patch.object(
            mock_wallet,
            'is_registered', MagicMock(return_value=False), # mock the wallet as not registered
        ):
    
        # Should exit without calling register
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            # Should not raise MockException
            bittensor.neurons.core_validator.neuron.__enter__(
                self=mock_self_neuron
            )

        # Should not try to register the neuron
        mock_register.assert_not_called()
        assert pytest_wrapped_e.type == SystemExit
        assert pytest_wrapped_e.value.code == 0 # No error

def test_corevalidator_reregister_flag_true():
    config = bittensor.Config()
    
    config.netuid = -1

    config.wallet = bittensor.Config()
    config.wallet.reregister = True # try to reregister the wallet

    config.subtensor = bittensor.Config()
    config.subtensor.register = bittensor.Config()
    config.subtensor.register.cuda = bittensor.Config()
    config.subtensor.register.cuda.use_cuda = False # don't use cuda on test
    # No need to specify the other config options as they are default to None

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_register = MagicMock(side_effect=exit_early)

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock",
            register=mock_register,
        ),
        config=config,
    )

    with patch.object(
            mock_wallet,
            'is_registered', MagicMock(return_value=False), # mock the wallet as not registered
        ):
        
        # Should not exit
        with pytest.raises(MockException):
            # Should raise MockException
            bittensor.neurons.core_validator.neuron.__enter__(
                self=mock_self_neuron
            )

        # Should try to register the neuron
        mock_register.assert_called_once()

class MockException(Exception):
    pass

class TestBlacklist(unittest.TestCase):

    @staticmethod
    def construct_config():
        defaults = bittensor.Config()
        bittensor.subtensor.add_defaults( defaults )
        bittensor.dendrite.add_defaults( defaults )
        bittensor.axon.add_defaults( defaults )
        bittensor.wallet.add_defaults( defaults )
        bittensor.dataset.add_defaults( defaults )
        bittensor.logging.add_defaults( defaults )
        bittensor.wandb.add_defaults( defaults )
        bittensor.prometheus.add_defaults( defaults )

        defaults.wandb.api_key = 'test'
        defaults.neuron = bittensor.neurons.core_server.neuron.config()
        defaults.neuron.learning_rate = 0.0001
        defaults.neuron.momentum = 0.9
        defaults.prometheus.level = "OFF"

        defaults.netuid = -1

        return defaults
    
    def exit_early(self, *args, **kwargs):
        raise MockException('exit_early')

    def test_stake_blacklist(self):
        import sys
        sys.setrecursionlimit(100)

        mock_hotkey = "0x0000000000000000000000000000000000000000"
        mock_hotkey_1 = "0x0000000000000000000000000000000000000001"

        mock_subtensor = MagicMock(
            is_hotkey_registered=MagicMock(return_value=True),
        )

        mock_wallet = MagicMock(
            reregister=MagicMock(),
            is_registered=MagicMock(return_value=True),
            hotkey=MagicMock(
                ss58_address=mock_hotkey
            )
        )

        mock_metagraph = MagicMock(
            hotkeys=[
                mock_hotkey,
                mock_hotkey_1,
            ],
            S=[
                torch.tensor(100), # stake for mock_hotkey, uid 0
                torch.tensor(1001), # stake for mock_hotkey_1, uid 1
            ]
        )

        mock_config = self.construct_config()
        
        mock_config.neuron.blacklist = bittensor.Config()
        mock_config.neuron.blacklist.stake = 1000 # blacklist if stake is less than 1000

        mock_model_config = bittensor.neurons.core_server.server.config()
        mock_model_config.neuron = MagicMock(
            disable_blacklist = False
        )

        mock_model = MagicMock(
                            spec=bittensor.neurons.core_server.server,
                            config=mock_model_config,
                        )

        with patch('bittensor.axon.__new__', side_effect=self.exit_early) as mock_new_axon:
            with patch('bittensor.neurons.core_server.neuron.check_config', return_value=True):
                with pytest.raises(MockException):
                    bittensor.neurons.core_server.serve(
                        config=mock_config,
                        model=MagicMock(
                            spec=bittensor.neurons.core_server.server,
                            device="cpu",
                            to=MagicMock(return_value=mock_model),
                            config=mock_model_config,
                        ),
                        subtensor=mock_subtensor,
                        wallet=mock_wallet,
                        axon=None,
                        metagraph=mock_metagraph
                    )

            # args, kwargs
            _, kwargs = mock_new_axon.call_args
            blacklist = kwargs['blacklist']

            # Check that the blacklist rejects below min stake
            assert blacklist(mock_hotkey, bittensor.proto.RequestType.FORWARD) == True

            # Check that the blacklist accepts above min stake
            assert blacklist(mock_hotkey_1, bittensor.proto.RequestType.FORWARD) == False


if __name__ == '__main__':
    pass
