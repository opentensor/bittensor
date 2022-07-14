from types import SimpleNamespace
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
          
    core_server = bittensor._neuron.text.core_server.server()
    # test for the basic default gpt2 case
    assert core_server.set_fine_tuning_params() == (True, 'transformer.h.11')
    
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
    config.wallet = bittensor.Config()
    config.wallet.reregister = False # don't reregister the wallet

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    class MockException(Exception):
        pass

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
            network="mock"
        ),
        config=config,
    )

    with patch.multiple(
            'bittensor.Wallet',
            register=mock_register,
            is_registered=MagicMock(return_value=False), # mock the wallet as not registered
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
    config.wallet = bittensor.Config()
    config.wallet.reregister = True # try to reregister the wallet

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    class MockException(Exception):
        pass

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
            network="mock"
        ),
        config=config,
    )

    with patch.multiple(
            'bittensor.Wallet',
            register=mock_register,
            is_registered=MagicMock(return_value=False), # mock the wallet as not registered
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
    config.wallet = bittensor.Config()
    config.wallet.reregister = False # don't reregister the wallet

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    class MockException(Exception):
        pass

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_register = MagicMock(side_effect=exit_early)

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock"
        ),
        config=config,
    )

    with patch.multiple(
            'bittensor.Wallet',
            register=mock_register,
            is_registered=MagicMock(return_value=False), # mock the wallet as not registered
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
    config.wallet = bittensor.Config()
    config.wallet.reregister = True # try to reregister the wallet

    mock_wallet = bittensor.wallet.mock()
    mock_wallet.config = config

    class MockException(Exception):
        pass

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_register = MagicMock(side_effect=exit_early)

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock"
        ),
        config=config,
    )

    with patch.multiple(
            'bittensor.Wallet',
            register=mock_register,
            is_registered=MagicMock(return_value=False), # mock the wallet as not registered
        ):
        
        # Should not exit
        with pytest.raises(MockException):
            # Should raise MockException
            bittensor.neurons.core_validator.neuron.__enter__(
                self=mock_self_neuron
            )

        # Should try to register the neuron
        mock_register.assert_called_once()



if __name__ == '__main__':
    pass
