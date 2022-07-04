from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    config.neuron = bittensor.Config()
    config.neuron.reregister = False

    mock_wallet = MagicMock(
        is_registered=MagicMock(return_value=False), # mock the wallet as not registered
    )
    mock_subtensor = MagicMock()
    mock_model = MagicMock(
        to=MagicMock(return_value=None),
        device=None
    )

    with patch('bittensor.metagraph') as mock_metagraph:
        class MockException(Exception):
            pass

        def exit_early():
            raise MockException('exit_early')

        mock_metagraph.return_value = MagicMock(
            load=exit_early
        )

        # Should not raise MockException
        bittensor.neurons.core_server.run.serve(
                config=config,
                model=mock_model,
                subtensor=mock_subtensor,
                wallet=mock_wallet,
                axon=None,
                metagraph=None,
        )

        # Should have exited before creating a metagraph object
        mock_metagraph.assert_not_called()

def test_coreserver_reregister_flag_true():
    config = bittensor.Config()
    config.neuron = bittensor.Config()
    config.neuron.reregister = True

    mock_wallet = MagicMock(
        is_registered=MagicMock(return_value=False), # mock the wallet as not registered
    )
    mock_subtensor = MagicMock()
    mock_model = MagicMock(
        to=MagicMock(return_value=None),
        device=None
    )

    with patch('bittensor.metagraph') as mock_metagraph:
        class MockException(Exception):
            pass

        def exit_early():
            raise MockException('exit_early')

        mock_metagraph.return_value = MagicMock(
            load=exit_early
        )

        with pytest.raises(MockException):
            # Should raise MockException
            bittensor.neurons.core_server.run.serve(
                    config=config,
                    model=mock_model,
                    subtensor=mock_subtensor,
                    wallet=mock_wallet,
                    axon=None,
                    metagraph=None,
            )

        # Should have continued to creating a metagraph object
        mock_metagraph.assert_called_once()

def test_corevalidator_reregister_flag_false_exit():
    config = bittensor.Config()
    config.neuron = bittensor.Config()
    config.neuron.reregister = False

    mock_register_func = MagicMock()

    class MockException(Exception):
        pass

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_wallet = MagicMock(
        is_registered=MagicMock(return_value=False), # mock the wallet as not registered
        create=MagicMock(return_value=None),
        register=mock_register_func,
        get_uid=exit_early
    )

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock"
        ),
        config=config,
    )

    # Should not raise MockException
    bittensor.neurons.core_validator.neuron.__enter__(
        self=mock_self_neuron
    )

    # Should not try to register the neuron
    mock_register_func.assert_not_called()

def test_corevalidator_reregister_flag_true():
    config = bittensor.Config()
    config.neuron = bittensor.Config()
    config.neuron.reregister = True

    mock_register_func = MagicMock()

    class MockException(Exception):
        pass

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_wallet = MagicMock(
        is_registered=MagicMock(return_value=False), # mock the wallet as not registered
        create=MagicMock(return_value=None),
        register=mock_register_func,
        get_uid=exit_early
    )

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock"
        ),
        config=config,
    )

    with pytest.raises(MockException):
        # Should raise MockException
        bittensor.neurons.core_validator.neuron.__enter__(
            self=mock_self_neuron
        )

    # Should try to register the neuron
    mock_register_func.assert_called_once()

def test_templateminer_reregister_flag_false_exit():
    config = bittensor.Config()
    config.neuron = bittensor.Config()
    config.neuron.reregister = False

    mock_register_func = MagicMock()

    class MockException(Exception):
        pass

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_wallet = MagicMock(
        is_registered=MagicMock(return_value=False), # mock the wallet as not registered
        register=mock_register_func,
        get_uid=exit_early
    )

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.core_validator.neuron,
        subtensor=MagicMock(
            network="mock"
        ),
        config=config,
    )

    # Should not raise MockException
    bittensor.neurons.template_miner.neuron.__enter__(
        self=mock_self_neuron
    )

    # Should not try to register the neuron
    mock_register_func.assert_not_called()

def test_templateminer_reregister_flag_true():
    config = bittensor.Config()
    config.neuron = bittensor.Config()
    config.neuron.reregister = True

    mock_register_func = MagicMock()

    class MockException(Exception):
        pass

    def exit_early(*args, **kwargs):
        raise MockException('exit_early')

    mock_wallet = MagicMock(
        is_registered=MagicMock(return_value=False), # mock the wallet as not registered
        register=mock_register_func,
        get_uid=exit_early
    )

    mock_self_neuron=MagicMock(
        wallet=mock_wallet,
        spec=bittensor.neurons.template_miner.neuron,
        subtensor=MagicMock(
            network="mock"
        ),
        config=config,
    )

    with pytest.raises(MockException):
        # Should raise MockException
        bittensor.neurons.core_validator.neuron.__enter__(
            self=mock_self_neuron
        )

    # Should try to register the neuron
    mock_register_func.assert_called_once()


if __name__ == '__main__':
    test_set_fine_tuning_params()
