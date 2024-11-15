import random

import pytest
from bittensor.core import settings
from bittensor.core.subtensor import Subtensor


@pytest.fixture
def subtensor():
    setup_data = Subtensor("test")
    yield setup_data
    print("Subtensor closed")


@pytest.mark.parametrize(
    "input_, runtime_api, method, output",
    [
        (
            [1],
            "NeuronInfoRuntimeApi",
            "get_neurons_lite",
            "0x0100"
        ),
        (
            [],
            "SubnetRegistrationRuntimeApi",
            "get_network_registration_cost",
            "0x"
        )
    ]
)
def test_encode_params(subtensor, input_, runtime_api, method, output):
    call_definition = settings.TYPE_REGISTRY["runtime_api"][runtime_api]["methods"][
        method
    ]
    result = subtensor._encode_params(call_definition=call_definition, params=input_)
    assert result == output


def test_blocks_since_last_update(subtensor):
    netuid = 1
    updates = subtensor._get_hyperparameter(param_name="LastUpdate", netuid=netuid)
    assert isinstance(updates, (list, None))
    if updates is not None:
        uid = random.randint(0, len(updates)-1)
        query = subtensor.blocks_since_last_update(netuid=netuid, uid=uid)
        assert isinstance(query, int)
