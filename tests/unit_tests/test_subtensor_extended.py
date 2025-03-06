import unittest.mock

import pytest

import async_substrate_interface.errors

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.chain_identity import ChainIdentity
from bittensor.core.chain_data.delegate_info import DelegatedInfo, DelegateInfo
from bittensor.core.chain_data.dynamic_info import DynamicInfo
from bittensor.core.chain_data.neuron_info import NeuronInfo
from bittensor.core.chain_data.neuron_info_lite import NeuronInfoLite
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.core.chain_data.stake_info import StakeInfo
from bittensor.utils.balance import Balance
from tests.helpers.helpers import assert_submit_signed_extrinsic


@pytest.fixture
def wallet():
    return unittest.mock.Mock()


@pytest.fixture
def mock_delegate_info():
    return {
        "delegate_ss58": tuple(bytearray(32)),
        "total_stake": {},
        "nominators": [],
        "owner_ss58": tuple(bytearray(32)),
        "take": 2**16 - 1,
        "validator_permits": [],
        "registrations": [],
        "return_per_1000": 2,
        "total_daily_return": 3,
    }


@pytest.fixture
def mock_dynamic_info():
    return {
        "netuid": 0,
        "owner_hotkey": tuple(bytearray(32)),
        "owner_coldkey": tuple(bytearray(32)),
        "subnet_name": (114, 111, 111, 116),
        "token_symbol": (206, 164),
        "tempo": 100,
        "last_step": 4919910,
        "blocks_since_last_step": 84234,
        "emission": 0,
        "alpha_in": 14723086336554,
        "alpha_out": 6035890271491007,
        "tao_in": 6035892206947246,
        "alpha_out_emission": 0,
        "alpha_in_emission": 0,
        "tao_in_emission": 0,
        "pending_alpha_emission": 0,
        "pending_root_emission": 0,
        "subnet_volume": 2240411565906691,
        "network_registered_at": 0,
        "subnet_identity": None,
        "moving_price": {"bits": 0},
    }


@pytest.fixture
def mock_neuron_info():
    return {
        "active": 0,
        "axon_info": {
            "ip_type": 4,
            "ip": 2130706433,
            "placeholder1": 0,
            "placeholder2": 0,
            "port": 8080,
            "protocol": 0,
            "version": 1,
        },
        "bonds": [],
        "coldkey": tuple(bytearray(32)),
        "consensus": 0.0,
        "dividends": 0.0,
        "emission": 0.0,
        "hotkey": tuple(bytearray(32)),
        "incentive": 0.0,
        "is_null": False,
        "last_update": 0,
        "netuid": 1,
        "prometheus_info": {
            "block": 0,
            "ip_type": 0,
            "ip": 0,
            "port": 0,
            "version": 1,
        },
        "pruning_score": 0.0,
        "rank": 0.0,
        "stake_dict": {},
        "stake": [],
        "total_stake": 1e12,
        "trust": 0.0,
        "uid": 1,
        "validator_permit": True,
        "validator_trust": 0.0,
        "weights": [],
    }


def test_all_subnets(mock_substrate, subtensor, mock_dynamic_info):
    mock_substrate.runtime_call.return_value.decode.return_value = [
        mock_dynamic_info,
    ]

    result = subtensor.all_subnets()

    assert result == [
        DynamicInfo(
            netuid=0,
            owner_hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            owner_coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            subnet_name="root",
            symbol="Τ",
            tempo=100,
            last_step=4919910,
            blocks_since_last_step=84234,
            emission=Balance(0),
            alpha_in=Balance(14723086336554),
            alpha_out=Balance(6035890271491007),
            tao_in=Balance(6035892206947246),
            price=Balance.from_tao(1),
            k=88866962081017766138079430284,
            is_dynamic=False,
            alpha_out_emission=Balance(0),
            alpha_in_emission=Balance(0),
            tao_in_emission=Balance(0),
            pending_alpha_emission=Balance(0),
            pending_root_emission=Balance(0),
            network_registered_at=0,
            subnet_volume=Balance(2240411565906691),
            subnet_identity=None,
            moving_price=0.0,
        ),
    ]

    mock_substrate.runtime_call.assert_called_once_with(
        "SubnetInfoRuntimeApi",
        "get_all_dynamic_info",
        block_hash=None,
    )


def test_bonds(mock_substrate, subtensor, mocker):
    mock_substrate.query_map.return_value = [
        (0, mocker.Mock(value=[(1, 100), (2, 200)])),
        (1, mocker.Mock(value=[(0, 150), (2, 250)])),
        (2, mocker.Mock(value=None)),
    ]

    result = subtensor.bonds(netuid=1)

    assert result == [
        (0, [(1, 100), (2, 200)]),
        (1, [(0, 150), (2, 250)]),
    ]

    mock_substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Bonds",
        params=[1],
        block_hash=None,
    )


def test_burned_register(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=NeuronInfo.get_null_neuron(),
    )
    mocker.patch.object(subtensor, "get_balance")

    success = subtensor.burned_register(
        wallet,
        netuid=1,
    )

    assert success is True

    subtensor.get_neuron_for_pubkey_and_subnet.assert_called_once_with(
        wallet.hotkey.ss58_address,
        netuid=1,
        block=mock_substrate.get_block_number.return_value,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="burned_register",
        call_params={
            "netuid": 1,
            "hotkey": wallet.hotkey.ss58_address,
        },
        wait_for_finalization=True,
        wait_for_inclusion=False,
    )


def test_get_all_commitments(mock_substrate, subtensor, mocker):
    mock_substrate.query_map.return_value = [
        (
            (tuple(bytearray(32)),),
            {
                "info": {
                    "fields": [
                        (
                            {
                                "Raw4": (tuple(b"Test"),),
                            },
                        ),
                    ],
                },
            },
        ),
    ]

    result = subtensor.get_all_commitments(netuid=1)

    assert result == {
        "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM": "Test",
    }

    mock_substrate.query_map.assert_called_once_with(
        module="Commitments",
        storage_function="CommitmentOf",
        params=[1],
        block_hash=None,
    )


def test_get_balance(mock_substrate, subtensor):
    mock_substrate.query.return_value = {
        "data": {
            "free": 123,
        },
    }

    result = subtensor.get_balance(
        "hotkey_ss58",
    )

    assert result == Balance(123)

    mock_substrate.query.assert_called_once_with(
        module="System",
        storage_function="Account",
        params=["hotkey_ss58"],
        block_hash=None,
    )


def test_get_balances(mock_substrate, subtensor, mocker):
    create_storage_keys = [
        mocker.Mock(),
        mocker.Mock(),
    ]

    mock_substrate.create_storage_key.side_effect = create_storage_keys
    mock_substrate.query_multi.return_value = [
        (
            mocker.Mock(
                params=["hotkey1_ss58"],
            ),
            {
                "data": {
                    "free": 1,
                },
            },
        ),
        (
            mocker.Mock(
                params=["hotkey2_ss58"],
            ),
            {
                "data": {
                    "free": 2,
                },
            },
        ),
    ]

    result = subtensor.get_balances(
        "hotkey1_ss58",
        "hotkey2_ss58",
    )

    assert result == {
        "hotkey1_ss58": Balance(1),
        "hotkey2_ss58": Balance(2),
    }

    mock_substrate.query_multi.assert_called_once_with(
        create_storage_keys,
        block_hash=mock_substrate.get_chain_head.return_value,
    )
    mock_substrate.create_storage_key.assert_has_calls(
        [
            mocker.call(
                "System",
                "Account",
                ["hotkey1_ss58"],
                block_hash=mock_substrate.get_chain_head.return_value,
            ),
            mocker.call(
                "System",
                "Account",
                ["hotkey2_ss58"],
                block_hash=mock_substrate.get_chain_head.return_value,
            ),
        ]
    )


def test_get_block_hash_none(mock_substrate, subtensor):
    result = subtensor.get_block_hash(block=None)

    assert result == mock_substrate.get_chain_head.return_value

    mock_substrate.get_chain_head.assert_called_once()


def test_get_children(mock_substrate, subtensor, wallet):
    mock_substrate.query.return_value.value = [
        (
            2**64 - 1,
            (tuple(bytearray(32)),),
        ),
    ]

    success, children, error = subtensor.get_children(
        "hotkey_ss58",
        netuid=1,
    )

    assert success is True
    assert children == [
        (
            1.0,
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        ),
    ]
    assert error == ""

    mock_substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="ChildKeys",
        params=["hotkey_ss58", 1],
        block_hash=None,
    )


def test_get_current_weight_commit_info(mock_substrate, subtensor, wallet, mocker):
    mock_substrate.query_map.return_value.records = [
        (
            mocker.ANY,
            [
                (
                    bytearray(32),
                    b"data",
                    123,
                ),
            ],
        ),
    ]

    result = subtensor.get_current_weight_commit_info(
        netuid=1,
    )

    assert result == [
        (
            "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            "0x64617461",
            123,
        ),
    ]

    mock_substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="CRV3WeightCommits",
        params=[1],
        block_hash=None,
    )


def test_get_delegate_by_hotkey(mock_substrate, subtensor, mock_delegate_info):
    mock_substrate.runtime_call.return_value.value = mock_delegate_info

    result = subtensor.get_delegate_by_hotkey(
        "hotkey_ss58",
    )

    assert result == DelegateInfo(
        hotkey_ss58="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        owner_ss58="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        take=1.0,
        validator_permits=[],
        registrations=[],
        return_per_1000=Balance(2),
        total_daily_return=Balance(3),
        total_stake={},
        nominators={},
    )

    mock_substrate.runtime_call.assert_called_once_with(
        "DelegateInfoRuntimeApi",
        "get_delegate",
        ["hotkey_ss58"],
        None,
    )


def test_get_delegate_identities(mock_substrate, subtensor, mocker):
    mock_substrate.query_map.return_value = [
        (
            (tuple(bytearray(32)),),
            mocker.Mock(
                value={
                    "additional": "Additional",
                    "description": "Description",
                    "discord": "",
                    "github_repo": "https://github.com/opentensor/bittensor",
                    "image": "",
                    "name": "Chain Delegate",
                    "url": "https://www.example.com",
                },
            ),
        ),
    ]

    result = subtensor.get_delegate_identities()

    assert result == {
        "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM": ChainIdentity(
            additional="Additional",
            description="Description",
            discord="",
            github="https://github.com/opentensor/bittensor",
            image="",
            name="Chain Delegate",
            url="https://www.example.com",
        ),
    }


def test_get_delegated(mock_substrate, subtensor, mock_delegate_info):
    mock_substrate.runtime_call.return_value.value = [
        (
            mock_delegate_info,
            (
                0,
                999,
            ),
        ),
    ]

    result = subtensor.get_delegated(
        "coldkey_ss58",
    )

    assert result == [
        DelegatedInfo(
            hotkey_ss58="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            owner_ss58="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            take=1.0,
            validator_permits=[],
            registrations=[],
            return_per_1000=Balance(2),
            total_daily_return=Balance(3),
            netuid=0,
            stake=Balance(999),
        ),
    ]

    mock_substrate.runtime_call.assert_called_once_with(
        "DelegateInfoRuntimeApi",
        "get_delegated",
        ["coldkey_ss58"],
        None,
    )


def test_get_neuron_certificate(mock_substrate, subtensor):
    mock_substrate.query.return_value = {
        "public_key": (tuple(b"CERTDATA"),),
        "algorithm": 63,
    }

    result = subtensor.get_neuron_certificate(
        "hotkey_ss58",
        netuid=1,
    )

    assert result == "?CERTDATA"

    mock_substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="NeuronCertificates",
        params=[1, "hotkey_ss58"],
        block_hash=None,
    )


def test_get_stake_for_coldkey(mock_substrate, subtensor, mocker):
    mock_substrate.runtime_call.return_value.value = [
        {
            "coldkey": tuple(bytearray(32)),
            "drain": 0,
            "emission": 3,
            "hotkey": tuple(bytearray(32)),
            "is_registered": True,
            "locked": 2,
            "netuid": 1,
            "stake": 999,
        },
        # filter out (stake=0):
        {
            "coldkey": tuple(bytearray(32)),
            "drain": 1000,
            "emission": 1000,
            "hotkey": tuple(bytearray(32)),
            "is_registered": True,
            "locked": 1000,
            "netuid": 2,
            "stake": 0,
        },
    ]

    result = subtensor.get_stake_for_coldkey(
        "coldkey_ss58",
    )

    assert result == [
        StakeInfo(
            coldkey_ss58="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            drain=0,
            emission=Balance(3),
            hotkey_ss58="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            is_registered=True,
            locked=Balance(2),
            netuid=1,
            stake=Balance(999),
        ),
    ]

    mock_substrate.runtime_call.assert_called_once_with(
        "StakeInfoRuntimeApi",
        "get_stake_info_for_coldkey",
        ["coldkey_ss58"],
        None,
    )


def test_filter_netuids_by_registered_hotkeys(
    mock_substrate, subtensor, wallet, mocker
):
    mock_substrate.query_map.return_value = mocker.MagicMock(
        **{
            "__iter__.return_value": iter(
                [
                    (
                        2,
                        mocker.Mock(
                            value=1,
                        ),
                    ),
                    (
                        3,
                        mocker.Mock(
                            value=1,
                        ),
                    ),
                ]
            ),
        },
    )

    result = subtensor.filter_netuids_by_registered_hotkeys(
        all_netuids=[0, 1, 2],
        filter_for_netuids=[2],
        all_hotkeys=[wallet],
        block=10,
    )

    assert result == [2]

    mock_substrate.get_block_hash.assert_called_once_with(10)
    mock_substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IsNetworkMember",
        params=[wallet.hotkey.ss58_address],
        block_hash=mock_substrate.get_block_hash.return_value,
    )


def test_last_drand_round(mock_substrate, subtensor):
    mock_substrate.query.return_value.value = 123

    result = subtensor.last_drand_round()

    assert result == 123

    mock_substrate.query.assert_called_once_with(
        module="Drand",
        storage_function="LastStoredRound",
    )


@pytest.mark.parametrize(
    "wait",
    (
        True,
        False,
    ),
)
def test_move_stake(mock_substrate, subtensor, wallet, wait):
    success = subtensor.move_stake(
        wallet,
        origin_hotkey="origin_hotkey",
        origin_netuid=1,
        destination_hotkey="destination_hotkey",
        destination_netuid=2,
        amount=Balance(1),
        wait_for_finalization=wait,
        wait_for_inclusion=wait,
    )

    assert success is True

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="move_stake",
        call_params={
            "origin_hotkey": "origin_hotkey",
            "origin_netuid": 1,
            "destination_hotkey": "destination_hotkey",
            "destination_netuid": 2,
            "alpha_amount": 1,
        },
        wait_for_finalization=wait,
        wait_for_inclusion=wait,
    )


def test_move_stake_insufficient_stake(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(subtensor, "get_stake", return_value=Balance(0))

    success = subtensor.move_stake(
        wallet,
        origin_hotkey="origin_hotkey",
        origin_netuid=1,
        destination_hotkey="destination_hotkey",
        destination_netuid=2,
        amount=Balance(1),
    )

    assert success is False

    mock_substrate.submit_extrinsic.assert_not_called()


def test_move_stake_error(mock_substrate, subtensor, wallet, mocker):
    mock_substrate.submit_extrinsic.return_value = mocker.Mock(
        error_message="ERROR",
        is_success=False,
    )

    success = subtensor.move_stake(
        wallet,
        origin_hotkey="origin_hotkey",
        origin_netuid=1,
        destination_hotkey="destination_hotkey",
        destination_netuid=2,
        amount=Balance(1),
    )

    assert success is False

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="move_stake",
        call_params={
            "origin_hotkey": "origin_hotkey",
            "origin_netuid": 1,
            "destination_hotkey": "destination_hotkey",
            "destination_netuid": 2,
            "alpha_amount": 1,
        },
        wait_for_finalization=False,
        wait_for_inclusion=True,
    )


def test_move_stake_exception(mock_substrate, subtensor, wallet):
    mock_substrate.submit_extrinsic.side_effect = RuntimeError

    success = subtensor.move_stake(
        wallet,
        origin_hotkey="origin_hotkey",
        origin_netuid=1,
        destination_hotkey="destination_hotkey",
        destination_netuid=2,
        amount=Balance(1),
    )

    assert success is False

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="move_stake",
        call_params={
            "origin_hotkey": "origin_hotkey",
            "origin_netuid": 1,
            "destination_hotkey": "destination_hotkey",
            "destination_netuid": 2,
            "alpha_amount": 1,
        },
        wait_for_finalization=False,
        wait_for_inclusion=True,
    )


def test_neurons(mock_substrate, subtensor, mock_neuron_info):
    mock_substrate.runtime_call.return_value.value = [
        mock_neuron_info,
    ]

    neurons = subtensor.neurons(netuid=1)

    assert neurons == [
        NeuronInfo(
            axon_info=AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            ),
            active=0,
            bonds=[],
            coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            consensus=0.0,
            dividends=0.0,
            emission=0.0,
            hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            incentive=0.0,
            is_null=False,
            last_update=0,
            netuid=1,
            prometheus_info=PrometheusInfo(
                block=0,
                version=1,
                ip="0.0.0.0",
                port=0,
                ip_type=0,
            ),
            pruning_score=0.0,
            rank=0.0,
            stake_dict={},
            stake=Balance(0),
            total_stake=Balance(0),
            trust=0.0,
            uid=1,
            validator_permit=True,
            validator_trust=0.0,
            weights=[],
        ),
    ]

    mock_substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neurons",
        [1],
        None,
    )


def test_neurons_lite(mock_substrate, subtensor, mock_neuron_info):
    mock_substrate.runtime_call.return_value.value = [
        mock_neuron_info,
    ]

    result = subtensor.neurons_lite(netuid=1)

    assert result == [
        NeuronInfoLite(
            axon_info=AxonInfo(
                version=1,
                ip="127.0.0.1",
                port=8080,
                ip_type=4,
                hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
                coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            ),
            active=0,
            coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            consensus=0.0,
            dividends=0.0,
            emission=0.0,
            hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
            incentive=0.0,
            is_null=False,
            last_update=0,
            netuid=1,
            prometheus_info=PrometheusInfo(
                block=0,
                version=1,
                ip="0.0.0.0",
                port=0,
                ip_type=0,
            ),
            pruning_score=0.0,
            rank=0.0,
            stake_dict={},
            stake=Balance(0),
            total_stake=Balance(0),
            trust=0.0,
            uid=1,
            validator_permit=True,
            validator_trust=0.0,
        ),
    ]

    mock_substrate.runtime_call.assert_called_once_with(
        "NeuronInfoRuntimeApi",
        "get_neurons_lite",
        [1],
        None,
    )


def test_subnet(mock_substrate, subtensor, mock_dynamic_info):
    mock_substrate.runtime_call.return_value.decode.return_value = mock_dynamic_info

    result = subtensor.subnet(netuid=0)

    assert result == DynamicInfo(
        netuid=0,
        owner_hotkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        owner_coldkey="5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM",
        subnet_name="root",
        symbol="Τ",
        tempo=100,
        last_step=4919910,
        blocks_since_last_step=84234,
        emission=Balance(0),
        alpha_in=Balance(14723086336554),
        alpha_out=Balance(6035890271491007),
        tao_in=Balance(6035892206947246),
        price=Balance.from_tao(1),
        k=88866962081017766138079430284,
        is_dynamic=False,
        alpha_out_emission=Balance(0),
        alpha_in_emission=Balance(0),
        tao_in_emission=Balance(0),
        pending_alpha_emission=Balance(0),
        pending_root_emission=Balance(0),
        network_registered_at=0,
        subnet_volume=Balance(2240411565906691),
        subnet_identity=None,
        moving_price=0.0,
    )

    mock_substrate.runtime_call.assert_called_once_with(
        "SubnetInfoRuntimeApi",
        "get_dynamic_info",
        params=[0],
        block_hash=None,
    )


def test_subtensor_contextmanager(mock_substrate, subtensor):
    with subtensor:
        pass

    mock_substrate.close.assert_called_once()


def test_swap_stake(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(subtensor, "get_stake", return_value=Balance(1000))
    mocker.patch.object(
        subtensor,
        "get_hotkey_owner",
        autospec=True,
        return_value=wallet.coldkeypub.ss58_address,
    )

    result = subtensor.swap_stake(
        wallet,
        wallet.hotkey.ss58_address,
        origin_netuid=1,
        destination_netuid=2,
        amount=Balance(999),
    )

    assert result is True

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="swap_stake",
        call_params={
            "hotkey": wallet.hotkey.ss58_address,
            "origin_netuid": 1,
            "destination_netuid": 2,
            "alpha_amount": 999,
        },
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )


@pytest.mark.parametrize(
    "query,result",
    (
        (
            None,
            None,
        ),
        (
            {
                "additional": "Additional",
                "description": "Description",
                "discord": "",
                "github_repo": "https://github.com/opentensor/bittensor",
                "image": "",
                "name": "Chain Delegate",
                "url": "https://www.example.com",
            },
            ChainIdentity(
                additional="Additional",
                description="Description",
                discord="",
                github="https://github.com/opentensor/bittensor",
                image="",
                name="Chain Delegate",
                url="https://www.example.com",
            ),
        ),
    ),
)
def test_query_identity(mock_substrate, subtensor, query, result):
    mock_substrate.query.return_value = query

    identity = subtensor.query_identity(
        "coldkey_ss58",
    )

    assert identity == result

    mock_substrate.query.assert_called_once_with(
        module="SubtensorModule",
        storage_function="IdentitiesV2",
        params=["coldkey_ss58"],
        block_hash=None,
    )


def test_register(mock_substrate, subtensor, wallet, mocker):
    create_pow = mocker.patch(
        "bittensor.core.extrinsics.registration.create_pow",
        return_value=mocker.Mock(
            **{
                "is_stale.return_value": False,
                "seal": b"\1\2\3",
            },
        ),
    )
    mocker.patch.object(
        subtensor,
        "get_neuron_for_pubkey_and_subnet",
        return_value=NeuronInfo.get_null_neuron(),
    )

    result = subtensor.register(
        wallet,
        netuid=1,
    )

    assert result is True

    subtensor.get_neuron_for_pubkey_and_subnet.assert_called_once_with(
        hotkey_ss58=wallet.hotkey.ss58_address,
        netuid=1,
        block=mock_substrate.get_block_number.return_value,
    )
    create_pow.assert_called_once_with(
        subtensor=subtensor,
        wallet=wallet,
        netuid=1,
        output_in_place=True,
        cuda=False,
        num_processes=None,
        update_interval=None,
        log_verbose=False,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="register",
        call_params={
            "block_number": create_pow.return_value.block_number,
            "coldkey": wallet.coldkeypub.ss58_address,
            "hotkey": wallet.hotkey.ss58_address,
            "netuid": 1,
            "nonce": create_pow.return_value.nonce,
            "work": [1, 2, 3],
        },
    )


@pytest.mark.parametrize(
    "success",
    [
        True,
        False,
    ],
)
def test_register_subnet(mock_substrate, subtensor, wallet, mocker, success):
    mocker.patch.object(subtensor, "get_balance", return_value=Balance(100))
    mocker.patch.object(subtensor, "get_subnet_burn_cost", return_value=Balance(10))

    mock_substrate.submit_extrinsic.return_value = mocker.Mock(
        is_success=success,
    )

    result = subtensor.register_subnet(
        wallet,
    )

    assert result is success

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="register_network",
        call_params={
            "hotkey": wallet.hotkey.ss58_address,
            "mechid": 1,
        },
    )


def test_register_subnet_insufficient_funds(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(subtensor, "get_balance", return_value=Balance(0))
    mocker.patch.object(subtensor, "get_subnet_burn_cost", return_value=Balance(10))

    success = subtensor.register_subnet(
        wallet,
    )

    assert success is False

    mock_substrate.submit_extrinsic.assert_not_called()


def test_root_register(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(
        subtensor, "get_balance", autospec=True, return_value=Balance(100)
    )
    mocker.patch.object(subtensor, "get_hyperparameter", autospec=True, return_value=10)
    mocker.patch.object(
        subtensor, "is_hotkey_registered_on_subnet", autospec=True, return_value=False
    )

    success = subtensor.root_register(wallet)

    assert success is True

    subtensor.get_balance.assert_called_once_with(
        wallet.coldkeypub.ss58_address,
        block=mock_substrate.get_block_number.return_value,
    )
    subtensor.get_hyperparameter.assert_called_once()
    subtensor.is_hotkey_registered_on_subnet.assert_called_once_with(
        wallet.hotkey.ss58_address,
        0,
        None,
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="root_register",
        call_params={
            "hotkey": wallet.hotkey.ss58_address,
        },
    )


def test_root_register_is_already_registered(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(
        subtensor, "get_balance", autospec=True, return_value=Balance(100)
    )
    mocker.patch.object(subtensor, "get_hyperparameter", autospec=True, return_value=10)
    mocker.patch.object(
        subtensor, "is_hotkey_registered_on_subnet", autospec=True, return_value=True
    )

    success = subtensor.root_register(wallet)

    assert success is True

    subtensor.is_hotkey_registered_on_subnet.assert_called_once_with(
        wallet.hotkey.ss58_address,
        0,
        None,
    )
    mock_substrate.submit_extrinsic.assert_not_called()


def test_root_register_insufficient_balance(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(
        subtensor, "get_balance", autospec=True, return_value=Balance(1)
    )
    mocker.patch.object(subtensor, "get_hyperparameter", autospec=True, return_value=10)

    success = subtensor.root_register(wallet)

    assert success is False

    subtensor.get_balance.assert_called_once_with(
        wallet.coldkeypub.ss58_address,
        block=mock_substrate.get_block_number.return_value,
    )
    mock_substrate.submit_extrinsic.assert_not_called()


def test_root_set_weights(mock_substrate, subtensor, wallet, mocker):
    MIN_ALLOWED_WEIGHTS = 0
    MAX_WEIGHTS_LIMIT = 1

    mock_substrate.query.return_value = 1
    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        autospec=True,
        side_effect=[
            MIN_ALLOWED_WEIGHTS,
            MAX_WEIGHTS_LIMIT,
        ],
    )

    subtensor.root_set_weights(
        wallet,
        netuids=[1, 2],
        weights=[0.5, 0.5],
    )

    subtensor.get_hyperparameter.assert_has_calls(
        [
            mocker.call("MinAllowedWeights", netuid=0),
            mocker.call("MaxWeightsLimit", netuid=0),
        ]
    )
    mock_substrate.query.assert_called_once_with(
        "SubtensorModule",
        "Uids",
        [0, wallet.hotkey.ss58_address],
    )

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="set_root_weights",
        call_params={
            "dests": [1, 2],
            "weights": [65535, 65535],
            "netuid": 0,
            "version_key": 0,
            "hotkey": wallet.hotkey.ss58_address,
        },
        era={
            "period": 5,
        },
        nonce=mock_substrate.get_account_next_index.return_value,
        wait_for_finalization=False,
    )


def test_root_set_weights_no_uid(mock_substrate, subtensor, wallet, mocker):
    mock_substrate.query.return_value = None

    success = subtensor.root_set_weights(
        wallet,
        netuids=[1, 2],
        weights=[0.5, 0.5],
    )

    assert success is False

    mock_substrate.query.assert_called_once_with(
        "SubtensorModule",
        "Uids",
        [0, wallet.hotkey.ss58_address],
    )
    mock_substrate.submit_extrinsic.assert_not_called()


def test_root_set_weights_min_allowed_weights(
    mock_substrate, subtensor, wallet, mocker
):
    mocker.patch.object(
        subtensor,
        "get_hyperparameter",
        autospec=True,
        return_value=5,
    )
    mock_substrate.query.return_value = 1

    with pytest.raises(
        ValueError,
        match="The minimum number of weights required to set weights is 5, got 2",
    ):
        subtensor.root_set_weights(
            wallet,
            netuids=[1, 2],
            weights=[0.5, 0.5],
        )

    subtensor.get_hyperparameter.assert_any_call("MinAllowedWeights", netuid=0)
    mock_substrate.submit_extrinsic.assert_not_called()


def test_sign_and_send_extrinsic(mock_substrate, subtensor, wallet, mocker):
    call = mocker.Mock()

    subtensor.sign_and_send_extrinsic(
        call,
        wallet,
        use_nonce=True,
        period=10,
    )

    mock_substrate.get_account_next_index.assert_called_once_with(
        wallet.hotkey.ss58_address,
    )
    mock_substrate.create_signed_extrinsic.assert_called_once_with(
        call=call,
        era={
            "period": 10,
        },
        keypair=wallet.coldkey,
        nonce=mock_substrate.get_account_next_index.return_value,
    )
    mock_substrate.submit_extrinsic.assert_called_once_with(
        mock_substrate.create_signed_extrinsic.return_value,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )


def test_sign_and_send_extrinsic_raises_error(
    mock_substrate, subtensor, wallet, mocker
):
    mock_substrate.submit_extrinsic.return_value = mocker.Mock(
        error_message={
            "name": "Exception",
        },
        is_success=False,
    )

    with pytest.raises(
        async_substrate_interface.errors.SubstrateRequestException,
        match="{'name': 'Exception'}",
    ):
        subtensor.sign_and_send_extrinsic(
            call=mocker.Mock(),
            wallet=wallet,
            raise_error=True,
        )


@pytest.mark.parametrize(
    "wait",
    (
        True,
        False,
    ),
)
def test_transfer_stake(mock_substrate, subtensor, wallet, mocker, wait):
    mocker.patch.object(
        subtensor,
        "get_hotkey_owner",
        autospec=True,
        return_value=wallet.coldkeypub.ss58_address,
    )

    success = subtensor.transfer_stake(
        wallet,
        "dest",
        "hotkey_ss58",
        origin_netuid=1,
        destination_netuid=1,
        amount=Balance(1),
        wait_for_finalization=wait,
        wait_for_inclusion=wait,
    )

    assert success is True

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="transfer_stake",
        call_params={
            "destination_coldkey": "dest",
            "hotkey": "hotkey_ss58",
            "origin_netuid": 1,
            "destination_netuid": 1,
            "alpha_amount": 1,
        },
        wait_for_finalization=wait,
        wait_for_inclusion=wait,
    )


@pytest.mark.parametrize(
    "side_effect",
    (
        (
            unittest.mock.Mock(
                error_message="ERROR",
                is_success=False,
            ),
        ),
        RuntimeError,
    ),
)
def test_transfer_stake_error(mock_substrate, subtensor, wallet, mocker, side_effect):
    mocker.patch.object(
        subtensor,
        "get_hotkey_owner",
        autospec=True,
        return_value=wallet.coldkeypub.ss58_address,
    )
    mock_substrate.submit_extrinsic.return_value = side_effect

    success = subtensor.transfer_stake(
        wallet,
        "dest",
        "hotkey_ss58",
        origin_netuid=1,
        destination_netuid=1,
        amount=Balance(1),
    )

    assert success is False

    assert_submit_signed_extrinsic(
        mock_substrate,
        wallet.coldkey,
        call_module="SubtensorModule",
        call_function="transfer_stake",
        call_params={
            "destination_coldkey": "dest",
            "hotkey": "hotkey_ss58",
            "origin_netuid": 1,
            "destination_netuid": 1,
            "alpha_amount": 1,
        },
        wait_for_finalization=False,
        wait_for_inclusion=True,
    )


def test_transfer_stake_non_owner(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(
        subtensor,
        "get_hotkey_owner",
        autospec=True,
        return_value="owner2_ss58",
    )

    success = subtensor.transfer_stake(
        wallet,
        "dest",
        "hotkey_ss58",
        origin_netuid=1,
        destination_netuid=1,
        amount=Balance(1),
    )

    assert success is False

    subtensor.get_hotkey_owner.assert_called_once_with(
        "hotkey_ss58",
    )
    mock_substrate.submit_extrinsic.assert_not_called()


def test_transfer_stake_insufficient_stake(mock_substrate, subtensor, wallet, mocker):
    mocker.patch.object(
        subtensor,
        "get_hotkey_owner",
        autospec=True,
        return_value=wallet.coldkeypub.ss58_address,
    )

    with unittest.mock.patch.object(
        subtensor,
        "get_stake",
        return_value=Balance(0),
    ):
        success = subtensor.transfer_stake(
            wallet,
            "dest",
            "hotkey_ss58",
            origin_netuid=1,
            destination_netuid=1,
            amount=Balance(1),
        )

        assert success is False

    mock_substrate.submit_extrinsic.assert_not_called()


def test_wait_for_block(mock_substrate, subtensor, mocker):
    mock_subscription_handler = None

    def get_block_handler(
        current_block_hash,
        header_only,
        subscription_handler,
    ):
        nonlocal mock_subscription_handler
        mock_subscription_handler = mocker.Mock(wraps=subscription_handler)

        for block in range(1, 20):
            if mock_subscription_handler(
                {
                    "header": {
                        "number": block,
                    },
                }
            ):
                return

        assert False

    mock_substrate.get_block.side_effect = [
        {
            "header": {
                "number": 1,
            },
        },
    ]
    mock_substrate._get_block_handler.side_effect = get_block_handler

    subtensor.wait_for_block(block=9)

    assert mock_subscription_handler.call_count == 9


def test_weights(mock_substrate, subtensor):
    mock_substrate.query_map.return_value = [
        (1, unittest.mock.Mock(value=0.5)),
    ]

    results = subtensor.weights(
        netuid=1,
    )

    assert results == [
        (
            1,
            0.5,
        ),
    ]

    mock_substrate.query_map.assert_called_once_with(
        module="SubtensorModule",
        storage_function="Weights",
        params=[1],
        block_hash=None,
    )
