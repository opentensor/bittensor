from bittensor.core.chain_data import StakeInfo
from bittensor.core.extrinsics import utils
from bittensor.utils.balance import Balance
from bittensor_wallet import Keypair


def test_old_stake(subtensor, mocker):
    wallet = mocker.Mock(
        hotkey=mocker.Mock(ss58_address="HK1"),
        coldkeypub=mocker.Mock(ss58_address="CK1"),
    )

    expected_stake = Balance.from_tao(100)

    hotkey_ss58s = ["HK1", "HK2"]
    netuids = [3, 4]
    all_stakes = [
        StakeInfo(
            hotkey_ss58="HK1",
            coldkey_ss58="CK1",
            netuid=3,
            stake=expected_stake,
            locked=Balance.from_tao(10),
            emission=Balance.from_tao(1),
            drain=0,
            is_registered=True,
        ),
    ]

    result = utils.get_old_stakes(wallet, hotkey_ss58s, netuids, all_stakes)

    assert result == [expected_stake, Balance.from_tao(0)]


def test_compute_coldkey_hash():
    """Test compute_coldkey_hash computes correct BlakeTwo256 hash."""
    # Prep
    keypair = Keypair(ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    expected_hash_length = 66  # 0x + 64 hex chars

    # Call
    result = utils.compute_coldkey_hash(keypair)

    # Asserts
    assert result.startswith("0x")
    assert len(result) == expected_hash_length
    assert all(c in "0123456789abcdef" for c in result[2:].lower())


def test_verify_coldkey_hash_match():
    """Test verify_coldkey_hash returns True when hash matches."""
    # Prep
    keypair = Keypair(ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    expected_hash = utils.compute_coldkey_hash(keypair)

    # Call
    result = utils.verify_coldkey_hash(keypair, expected_hash)

    # Asserts
    assert result is True


def test_verify_coldkey_hash_mismatch():
    """Test verify_coldkey_hash returns False when hash doesn't match."""
    # Prep
    keypair = Keypair(ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    wrong_hash = "0x" + "00" * 32

    # Call
    result = utils.verify_coldkey_hash(keypair, wrong_hash)

    # Asserts
    assert result is False


def test_verify_coldkey_hash_case_insensitive():
    """Test verify_coldkey_hash is case insensitive."""
    # Prep
    keypair = Keypair(ss58_address="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
    expected_hash = utils.compute_coldkey_hash(keypair)
    upper_hash = expected_hash.upper()
    lower_hash = expected_hash.lower()

    # Call
    result_upper = utils.verify_coldkey_hash(keypair, upper_hash)
    result_lower = utils.verify_coldkey_hash(keypair, lower_hash)

    # Asserts
    assert result_upper is True
    assert result_lower is True
