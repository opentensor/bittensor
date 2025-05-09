from bittensor.core.chain_data import StakeInfo
from bittensor.core.extrinsics import utils
from bittensor.utils.balance import Balance


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
