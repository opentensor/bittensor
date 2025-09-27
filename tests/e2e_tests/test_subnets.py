import pytest
from bittensor.utils.btlogging import logging


def test_subnets(subtensor, alice_wallet):
    """
    Tests:
    - Querying subnets
    - Filtering subnets
    - Checks default TxRateLimit
    """
    subnets = subtensor.subnets.all_subnets()
    assert len(subnets) == 2

    subtensor.subnets.register_subnet(alice_wallet)

    subnets = subtensor.subnets.all_subnets()
    assert len(subnets) == 3

    netuids = subtensor.wallets.filter_netuids_by_registered_hotkeys(
        all_netuids=[0, 1, 2],
        filter_for_netuids=[2],
        all_hotkeys=[alice_wallet],
        block=subtensor.block,
    )
    assert netuids == [2]

    tx_rate_limit = subtensor.chain.tx_rate_limit()
    assert tx_rate_limit == 1000


@pytest.mark.asyncio
async def test_subnets_async(async_subtensor, alice_wallet):
    """
    Async tests:
    - Querying subnets
    - Filtering subnets
    - Checks default TxRateLimit
    """
    subnets = await async_subtensor.subnets.all_subnets()
    assert len(subnets) == 2

    assert await async_subtensor.subnets.register_subnet(alice_wallet)

    subnets = await async_subtensor.subnets.all_subnets()
    assert len(subnets) == 3

    netuids = await async_subtensor.wallets.filter_netuids_by_registered_hotkeys(
        all_netuids=[0, 1, 2],
        filter_for_netuids=[2],
        all_hotkeys=[alice_wallet],
        block=await async_subtensor.block,
    )
    assert netuids == [2]

    tx_rate_limit = await async_subtensor.chain.tx_rate_limit()
    assert tx_rate_limit == 1000
