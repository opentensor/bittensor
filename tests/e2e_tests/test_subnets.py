import pytest


@pytest.mark.asyncio
async def test_subnets(subtensor, alice_wallet):
    """
    Tests:
    - Querying subnets
    - Filtering subnets
    - Checks default TxRateLimit
    """

    subnets = await subtensor.all_subnets()

    assert len(subnets) == 2

    await subtensor.register_subnet(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    subnets = await subtensor.all_subnets()

    assert len(subnets) == 3

    netuids = await subtensor.filter_netuids_by_registered_hotkeys(
        all_netuids=[0, 1, 2],
        filter_for_netuids=[2],
        all_hotkeys=[alice_wallet],
        block=await subtensor.block,
    )

    assert netuids == [2]

    tx_rate_limit = await subtensor.tx_rate_limit()

    assert tx_rate_limit == 1000
