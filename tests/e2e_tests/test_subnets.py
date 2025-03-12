from tests.e2e_tests.utils.chain_interactions import (
    sudo_set_admin_utils,
)

def test_subnets(subtensor, alice_wallet):
    """
    Tests:
    - Querying subnets
    - Filtering subnets
    - Checks default TxRateLimit
    """

    subnets = subtensor.all_subnets()

    assert sudo_set_admin_utils(
        subtensor, 
        alice_wallet,
        "sudo_set_network_rate_limit",
        call_params={"rate_limit", "0"},
    ), "Unable to set network rate limit"

    assert len(subnets) == 2

    subtensor.register_subnet(
        alice_wallet,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )

    subnets = subtensor.all_subnets()

    assert len(subnets) == 3

    netuids = subtensor.filter_netuids_by_registered_hotkeys(
        all_netuids=[0, 1, 2],
        filter_for_netuids=[2],
        all_hotkeys=[alice_wallet],
        block=subtensor.block,
    )

    assert netuids == [2]

    tx_rate_limit = subtensor.tx_rate_limit()

    assert tx_rate_limit == 1000
