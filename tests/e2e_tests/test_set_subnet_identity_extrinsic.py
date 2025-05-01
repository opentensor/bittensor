import pytest

from bittensor.core.chain_data import SubnetIdentity
from bittensor.utils.btlogging import logging


@pytest.mark.asyncio
async def test_set_subnet_identity_extrinsic_happy_pass(subtensor, alice_wallet):
    logging.console.info(
        "[magenta]Testing `set_subnet_identity_extrinsic` with success result.[/magenta]"
    )

    netuid = subtensor.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # make sure subnet_identity is empty
    assert subtensor.subnet(netuid).subnet_identity is None, (
        "Subnet identity should be None before set"
    )

    # prepare SubnetIdentity for subnet
    subnet_identity = SubnetIdentity(
        subnet_name="e2e test subnet",
        github_repo="e2e test repo",
        subnet_contact="e2e test contact",
        subnet_url="e2e test url",
        discord="e2e test discord",
        description="e2e test description",
        additional="e2e test additional",
    )

    # set SubnetIdentity to subnet
    assert (
        subtensor.set_subnet_identity(
            wallet=alice_wallet,
            netuid=netuid,
            subnet_identity=subnet_identity,
        )[0]
        is True
    ), "Set subnet identity failed"

    # check SubnetIdentity of the subnet
    assert subtensor.subnet(netuid).subnet_identity == subnet_identity


@pytest.mark.asyncio
async def test_set_subnet_identity_extrinsic_failed(
    subtensor, alice_wallet, bob_wallet
):
    """
    Test case for verifying the behavior of the `set_subnet_identity_extrinsic` function in the
    scenario where the result of the function is expected to fail. It ensures proper handling
    and validation when attempting to set the subnet identity under specific conditions.

    Args:
        subtensor: The instance of the subtensor class under test.
        alice_wallet: A mock or test wallet associated with Alice, used for creating a subnet.
        bob_wallet: A mock or test wallet associated with Bob, used for setting the subnet identity.

    Decorators:
        @pytest.mark.asyncio: Marks this test as an asynchronous test.
    """
    logging.console.info(
        "[magenta]Testing `set_subnet_identity_extrinsic` with failed result.[/magenta]"
    )

    netuid = 2

    # Register a subnet, netuid 2
    assert subtensor.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnet_exists(netuid), "Subnet wasn't created successfully"

    # make sure subnet_identity is empty
    assert subtensor.subnet(netuid).subnet_identity is None, (
        "Subnet identity should be None before set"
    )

    # prepare SubnetIdentity for subnet
    subnet_identity = SubnetIdentity(
        subnet_name="e2e test subnet",
        github_repo="e2e test repo",
        subnet_contact="e2e test contact",
        subnet_url="e2e test url",
        discord="e2e test discord",
        description="e2e test description",
        additional="e2e test additional",
    )

    # set SubnetIdentity to subnet
    assert (
        subtensor.set_subnet_identity(
            wallet=bob_wallet,
            netuid=netuid,
            subnet_identity=subnet_identity,
        )[0]
        is False
    ), "Set subnet identity failed"
