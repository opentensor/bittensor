import pytest

from bittensor.core.chain_data import SubnetIdentity
from bittensor.utils.btlogging import logging


def test_set_subnet_identity_extrinsic_happy_pass(subtensor, alice_wallet):
    logging.console.info(
        "Testing [blue]test_set_subnet_identity_extrinsic_happy_pass[/blue]"
    )

    netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.subnets.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Make sure subnet_identity is empty
    assert subtensor.subnets.subnet(netuid).subnet_identity is None, (
        "Subnet identity should be None before set"
    )

    # Prepare SubnetIdentity for subnet
    subnet_identity = SubnetIdentity(
        subnet_name="e2e test subnet",
        github_repo="e2e test repo",
        subnet_contact="e2e test contact",
        subnet_url="e2e test url",
        logo_url="e2e test logo url",
        discord="e2e test discord",
        description="e2e test description",
        additional="e2e test additional",
    )

    # Set SubnetIdentity to subnet
    assert (
        subtensor.subnets.set_subnet_identity(
            wallet=alice_wallet,
            netuid=netuid,
            subnet_identity=subnet_identity,
        )[0]
        is True
    ), "Set subnet identity failed"

    # Check SubnetIdentity of the subnet
    assert subtensor.subnets.subnet(netuid).subnet_identity == subnet_identity

    logging.console.success(
        "✅ Passed [blue]test_set_subnet_identity_extrinsic_happy_pass[/blue]"
    )


@pytest.mark.asyncio
async def test_set_subnet_identity_extrinsic_happy_pass_async(
    async_subtensor, alice_wallet
):
    logging.console.info(
        "Testing [blue]test_set_subnet_identity_extrinsic_happy_pass_async[/blue]"
    )

    netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Subnet wasn't created"
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(netuid), (
        "Subnet wasn't created successfully"
    )

    # Make sure subnet_identity is empty
    assert (await async_subtensor.subnets.subnet(netuid)).subnet_identity is None, (
        "Subnet identity should be None before set"
    )

    # Prepare SubnetIdentity for subnet
    subnet_identity = SubnetIdentity(
        subnet_name="e2e test subnet",
        github_repo="e2e test repo",
        subnet_contact="e2e test contact",
        subnet_url="e2e test url",
        logo_url="e2e test logo url",
        discord="e2e test discord",
        description="e2e test description",
        additional="e2e test additional",
    )

    # Set SubnetIdentity to subnet
    assert (
        await async_subtensor.subnets.set_subnet_identity(
            wallet=alice_wallet,
            netuid=netuid,
            subnet_identity=subnet_identity,
        )
    )[0] is True, "Set subnet identity failed"

    # Check SubnetIdentity of the subnet
    assert (
        await async_subtensor.subnets.subnet(netuid)
    ).subnet_identity == subnet_identity
    logging.console.success(
        "✅ Passed [blue]test_set_subnet_identity_extrinsic_happy_pass_async[/blue]"
    )


def test_set_subnet_identity_extrinsic_failed(subtensor, alice_wallet, bob_wallet):
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
        "Testing [blue]test_set_subnet_identity_extrinsic_failed[/blue]"
    )

    netuid = subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert subtensor.subnets.register_subnet(alice_wallet), "Subnet wasn't created"

    # Verify subnet <netuid> created successfully
    assert subtensor.subnets.subnet_exists(netuid), "Subnet wasn't created successfully"

    # Make sure subnet_identity is empty
    assert subtensor.subnets.subnet(netuid).subnet_identity is None, (
        "Subnet identity should be None before set"
    )

    # Prepare SubnetIdentity for subnet
    subnet_identity = SubnetIdentity(
        subnet_name="e2e test subnet",
        github_repo="e2e test repo",
        subnet_contact="e2e test contact",
        subnet_url="e2e test url",
        logo_url="e2e test logo url",
        discord="e2e test discord",
        description="e2e test description",
        additional="e2e test additional",
    )

    # Set SubnetIdentity to subnet with wrong wallet
    assert (
        subtensor.subnets.set_subnet_identity(
            wallet=bob_wallet,
            netuid=netuid,
            subnet_identity=subnet_identity,
        )[0]
        is False
    ), "Set subnet identity failed"

    logging.console.success(
        "✅ Passed [blue]test_set_subnet_identity_extrinsic_failed[/blue]"
    )


@pytest.mark.asyncio
async def test_set_subnet_identity_extrinsic_failed_async(
    async_subtensor, alice_wallet, bob_wallet
):
    """
    Async test case for verifying the behavior of the `set_subnet_identity_extrinsic` function in the
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
        "Testing [blue]test_set_subnet_identity_extrinsic_failed[/blue]"
    )

    netuid = await async_subtensor.subnets.get_total_subnets()  # 2

    # Register a subnet, netuid 2
    assert await async_subtensor.subnets.register_subnet(alice_wallet), (
        "Subnet wasn't created"
    )

    # Verify subnet <netuid> created successfully
    assert await async_subtensor.subnets.subnet_exists(netuid), (
        "Subnet wasn't created successfully"
    )

    # Make sure subnet_identity is empty
    assert (await async_subtensor.subnets.subnet(netuid)).subnet_identity is None, (
        "Subnet identity should be None before set"
    )

    # Prepare SubnetIdentity for subnet
    subnet_identity = SubnetIdentity(
        subnet_name="e2e test subnet",
        github_repo="e2e test repo",
        subnet_contact="e2e test contact",
        subnet_url="e2e test url",
        logo_url="e2e test logo url",
        discord="e2e test discord",
        description="e2e test description",
        additional="e2e test additional",
    )

    # Set SubnetIdentity to subnet with wrong wallet
    assert (
        await async_subtensor.subnets.set_subnet_identity(
            wallet=bob_wallet,
            netuid=netuid,
            subnet_identity=subnet_identity,
        )
    )[0] is False, "Set subnet identity failed"

    logging.console.success(
        "✅ Passed [blue]test_set_subnet_identity_extrinsic_failed[/blue]"
    )
