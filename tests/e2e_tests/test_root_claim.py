from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from tests.e2e_tests.utils import (
    AdminUtils,
    NETUID,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    REGISTER_NEURON,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_NUM_ROOT_CLAIMS,
    SUDO_SET_TEMPO,
)
import pytest


PROOF_COUNTER = 2


def test_root_claim_swap(subtensor, alice_wallet, bob_wallet, charlie_wallet):
    """Tests root claim Swap logic.

    Steps:
    - activate ROOT net to stake on Alice
    - Register SN and the same validator (Alice) on that subnet to ROOT has an emissions
    - Make sure CK has claim type as Swap
    - Stake to Alice in ROOT
    - Checks in the loop with PROOF_COUNTER numbers of epochs that stake in increased in normative (auto) way
    """
    TEMPO_TO_SET = 10 if subtensor.chain.is_fast_blocks() else 20

    # Activate ROOT net to stake on Alice
    root_sn = TestSubnet(subtensor, 0)
    root_sn.execute_steps(
        [
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    # Register SN and the same validator (Alice) on that subnet to ROOT has an emissions
    sn2 = TestSubnet(subtensor)
    sn2.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
        ]
    )

    stake_balance = Balance.from_tao(10)

    # Stake to Alice in ROOT
    response = subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=root_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    # Make sure stake is the same or greater (if auto claim happened) as before
    stake_info_before = subtensor.staking.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuids=[root_sn.netuid],
    )
    assert stake_info_before[root_sn.netuid].stake >= stake_balance

    # Set claim type to Swap (actually it's already Swap, but just to be sure if default is changed in the future)
    assert subtensor.staking.set_root_claim_type(
        wallet=charlie_wallet, new_root_claim_type="Swap"
    ).success
    assert (
        subtensor.staking.get_root_claim_type(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address
        )
        == "Swap"
    )

    # We skip the era in which the stake was installed, since the emission doesn't occur (Subtensor implementation)
    logging.console.info(f"Skipping stake epoch")
    next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
        netuid=root_sn.netuid
    )
    subtensor.wait_for_block(block=next_epoch_start_block)

    # We do the check over a few epochs
    proof_counter = PROOF_COUNTER
    prev_root_stake = subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=root_sn.netuid,
    )

    # Proof that ROOT stake is changing each last epoch block
    while proof_counter > 0:
        next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
            netuid=root_sn.netuid
        )
        subtensor.wait_for_block(block=next_epoch_start_block)
        charlie_root_stake = subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=root_sn.netuid,
        )
        assert charlie_root_stake > prev_root_stake
        prev_root_stake = charlie_root_stake
        proof_counter -= 1


@pytest.mark.asyncio
async def test_root_claim_swap_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Async tests root claim Swap logic.

    Steps:
    - activate ROOT net to stake on Alice
    - Register SN and the same validator (Alice) on that subnet to ROOT has an emissions
    - Make sure CK has claim type as Swap
    - Stake to Alice in ROOT
    - Checks in the loop with PROOF_COUNTER numbers of epochs that stake in increased in normative (auto) way
    """
    TEMPO_TO_SET = 10 if await async_subtensor.chain.is_fast_blocks() else 20

    # Activate ROOT net to stake on Alice
    root_sn = TestSubnet(async_subtensor, 0)
    await root_sn.async_execute_steps(
        [
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    # Register SN and the same validator (Alice) on that subnet to ROOT has an emissions
    sn2 = TestSubnet(async_subtensor)
    await sn2.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
        ]
    )

    stake_balance = Balance.from_tao(10)

    # Stake to Alice in ROOT
    response = await async_subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=root_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    # Make sure stake is the same or greater (if auto claim happened) as before
    stake_info_before = await async_subtensor.staking.get_stake_for_coldkey_and_hotkey(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuids=[root_sn.netuid],
    )
    assert stake_info_before[root_sn.netuid].stake >= stake_balance

    # Set claim type to Swap (actually it's already Swap, but just to be sure if default is changed in the future)
    assert (
        await async_subtensor.staking.set_root_claim_type(
            wallet=charlie_wallet, new_root_claim_type="Swap"
        )
    ).success
    assert (
        await async_subtensor.staking.get_root_claim_type(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address
        )
        == "Swap"
    )

    # We skip the era in which the stake was installed, since the emission doesn't occur (Subtensor implementation)
    logging.console.info(f"Skipping stake epoch")
    next_epoch_start_block = await async_subtensor.subnets.get_next_epoch_start_block(
        netuid=root_sn.netuid
    )
    await async_subtensor.wait_for_block(block=next_epoch_start_block)

    # We do the check over a few epochs
    proof_counter = PROOF_COUNTER
    prev_root_stake = await async_subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=root_sn.netuid,
    )

    # Proof that ROOT stake is changing each last epoch block
    while proof_counter > 0:
        next_epoch_start_block = (
            await async_subtensor.subnets.get_next_epoch_start_block(
                netuid=root_sn.netuid
            )
        )
        await async_subtensor.wait_for_block(block=next_epoch_start_block)
        charlie_root_stake = await async_subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=root_sn.netuid,
        )
        assert charlie_root_stake > prev_root_stake
        prev_root_stake = charlie_root_stake
        proof_counter -= 1


def test_root_claim_keep_with_zero_num_root_auto_claims(
    subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests root claim Keep logic when NumRootClaim is 0, RootClaimType is Keep.

    Steps:
    - Disable admin freeze window
    - Set NumRootClaim to 0 (THE DEVIL IS IN THE DETAILS, logic compliantly broken without this bc of random auto claim)
    - Activate ROOT net to stake on Alice
    - Set claim type to Keep to staker CK
    - Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    - Make NumRootClaim is 0 to avoid auto claims from random list of coldkeys (just fot test)
    - Make sure CK has claim type as Keep
    - Stake to Alice in ROOT
    - Checks in the loop with PROOF_COUNTER numbers of epochs and check that claimed is 0 and stake is not changed.
    - Root claim manually
    - Check that claimed is recalculated and stake is increased for delegate
    """
    TEMPO_TO_SET = 50 if subtensor.chain.is_fast_blocks() else 10

    # Activate ROOT net to stake on Alice
    # To important set NumRootClaim to 0, so that we can check that it's not changed with random auto claim.
    # Random auto claim is happening even if CK has root claim type as `Keep`
    root_sn = TestSubnet(subtensor, 0)
    root_sn.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            SUDO_SET_NUM_ROOT_CLAIMS(alice_wallet, "SubtensorModule", True, 0),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    # Set claim type to Keep
    assert subtensor.staking.set_root_claim_type(
        wallet=charlie_wallet, new_root_claim_type="Keep"
    ).success

    # Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    # - owner - Bob
    # - validator - Alice
    # - neuron, staker - Charlie (doesn't stake to validator in SN2, in ROOT only)
    sn2 = TestSubnet(subtensor)
    sn2.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    # Make NumRootClaim is 0 (imposable to test if not 0)
    assert subtensor.queries.query_subtensor("NumRootClaim").value == 0

    # Make sure CK has claim type as Keep
    assert (
        subtensor.staking.get_root_claim_type(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address
        )
        == "Keep"
    )

    stake_balance = Balance.from_tao(1000)  # just a dream - stake 1000 TAO to SN0 :D

    # Stake from Charlie to Alice in ROOT
    response = subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=root_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    proof_counter = PROOF_COUNTER

    # proof that ROOT stake isn't changes until it's claimed manually
    while proof_counter > 0:
        next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
            root_sn.netuid
        )
        subtensor.wait_for_block(next_epoch_start_block)

        # Check Charlie stake and claimed
        claimed_stake_charlie = subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert claimed_stake_charlie == 0

        root_claimed_charlie = subtensor.staking.get_root_claimed(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert root_claimed_charlie == 0

        proof_counter -= 1

    # === Check Charlie before manual claim ===
    claimed_before_charlie = subtensor.staking.get_root_claimed(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert claimed_before_charlie == 0

    claimable_stake_before_charlie = subtensor.staking.get_root_claimable_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert claimable_stake_before_charlie != 0

    stake_before_charlie = subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert stake_before_charlie == 0

    logging.console.info(f"[blue]Charlie before:[/blue]")
    logging.console.info(f"RootClaimed: {claimed_before_charlie}")
    logging.console.info(f"RootClaimable stake: {claimable_stake_before_charlie}")
    logging.console.info(f"SN2 Stake: {stake_before_charlie}")

    # === ROOT CLAIM MANUAL ===
    response = subtensor.staking.claim_root(wallet=charlie_wallet, netuids=[sn2.netuid])
    assert response.success, response.message

    # === Check Charlie after manual claim ===
    claimed_after_charlie = subtensor.staking.get_root_claimed(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert claimed_after_charlie >= claimable_stake_before_charlie

    claimable_stake_after_charlie = subtensor.staking.get_root_claimable_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert claimable_stake_after_charlie == claimed_before_charlie

    stake_after_charlie = subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert stake_after_charlie >= claimable_stake_before_charlie

    logging.console.info(f"[blue]Charlie after:[/blue]")
    logging.console.info(f"RootClaimed: {claimed_after_charlie}")
    logging.console.info(f"RootClaimable stake: {claimable_stake_after_charlie}")
    logging.console.info(f"SN2 Stake: {stake_after_charlie}")


@pytest.mark.asyncio
async def test_root_claim_keep_with_zero_num_root_auto_claims_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet
):
    """Tests root claim Keep logic when NumRootClaim is 0, RootClaimType is Keep.

    Steps:
    - Disable admin freeze window
    - Set NumRootClaim to 0 (THE DEVIL IS IN THE DETAILS, logic compliantly broken without this bc of random auto claim)
    - Activate ROOT net to stake on Alice
    - Set claim type to Keep to staker CK
    - Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    - Make NumRootClaim is 0 to avoid auto claims from random list of coldkeys (just fot test)
    - Make sure CK has claim type as Keep
    - Stake to Alice in ROOT
    - Checks in the loop with PROOF_COUNTER numbers of epochs and check that claimed is 0 and stake is not changed.
    - Root claim manually
    - Check that claimed is recalculated and stake is increased for delegate
    """
    TEMPO_TO_SET = 50 if await async_subtensor.chain.is_fast_blocks() else 10

    # Activate ROOT net to stake on Alice
    # To important set NumRootClaim to 0, so that we can check that it's not changed with random auto claim.
    # Random auto claim is happening even if CK has root claim type as `Keep`
    root_sn = TestSubnet(async_subtensor, 0)
    await root_sn.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            SUDO_SET_NUM_ROOT_CLAIMS(alice_wallet, "SubtensorModule", True, 0),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    # Set claim type to Keep
    assert (
        await async_subtensor.staking.set_root_claim_type(
            wallet=charlie_wallet, new_root_claim_type="Keep"
        )
    ).success

    # Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    # - owner - Bob
    # - validator - Alice
    # - neuron, staker - Charlie (doesn't stake to validator in SN2, in ROOT only)
    sn2 = TestSubnet(async_subtensor)
    await sn2.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    # Make NumRootClaim is 0 (imposable to test if not 0)
    assert (await async_subtensor.queries.query_subtensor("NumRootClaim")).value == 0

    # Make sure CK has claim type as Keep
    assert (
        await async_subtensor.staking.get_root_claim_type(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address
        )
        == "Keep"
    )

    stake_balance = Balance.from_tao(1000)  # just a dream - stake 1000 TAO to SN0 :D

    # Stake from Charlie to Alice in ROOT
    response = await async_subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=root_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    proof_counter = PROOF_COUNTER

    # proof that ROOT stake isn't changes until it's claimed manually
    while proof_counter > 0:
        next_epoch_start_block = (
            await async_subtensor.subnets.get_next_epoch_start_block(root_sn.netuid)
        )
        await async_subtensor.wait_for_block(next_epoch_start_block)

        # Check Charlie stake and claimed
        claimed_stake_charlie = await async_subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert claimed_stake_charlie == 0

        root_claimed_charlie = await async_subtensor.staking.get_root_claimed(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert root_claimed_charlie == 0

        proof_counter -= 1

    # === Check Charlie before manual claim ===
    claimed_before_charlie = await async_subtensor.staking.get_root_claimed(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert claimed_before_charlie == 0

    claimable_stake_before_charlie = (
        await async_subtensor.staking.get_root_claimable_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
    )
    assert claimable_stake_before_charlie != 0

    stake_before_charlie = await async_subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert stake_before_charlie == 0

    logging.console.info(f"[blue]Charlie before:[/blue]")
    logging.console.info(f"RootClaimed: {claimed_before_charlie}")
    logging.console.info(f"RootClaimable stake: {claimable_stake_before_charlie}")
    logging.console.info(f"SN2 Stake: {stake_before_charlie}")

    # === ROOT CLAIM MANUAL ===
    response = await async_subtensor.staking.claim_root(
        wallet=charlie_wallet, netuids=[sn2.netuid]
    )
    assert response.success, response.message

    # === Check Charlie after manual claim ===
    claimed_after_charlie = await async_subtensor.staking.get_root_claimed(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert claimed_after_charlie >= claimable_stake_before_charlie

    claimable_stake_after_charlie = (
        await async_subtensor.staking.get_root_claimable_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
    )
    assert claimable_stake_after_charlie == claimed_before_charlie

    stake_after_charlie = await async_subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )
    assert stake_after_charlie >= claimable_stake_before_charlie

    logging.console.info(f"[blue]Charlie after:[/blue]")
    logging.console.info(f"RootClaimed: {claimed_after_charlie}")
    logging.console.info(f"RootClaimable stake: {claimable_stake_after_charlie}")
    logging.console.info(f"SN2 Stake: {stake_after_charlie}")


def test_root_claim_keep_with_random_auto_claims(
    subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet
):
    """Tests root claim Keep logic when NumRootClaim is greater than 0, RootClaimType is Keep.

    Steps:
    - Disable admin freeze window
    - Activate ROOT net to stake on Alice
    - Set claim type to Keep to staker CK
    - Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    - Check NumRootClaim is 5 to have random auto claims logic
    - Make sure CK has claim type as Keep
    - Stake to Alice in ROOT
    - Checks in the loop with PROOF_COUNTER numbers of epochs and check that claimed is greater than the previous one
        and stake is increased for delegate
    """
    TEMPO_TO_SET = 20 if subtensor.chain.is_fast_blocks() else 10

    # Activate ROOT net to stake on Alice
    # To important set NumRootClaim to 0, so that we can check that it's not changed with random auto claim.
    # Random auto claim is happening even if CK has root claim type as `Keep`
    root_sn = TestSubnet(subtensor, 0)
    root_sn.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    # Set claim type to Keep
    assert subtensor.staking.set_root_claim_type(
        wallet=charlie_wallet, new_root_claim_type="Keep"
    ).success

    # Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    # - owner - Bob
    # - validator - Alice
    # - neuron, staker - Charlie (doesn't stake to validator in SN2, in ROOT only)
    sn2 = TestSubnet(subtensor)
    sn2.execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    # Make NumRootClaim is 5 (to have random auto claims logic)
    assert subtensor.queries.query_subtensor("NumRootClaim").value == 5

    # Make sure CK has claim type as Keep
    assert (
        subtensor.staking.get_root_claim_type(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address
        )
        == "Keep"
    )

    stake_balance = Balance.from_tao(1000)  # just a dream - stake 1000 TAO to SN0 :D

    # Stake from Charlie to Alice in ROOT
    response = subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=root_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    proof_counter = PROOF_COUNTER

    prev_claimed_stake_charlie = subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )

    prev_root_claimed_charlie = subtensor.staking.get_root_claimed(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )

    # proof that ROOT stake increases each epoch even RootClaimType is Keep bc of random auto claim takes min 5 coldkeys
    # to do release emissions
    while proof_counter > 0:
        next_epoch_start_block = subtensor.subnets.get_next_epoch_start_block(
            root_sn.netuid
        )
        subtensor.wait_for_block(next_epoch_start_block)

        # Check Charlie stake and claimed
        claimed_stake_charlie = subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert claimed_stake_charlie > prev_claimed_stake_charlie
        prev_claimed_stake_charlie = claimed_stake_charlie

        root_claimed_charlie = subtensor.staking.get_root_claimed(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert root_claimed_charlie > prev_root_claimed_charlie
        prev_root_claimed_charlie = root_claimed_charlie

        proof_counter -= 1


@pytest.mark.asyncio
async def test_root_claim_keep_with_random_auto_claims_async(
    async_subtensor, alice_wallet, bob_wallet, charlie_wallet, dave_wallet
):
    """Tests root claim Keep logic when NumRootClaim is greater than 0, RootClaimType is Keep.

    Steps:
    - Disable admin freeze window
    - Activate ROOT net to stake on Alice
    - Set claim type to Keep to staker CK
    - Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    - Check NumRootClaim is 5 to have random auto claims logic
    - Make sure CK has claim type as Keep
    - Stake to Alice in ROOT
    - Checks in the loop with PROOF_COUNTER numbers of epochs and check that claimed is greater than the previous one
        and stake is increased for delegate
    """
    TEMPO_TO_SET = 20 if await async_subtensor.chain.is_fast_blocks() else 10

    # Activate ROOT net to stake on Alice
    # To important set NumRootClaim to 0, so that we can check that it's not changed with random auto claim.
    # Random auto claim is happening even if CK has root claim type as `Keep`
    root_sn = TestSubnet(async_subtensor, 0)
    await root_sn.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(alice_wallet),
        ]
    )

    # Set claim type to Keep
    assert (
        await async_subtensor.staking.set_root_claim_type(
            wallet=charlie_wallet, new_root_claim_type="Keep"
        )
    ).success

    # Register SN2 and the same validator (Alice) on that subnet to ROOT has an emissions
    # - owner - Bob
    # - validator - Alice
    # - neuron, staker - Charlie (doesn't stake to validator in SN2, in ROOT only)
    sn2 = TestSubnet(async_subtensor)
    await sn2.async_execute_steps(
        [
            SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
            REGISTER_SUBNET(bob_wallet),
            SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, TEMPO_TO_SET),
            ACTIVATE_SUBNET(bob_wallet),
            REGISTER_NEURON(alice_wallet),
            REGISTER_NEURON(charlie_wallet),
        ]
    )

    # Make NumRootClaim is 5 (to have random auto claims logic)
    assert (await async_subtensor.queries.query_subtensor("NumRootClaim")).value == 5

    # Make sure CK has claim type as Keep
    assert (
        await async_subtensor.staking.get_root_claim_type(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address
        )
        == "Keep"
    )

    stake_balance = Balance.from_tao(1000)  # just a dream - stake 1000 TAO to SN0 :D

    # Stake from Charlie to Alice in ROOT
    response = await async_subtensor.staking.add_stake(
        wallet=charlie_wallet,
        netuid=root_sn.netuid,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        amount=stake_balance,
    )
    assert response.success, response.message

    proof_counter = PROOF_COUNTER

    prev_claimed_stake_charlie = await async_subtensor.staking.get_stake(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )

    prev_root_claimed_charlie = await async_subtensor.staking.get_root_claimed(
        coldkey_ss58=charlie_wallet.coldkey.ss58_address,
        hotkey_ss58=alice_wallet.hotkey.ss58_address,
        netuid=sn2.netuid,
    )

    # proof that ROOT stake increases each epoch even RootClaimType is Keep bc of random auto claim takes min 5 coldkeys
    # to do release emissions
    while proof_counter > 0:
        next_epoch_start_block = (
            await async_subtensor.subnets.get_next_epoch_start_block(root_sn.netuid)
        )
        await async_subtensor.wait_for_block(next_epoch_start_block)

        # Check Charlie stake and claimed
        claimed_stake_charlie = await async_subtensor.staking.get_stake(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert claimed_stake_charlie > prev_claimed_stake_charlie
        prev_claimed_stake_charlie = claimed_stake_charlie

        root_claimed_charlie = await async_subtensor.staking.get_root_claimed(
            coldkey_ss58=charlie_wallet.coldkey.ss58_address,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=sn2.netuid,
        )
        assert root_claimed_charlie > prev_root_claimed_charlie
        prev_root_claimed_charlie = root_claimed_charlie

        proof_counter -= 1
