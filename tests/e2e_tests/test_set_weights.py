import numpy as np
import pytest
import retry

from bittensor.core.extrinsics.sudo import (
    sudo_set_mechanism_count_extrinsic,
    sudo_set_admin_freeze_window_extrinsic,
)
import time
from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils.chain_interactions import (
    async_sudo_set_hyperparameter_bool,
    async_sudo_set_admin_utils,
    sudo_set_hyperparameter_bool,
    sudo_set_admin_utils,
    execute_and_wait_for_next_nonce,
)
from tests.e2e_tests.utils.e2e_test_utils import (
    async_wait_to_start_call,
    wait_to_start_call,
)


def test_set_weights_uses_next_nonce(subtensor, alice_wallet):
    """
    Tests that setting weights doesn't re-use a nonce in the transaction pool.

    Steps:
        1. Register three subnets through Alice
        2. Register Alice's neuron on each subnet and add stake
        3. Verify Alice has a vpermit on each subnet
        4. Lower the set weights rate limit on each subnet
        5. Set weights on each subnet
        6. Assert that all the set weights succeeded
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.console.info("Testing [blue]test_set_weights_uses_next_nonce[/blue]")
    # turn off admin freeze window limit for testing
    assert sudo_set_admin_freeze_window_extrinsic(
        subtensor=subtensor,
        wallet=alice_wallet,
        window=0,
    )

    netuids = [2, 3]
    TESTED_SUB_SUBNETS = 2

    # 12 for non-fast-block, 0.25 for fast block
    block_time, subnet_tempo = (
        (0.25, 50) if subtensor.chain.is_fast_blocks() else (12.0, 20)
    )

    print("Testing test_set_weights_uses_next_nonce")
    subnet_tempo = 50

    # Lower the network registration rate limit and cost
    sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_network_rate_limit",
        call_params={"rate_limit": "0"},  # No limit
    )
    # Set lock reduction interval
    sudo_set_admin_utils(
        substrate=subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_lock_reduction_interval",
        call_params={"interval": "1"},  # 1 block # reduce lock every block
    )

    for netuid in netuids:
        # Register the subnets
        assert subtensor.subnets.register_subnet(alice_wallet), (
            "Unable to register the subnet"
        )

        # Verify all subnets created successfully
        assert subtensor.subnets.subnet_exists(netuid), (
            "Subnet wasn't created successfully"
        )

        # Weights sensitive to epoch changes
        assert sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={
                "netuid": netuid,
                "tempo": subnet_tempo,
            },
        )
        assert sudo_set_mechanism_count_extrinsic(
            subtensor=subtensor,
            wallet=alice_wallet,
            netuid=netuid,
            mech_count=2,
        )

        assert wait_to_start_call(subtensor, alice_wallet, netuid)

    # Make sure 2 epochs are passed
    subtensor.wait_for_block(subnet_tempo * 2 + 1)

    # Stake to become to top neuron after the first epoch
    for netuid in netuids:
        assert subtensor.staking.add_stake(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=netuid,
            amount=Balance.from_tao(10_000),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

    # Set weight hyperparameters per subnet
    for netuid in netuids:
        assert sudo_set_hyperparameter_bool(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_commit_reveal_weights_enabled",
            value=False,
            netuid=netuid,
        ), "Unable to enable commit reveal on the subnet"

        assert not subtensor.subnets.commit_reveal_enabled(
            netuid,
        ), "Failed to enable commit/reveal"

        assert subtensor.subnets.weights_rate_limit(netuid=netuid) > 0, (
            "Weights rate limit is below 0"
        )

        # Lower set weights rate limit
        status, error = sudo_set_admin_utils(
            substrate=subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_weights_set_rate_limit",
            call_params={"netuid": netuid, "weights_set_rate_limit": "0"},
        )

        assert error is None
        assert status is True

        assert (
            subtensor.subnets.get_subnet_hyperparameters(
                netuid=netuid
            ).weights_rate_limit
            == 0
        ), "Failed to set weights_rate_limit"
        assert subtensor.subnets.get_hyperparameter("WeightsSetRateLimit", netuid) == 0
        assert subtensor.subnets.weights_rate_limit(netuid=netuid) == 0

    # Weights values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.5], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    logging.console.info(
        f"[orange]Nonce before first set_weights: "
        f"{subtensor.substrate.get_account_next_index(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    # 3 time doing call if nonce wasn't updated, then raise error
    @retry.retry(exceptions=Exception, tries=3, delay=1)
    @execute_and_wait_for_next_nonce(subtensor=subtensor, wallet=alice_wallet)
    def set_weights(netuid_, mechid_):
        success, message = subtensor.extrinsics.set_weights(
            wallet=alice_wallet,
            netuid=netuid_,
            mechid=mechid_,
            uids=weight_uids,
            weights=weight_vals,
            wait_for_inclusion=True,
            wait_for_finalization=False,
            period=subnet_tempo,
            block_time=block_time,
        )
        assert success is True, message

    logging.console.info(
        f"[orange]Nonce after second set_weights: "
        f"{subtensor.substrate.get_account_next_index(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    for mechid in range(TESTED_SUB_SUBNETS):
        # Set weights for each subnet
        for netuid in netuids:
            set_weights(netuid, mechid)

        for netuid in netuids:
            # Query the Weights storage map for all three subnets
            weights = subtensor.subnets.weights(
                netuid=netuid,
                mechid=mechid,
            )
            alice_weights = weights[0][1]
            logging.console.info(
                f"Weights for subnet mechanism {netuid}.{mechid}: {alice_weights}"
            )

            assert alice_weights is not None, (
                f"Weights not found for subnet mechanism {netuid}.{mechid}"
            )
            assert alice_weights == list(zip(weight_uids, weight_vals)), (
                f"Weights do not match for subnet {netuid}"
            )

    logging.console.info("✅ Passed [blue]test_set_weights_uses_next_nonce[/blue]")


@pytest.mark.asyncio
async def test_set_weights_uses_next_nonce_async(async_subtensor, alice_wallet):
    """
    Async tests that setting weights doesn't re-use a nonce in the transaction pool.

    Steps:
        1. Register three subnets through Alice
        2. Register Alice's neuron on each subnet and add stake
        3. Verify Alice has a vpermit on each subnet
        4. Lower the set weights rate limit on each subnet
        5. Set weights on each subnet
        6. Assert that all the set weights succeeded
    Raises:
        AssertionError: If any of the checks or verifications fail
    """
    logging.console.info("Testing [blue]test_set_weights_uses_next_nonce_async[/blue]")

    netuids = [2, 3]
    subnet_tempo = 50

    # Lower the network registration rate limit and cost
    await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_network_rate_limit",
        call_params={"rate_limit": "0"},  # No limit
    )
    # Set lock reduction interval
    await async_sudo_set_admin_utils(
        substrate=async_subtensor.substrate,
        wallet=alice_wallet,
        call_function="sudo_set_lock_reduction_interval",
        call_params={"interval": "1"},  # 1 block # reduce lock every block
    )

    for netuid in netuids:
        # Register the subnets
        assert await async_subtensor.subnets.register_subnet(alice_wallet), (
            "Unable to register the subnet"
        )

        # Verify all subnets created successfully
        assert await async_subtensor.subnets.subnet_exists(netuid), (
            "Subnet wasn't created successfully"
        )

        # Weights sensitive to epoch changes
        assert await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_tempo",
            call_params={
                "netuid": netuid,
                "tempo": subnet_tempo,
            },
        )

        assert await async_wait_to_start_call(async_subtensor, alice_wallet, netuid)

    # Make sure 2 epochs are passed
    await async_subtensor.wait_for_block(subnet_tempo * 2 + 1)

    # Stake to become to top neuron after the first epoch
    for netuid in netuids:
        assert await async_subtensor.staking.add_stake(
            wallet=alice_wallet,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            netuid=netuid,
            amount=Balance.from_tao(10_000),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

    # Set weight hyperparameters per subnet
    for netuid in netuids:
        assert await async_sudo_set_hyperparameter_bool(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_commit_reveal_weights_enabled",
            value=False,
            netuid=netuid,
        ), "Unable to enable commit reveal on the subnet"

        assert not await async_subtensor.subnets.commit_reveal_enabled(
            netuid,
        ), "Failed to enable commit/reveal"

        assert await async_subtensor.subnets.weights_rate_limit(netuid=netuid) > 0, (
            "Weights rate limit is below 0"
        )

        # Lower set weights rate limit
        status, error = await async_sudo_set_admin_utils(
            substrate=async_subtensor.substrate,
            wallet=alice_wallet,
            call_function="sudo_set_weights_set_rate_limit",
            call_params={"netuid": netuid, "weights_set_rate_limit": "0"},
        )

        assert error is None
        assert status is True

        assert (
            await async_subtensor.subnets.get_subnet_hyperparameters(netuid=netuid)
        ).weights_rate_limit == 0, "Failed to set weights_rate_limit"
        assert (
            await async_subtensor.subnets.get_hyperparameter(
                "WeightsSetRateLimit", netuid
            )
            == 0
        )
        assert await async_subtensor.subnets.weights_rate_limit(netuid=netuid) == 0

    # Weights values
    uids = np.array([0], dtype=np.int64)
    weights = np.array([0.5], dtype=np.float32)
    weight_uids, weight_vals = convert_weights_and_uids_for_emit(
        uids=uids, weights=weights
    )

    logging.console.info(
        f"[orange]Nonce before first set_weights: "
        f"{await async_subtensor.substrate.get_account_nonce(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    # # 3 time doing call if nonce wasn't updated, then raise error
    # @retry.retry(exceptions=Exception, tries=3, delay=1)
    # @execute_and_wait_for_next_nonce(subtensor=async_subtensor, wallet=alice_wallet)
    # def set_weights(netuid_):
    #     success, message = subtensor.extrinsics.set_weights(
    #         wallet=alice_wallet,
    #         netuid=netuid_,
    #         uids=weight_uids,
    #         weights=weight_vals,
    #         wait_for_inclusion=True,
    #         wait_for_finalization=False,
    #         period=subnet_tempo,
    #     )
    #     assert success is True, message

    async def set_weights(netuid_):
        """
        To avoid adding asynchronous retrieval to dependencies, we implement a retrieval behavior with asynchronous
        behavior.
        """

        async def set_weights_():
            success_, message_ = await async_subtensor.extrinsics.set_weights(
                wallet=alice_wallet,
                netuid=netuid_,
                uids=weight_uids,
                weights=weight_vals,
                wait_for_inclusion=True,
                wait_for_finalization=False,
                period=subnet_tempo,
            )
            assert success_ is True, message_

        max_retries = 3
        timeout = 60.0
        sleep = 0.25 if async_subtensor.chain.is_fast_blocks() else 12.0

        for attempt in range(1, max_retries + 1):
            try:
                start_nonce = await async_subtensor.substrate.get_account_nonce(
                    alice_wallet.hotkey.ss58_address
                )

                result = await set_weights_()

                start = time.time()
                while (time.time() - start) < timeout:
                    current_nonce = await async_subtensor.substrate.get_account_nonce(
                        alice_wallet.hotkey.ss58_address
                    )

                    if current_nonce != start_nonce:
                        logging.console.info(
                            f"✅ Nonce changed from {start_nonce} to {current_nonce}"
                        )
                        return result
                    logging.console.info(
                        f"⏳ Waiting for nonce increment. Current: {current_nonce}"
                    )
                    time.sleep(sleep)
            except Exception as e:
                raise e
        raise Exception(f"Failed to commit weights after {max_retries} attempts.")

    # Set weights for each subnet
    for netuid in netuids:
        await set_weights(netuid)

    logging.console.info(
        f"[orange]Nonce after second set_weights: "
        f"{await async_subtensor.substrate.get_account_nonce(alice_wallet.hotkey.ss58_address)}[/orange]"
    )

    for netuid in netuids:
        # Query the Weights storage map for all three subnets
        query = await async_subtensor.queries.query_module(
            module="SubtensorModule",
            name="Weights",
            params=[netuid, 0],  # Alice should be the only UID
        )

        weights = query.value
        logging.console.info(f"Weights for subnet {netuid}: {weights}")

        assert weights is not None, f"Weights not found for subnet {netuid}"
        assert weights == list(zip(weight_uids, weight_vals)), (
            f"Weights do not match for subnet {netuid}"
        )

    logging.console.info(
        "✅ Passed [blue]test_set_weights_uses_next_nonce_async[/blue]"
    )
