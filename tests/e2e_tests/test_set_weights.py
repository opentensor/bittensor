import time

import numpy as np
import pytest
import retry

from bittensor.utils.balance import Balance
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
from tests.e2e_tests.utils import (
    execute_and_wait_for_next_nonce,
    AdminUtils,
    TestSubnet,
    ACTIVATE_SUBNET,
    REGISTER_SUBNET,
    NETUID,
    SUDO_SET_ADMIN_FREEZE_WINDOW,
    SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED,
    SUDO_SET_LOCK_REDUCTION_INTERVAL,
    SUDO_SET_MECHANISM_COUNT,
    SUDO_SET_NETWORK_RATE_LIMIT,
    SUDO_SET_TEMPO,
    SUDO_SET_WEIGHTS_SET_RATE_LIMIT,
)

TESTED_MECHANISMS = 2
TESTED_NETUIDS = [2, 3]


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
    block_time, subnet_tempo = (
        (0.25, 50) if subtensor.chain.is_fast_blocks() else (12.0, 20)
    )

    sns = [TestSubnet(subtensor) for _ in TESTED_NETUIDS]

    hps_set_steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        SUDO_SET_NETWORK_RATE_LIMIT(alice_wallet, AdminUtils, True, 0),
        SUDO_SET_LOCK_REDUCTION_INTERVAL(alice_wallet, AdminUtils, True, 1),
    ]

    sns[0].execute_steps(hps_set_steps)

    sns_steps = [
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, subnet_tempo),
        SUDO_SET_MECHANISM_COUNT(
            alice_wallet, AdminUtils, True, NETUID, TESTED_MECHANISMS
        ),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    for sn in sns:
        sn.execute_steps(sns_steps)

    # Make sure 2 epochs are passed
    subtensor.wait_for_block(subnet_tempo * 2 + 1)

    # Stake to become to top neuron after the first epoch
    for sn in sns:
        assert subtensor.staking.add_stake(
            wallet=alice_wallet,
            netuid=sn.netuid,
            hotkey_ss58=alice_wallet.hotkey.ss58_address,
            amount=Balance.from_tao(10_000),
        ).success

    # Set weight hyperparameters per subnet
    for sn in sns:
        sn.execute_one(
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, False
            )
        )

        assert not subtensor.subnets.commit_reveal_enabled(
            netuid=sn.netuid,
        ), "Failed to enable commit/reveal"

        assert subtensor.subnets.weights_rate_limit(netuid=sn.netuid) > 0, (
            "Weights rate limit is below 0"
        )

        # Lower set weights rate limit
        response = sn.execute_one(
            SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0)
        )
        assert response.success, response.message

        assert (
            subtensor.subnets.get_subnet_hyperparameters(
                netuid=sn.netuid
            ).weights_rate_limit
            == 0
        ), "Failed to set weights_rate_limit"
        assert (
            subtensor.subnets.get_hyperparameter("WeightsSetRateLimit", sn.netuid) == 0
        )
        assert subtensor.subnets.weights_rate_limit(netuid=sn.netuid) == 0

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

    for mechid in range(TESTED_MECHANISMS):
        # Set weights for each subnet
        for sn in sns:
            set_weights(sn.netuid, mechid)

        for sn in sns:
            # Query the Weights storage map for all three subnets
            weights = subtensor.subnets.weights(
                netuid=sn.netuid,
                mechid=mechid,
            )
            alice_weights = weights[0][1]
            logging.console.info(
                f"Weights for subnet mechanism {sn.netuid}.{mechid}: {alice_weights}"
            )

            assert alice_weights is not None, (
                f"Weights not found for subnet mechanism {sn.netuid}.{mechid}"
            )
            assert alice_weights == list(zip(weight_uids, weight_vals)), (
                f"Weights do not match for subnet {sn.netuid}"
            )


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
    block_time, subnet_tempo = (
        (0.25, 50) if await async_subtensor.chain.is_fast_blocks() else (12.0, 20)
    )

    sns = [TestSubnet(async_subtensor) for _ in TESTED_NETUIDS]

    hps_set_steps = [
        SUDO_SET_ADMIN_FREEZE_WINDOW(alice_wallet, AdminUtils, True, 0),
        SUDO_SET_NETWORK_RATE_LIMIT(alice_wallet, AdminUtils, True, 0),
        SUDO_SET_LOCK_REDUCTION_INTERVAL(alice_wallet, AdminUtils, True, 1),
    ]
    await sns[0].async_execute_steps(hps_set_steps)

    sns_steps = [
        REGISTER_SUBNET(alice_wallet),
        SUDO_SET_TEMPO(alice_wallet, AdminUtils, True, NETUID, subnet_tempo),
        SUDO_SET_MECHANISM_COUNT(
            alice_wallet, AdminUtils, True, NETUID, TESTED_MECHANISMS
        ),
        ACTIVATE_SUBNET(alice_wallet),
    ]
    for sn in sns:
        await sn.async_execute_steps(sns_steps)

    # Make sure 2 epochs are passed
    await async_subtensor.wait_for_block(subnet_tempo * 2 + 1)

    # Stake to become to top neuron after the first epoch
    for sn in sns:
        assert (
            await async_subtensor.staking.add_stake(
                wallet=alice_wallet,
                netuid=sn.netuid,
                hotkey_ss58=alice_wallet.hotkey.ss58_address,
                amount=Balance.from_tao(10_000),
            )
        ).success

    # Set weight hyperparameters per subnet
    for sn in sns:
        await sn.async_execute_one(
            SUDO_SET_COMMIT_REVEAL_WEIGHTS_ENABLED(
                alice_wallet, AdminUtils, True, NETUID, False
            )
        )
        assert not await async_subtensor.subnets.commit_reveal_enabled(
            sn.netuid,
        ), "Failed to enable commit/reveal"

        assert await async_subtensor.subnets.weights_rate_limit(netuid=sn.netuid) > 0, (
            "Weights rate limit is below 0"
        )

        # Lower set weights rate limit
        response = await sn.async_execute_one(
            SUDO_SET_WEIGHTS_SET_RATE_LIMIT(alice_wallet, AdminUtils, True, NETUID, 0)
        )
        assert response.success, response.message

        assert (
            await async_subtensor.subnets.get_subnet_hyperparameters(netuid=sn.netuid)
        ).weights_rate_limit == 0, "Failed to set weights_rate_limit"
        assert (
            await async_subtensor.subnets.get_hyperparameter(
                "WeightsSetRateLimit", sn.netuid
            )
            == 0
        )
        assert await async_subtensor.subnets.weights_rate_limit(netuid=sn.netuid) == 0

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

    async def set_weights(netuid_, mechid_):
        """
        To avoid adding asynchronous retrieval to dependencies, we implement a retrieval behavior with asynchronous
        behavior.
        """

        async def set_weights_():
            success_, message_ = await async_subtensor.extrinsics.set_weights(
                wallet=alice_wallet,
                netuid=netuid_,
                mechid=mechid_,
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

    for mechid in range(TESTED_MECHANISMS):
        # Set weights for each subnet
        for sn in sns:
            await set_weights(sn.netuid, mechid)

        logging.console.info(
            f"[orange]Nonce after second set_weights: "
            f"{await async_subtensor.substrate.get_account_nonce(alice_wallet.hotkey.ss58_address)}[/orange]"
        )

        for sn in sns:
            # Query the Weights storage map for all three subnets
            weights = await async_subtensor.subnets.weights(
                netuid=sn.netuid,
                mechid=mechid,
            )
            alice_weights = weights[0][1]
            logging.console.info(
                f"Weights for subnet mechanism {sn.netuid}.{mechid}: {alice_weights}"
            )
            assert alice_weights is not None, (
                f"Weights not found for subnet mechanism {sn.netuid}.{mechid}"
            )
            assert alice_weights == list(zip(weight_uids, weight_vals)), (
                f"Weights do not match for subnet {sn.netuid}"
            )
