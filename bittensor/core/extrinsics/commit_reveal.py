"""This module provides sync functionality for commit reveal in the Bittensor network."""

from typing import Union, TYPE_CHECKING, Optional

import numpy as np
from bittensor_drand import get_encrypted_commit
from numpy.typing import NDArray

from bittensor.core.settings import version_as_int
from bittensor.utils.btlogging import logging
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor
    from bittensor.utils.registration import torch


# TODO: remove in SDKv10
def commit_reveal_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    block_time: Union[int, float] = 12.0,
    commit_reveal_version: int = 4,
    version_key: int = version_as_int,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Commits and reveals weights for a given subtensor and wallet with provided uids and weights.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet to use for committing and revealing.
        netuid: The id of the network.
        uids: The uids to commit.
        weights: The weights associated with the uids.
        block_time: The number of seconds for block duration.
        commit_reveal_version: The version of the chain commit-reveal protocol to use.
        version_key: The version key to use for committing and revealing.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        Tuple[bool, str]:
            - True and a success message if the extrinsic is successfully submitted or processed.
            - False and an error message if the submission fails or the wallet cannot be unlocked.
    """
    try:
        uids, weights = convert_and_normalize_weights_and_uids(uids, weights)

        current_block = subtensor.get_current_block()
        subnet_hyperparameters = subtensor.get_subnet_hyperparameters(
            netuid, block=current_block
        )
        tempo = subnet_hyperparameters.tempo
        subnet_reveal_period_epochs = subnet_hyperparameters.commit_reveal_period

        # Encrypt `commit_hash` with t-lock and `get reveal_round`
        commit_for_reveal, reveal_round = get_encrypted_commit(
            uids=uids,
            weights=weights,
            version_key=version_key,
            tempo=tempo,
            current_block=current_block,
            netuid=netuid,
            subnet_reveal_period_epochs=subnet_reveal_period_epochs,
            block_time=block_time,
            hotkey=wallet.hotkey.public_key,
        )

        logging.info(
            f"Committing weights hash [blue]{commit_for_reveal.hex()}[/blue] for subnet #[blue]{netuid}[/blue] with "
            f"reveal round [blue]{reveal_round}[/blue]..."
        )

        call = subtensor.substrate.compose_call(
            call_module="SubtensorModule",
            call_function="commit_timelocked_weights",
            call_params={
                "netuid": netuid,
                "commit": commit_for_reveal,
                "reveal_round": reveal_round,
                "commit_reveal_version": commit_reveal_version,
            },
        )
        success, message = subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            sign_with="hotkey",
            period=period,
            raise_error=raise_error,
        )

        if not success:
            logging.error(message)
            return False, message

        logging.success(
            f"[green]Finalized![/green] Weights committed with reveal round [blue]{reveal_round}[/blue]."
        )
        return True, f"reveal_round:{reveal_round}"

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed. Error:[/red] {e}")
        return False, str(e)
