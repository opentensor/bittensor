import random
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.core.settings import version_as_int
from bittensor.utils import format_error_message
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected
from bittensor.utils.registration import torch, use_torch
from bittensor.utils.weight_utils import (
    convert_weights_and_uids_for_emit,
    generate_weight_hash,
)

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.subtensor import Subtensor


# this will be replaced with rust-based ffi import from here https://github.com/opentensor/bittensor-commit-reveal
def get_encrypted_commit(
    commit_hash: str,
    subnet_reveal_period_epochs: int,
) -> tuple[bytes, int]:
    """
    Decrypts to t-lock bytes.

    Arguments:
        commit_hash: The hash of the commit (uids, weights) to be revealed.
        subnet_reveal_period_epochs: Number of epochs after which the revive will be performed.

    Returns:
        t-lock encrypted commit for commit_crv3_weights extrinsic.
        reveal_period: drand period when Subtensor reveal the weights to the chain.
    """
    return commit_hash.encode(), subnet_reveal_period_epochs


@ensure_connected
def _do_commit_reveal_v3(
    self: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    commit: bytes,
    reveal_round: int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    Executes the commit-reveal phase 3 for a given netuid and commit, and optionally waits for extrinsic inclusion or finalization.

    Arguments:
        wallet : Wallet An instance of the Wallet class containing the user's keypair.
        netuid : int The network unique identifier.
        commit : bytes The commit data in bytes format.
        reveal_round : int The round number for the reveal phase.
        wait_for_inclusion : bool, optional Flag indicating whether to wait for the extrinsic to be included in a block.
        wait_for_finalization : bool, optional Flag indicating whether to wait for the extrinsic to be finalized.

    Returns:
        A tuple where the first element is a boolean indicating success or failure, and the second element is an optional string containing error message if any.
    """
    logging.info(
        f"Committing weights hash [blue]{commit}[/blue] for subnet #[blue]{netuid}[/blue] with reveal round [blue]{reveal_round}[/blue]..."
    )

    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="commit_crv3_weights",
        call_params={
            "netuid": netuid,
            "commit": commit,
            "reveal_round": reveal_round,
        },
    )
    extrinsic = self.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.hotkey,
    )

    response = submit_extrinsic(
        substrate=self.substrate,
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        return True, "Not waiting for finalization or inclusion."

    response.process_events()
    if response.is_success:
        return True, None
    else:
        return False, format_error_message(
            response.error_message, substrate=self.substrate
        )


def commit_reveal_v3_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    uids: Union[NDArray[np.int64], "torch.LongTensor", list],
    weights: Union[NDArray[np.float32], "torch.FloatTensor", list],
    version_key: int = version_as_int,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str]:
    """
    Commits and reveals weights for given subtensor and wallet with provided uids and weights.

    Arguments:
        subtensor (Subtensor): The Subtensor instance.
        wallet (Wallet): The wallet to use for committing and revealing.
        netuid (int): The id of the network.
        uids (Union[NDArray[np.int64], torch.LongTensor, list]): The uids to commit.
        weights (Union[NDArray[np.float32], torch.FloatTensor, list]): The weights associated with the uids.
        version_key (int, optional): The version key to use for committing and revealing. Default is version_as_int.
        wait_for_inclusion (bool, optional): Whether to wait for the inclusion of the transaction. Default is False.
        wait_for_finalization (bool, optional): Whether to wait for the finalization of the transaction. Default is False.

    Returns:
        tuple[bool, str]: A tuple where the first element is a boolean indicating success or failure, and the second element is a message associated with the result.
    """
    try:
        # Convert uids and weights
        if use_torch():
            if isinstance(uids, list):
                uids = torch.tensor(uids, dtype=torch.int64)
            if isinstance(weights, list):
                weights = torch.tensor(weights, dtype=torch.float32)
        else:
            if isinstance(uids, list):
                uids = np.array(uids, dtype=np.int64)
            if isinstance(weights, list):
                weights = np.array(weights, dtype=np.float32)

        # Reformat and normalize.
        uids, weights = convert_weights_and_uids_for_emit(uids, weights)

        # Generate the salt
        salt = [random.randint(0, 350) for _ in range(8)]

        # Generate the hash of the weights
        commit_hash = generate_weight_hash(
            address=wallet.hotkey.ss58_address,
            netuid=netuid,
            uids=list(uids),
            values=list(weights),
            salt=salt,
            version_key=version_key,
        )

        # Get subnet's reveal (in epochs)
        subnet_reveal_period_epochs = subtensor.get_subnet_reveal_period_epochs(netuid)

        # Encrypt `commit_hash` with t-lock and `get reveal_round`
        commit_for_reveal, reveal_round = get_encrypted_commit(
            commit_hash, subnet_reveal_period_epochs
        )

        success, message = _do_commit_reveal_v3(
            self=subtensor,
            wallet=wallet,
            netuid=netuid,
            commit=commit_for_reveal,
            reveal_round=reveal_round,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success is True:
            logging.success(
                f"[green]Finalized![/green] Weights commited with reveal round [blue]{reveal_round}[/blue]."
            )
            return True, message
        else:
            logging.error(message)
            return False, message

    except Exception as e:
        logging.error(f":cross_mark: [red]Failed. Error:[/red] {e}")
        return False, str(e)
