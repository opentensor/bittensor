# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Optional, TYPE_CHECKING

from bittensor.core.errors import MetadataError
from bittensor.core.extrinsics.utils import submit_extrinsic
from bittensor.core.settings import version_as_int
from bittensor.utils import format_error_message, networking as net, unlock_key
from bittensor.utils.btlogging import logging
from bittensor.utils.networking import ensure_connected

# For annotation purposes
if TYPE_CHECKING:
    from bittensor.core.axon import Axon
    from bittensor.core.subtensor import Subtensor
    from bittensor.core.types import AxonServeCallParams
    from bittensor_wallet import Wallet


# Chain call for `serve_extrinsic` and `serve_axon_extrinsic`
@ensure_connected
def do_serve_axon(
    self: "Subtensor",
    wallet: "Wallet",
    call_params: "AxonServeCallParams",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[dict]]:
    """
    Internal method to submit a serve axon transaction to the Bittensor blockchain. This method creates and submits a transaction, enabling a neuron's ``Axon`` to serve requests on the network.

    Args:
        self (bittensor.core.subtensor.Subtensor): Subtensor instance object.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron.
        call_params (bittensor.core.types.AxonServeCallParams): Parameters required for the serve axon call.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.

    Returns:
        tuple[bool, Optional[str]]: A tuple containing a success flag and an optional error message.

    This function is crucial for initializing and announcing a neuron's ``Axon`` service on the network, enhancing the decentralized computation capabilities of Bittensor.
    """

    call = self.substrate.compose_call(
        call_module="SubtensorModule",
        call_function="serve_axon",
        call_params=call_params,
    )
    extrinsic = self.substrate.create_signed_extrinsic(call=call, keypair=wallet.hotkey)
    response = submit_extrinsic(
        substrate=self.substrate,
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    if wait_for_inclusion or wait_for_finalization:
        response.process_events()
        if response.is_success:
            return True, None
        else:
            return False, response.error_message
    else:
        return True, None


def serve_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    ip: str,
    port: int,
    protocol: int,
    netuid: int,
    placeholder1: int = 0,
    placeholder2: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization=True,
) -> bool:
    """Subscribes a Bittensor endpoint to the subtensor chain.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance object.
        wallet (bittensor_wallet.Wallet): Bittensor wallet object.
        ip (str): Endpoint host port i.e., ``192.122.31.4``.
        port (int): Endpoint port number i.e., ``9221``.
        protocol (int): An ``int`` representation of the protocol.
        netuid (int): The network uid to serve on.
        placeholder1 (int): A placeholder for future use.
        placeholder2 (int): A placeholder for future use.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    # Decrypt hotkey
    if not (unlock := unlock_key(wallet, "hot")).success:
        logging.error(unlock.message)
        return False

    params: "AxonServeCallParams" = {
        "version": version_as_int,
        "ip": net.ip_to_int(ip),
        "port": port,
        "ip_type": net.ip_version(ip),
        "netuid": netuid,
        "hotkey": wallet.hotkey.ss58_address,
        "coldkey": wallet.coldkeypub.ss58_address,
        "protocol": protocol,
        "placeholder1": placeholder1,
        "placeholder2": placeholder2,
    }
    logging.debug("Checking axon ...")
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid
    )
    neuron_up_to_date = not neuron.is_null and params == {
        "version": neuron.axon_info.version,
        "ip": net.ip_to_int(neuron.axon_info.ip),
        "port": neuron.axon_info.port,
        "ip_type": neuron.axon_info.ip_type,
        "netuid": neuron.netuid,
        "hotkey": neuron.hotkey,
        "coldkey": neuron.coldkey,
        "protocol": neuron.axon_info.protocol,
        "placeholder1": neuron.axon_info.placeholder1,
        "placeholder2": neuron.axon_info.placeholder2,
    }
    output = params.copy()
    output["coldkey"] = wallet.coldkeypub.ss58_address
    output["hotkey"] = wallet.hotkey.ss58_address
    if neuron_up_to_date:
        logging.debug(
            f"Axon already served on: AxonInfo({wallet.hotkey.ss58_address},{ip}:{port}) "
        )
        return True

    logging.debug(
        f"Serving axon with: AxonInfo({wallet.hotkey.ss58_address},{ip}:{port}) -> {subtensor.network}:{netuid}"
    )
    success, error_message = do_serve_axon(
        self=subtensor,
        wallet=wallet,
        call_params=params,
        wait_for_finalization=wait_for_finalization,
        wait_for_inclusion=wait_for_inclusion,
    )

    if wait_for_inclusion or wait_for_finalization:
        if success is True:
            logging.debug(
                f"Axon served with: AxonInfo({wallet.hotkey.ss58_address},{ip}:{port}) on {subtensor.network}:{netuid} "
            )
            return True
        else:
            logging.error(
                f"Failed: {format_error_message(error_message, substrate=subtensor.substrate)}"
            )
            return False
    else:
        return True


def serve_axon_extrinsic(
    subtensor: "Subtensor",
    netuid: int,
    axon: "Axon",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    """Serves the axon to the network.

    Args:
        subtensor (bittensor.core.subtensor.Subtensor): Subtensor instance object.
        netuid (int): The ``netuid`` being served on.
        axon (bittensor.core.axon.Axon): Axon to serve.
        wait_for_inclusion (bool): If set, waits for the extrinsic to enter a block before returning ``true``, or returns ``false`` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool): If set, waits for the extrinsic to be finalized on the chain before returning ``true``, or returns ``false`` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success (bool): Flag is ``true`` if extrinsic was finalized or uncluded in the block. If we did not wait for finalization / inclusion, the response is ``true``.
    """
    if not (unlock := unlock_key(axon.wallet)).success:
        logging.error(unlock.message)
        return False
    if not (unlock := unlock_key(axon.wallet, "coldkeypub")).success:
        logging.error(unlock.message)
        return False
    external_port = axon.external_port

    # ---- Get external ip ----
    if axon.external_ip is None:
        try:
            external_ip = net.get_external_ip()
            logging.success(
                f":white_heavy_check_mark: <green>Found external ip:</green> <blue>{external_ip}</blue>"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unable to attain your external ip. Check your internet connection. error: {e}"
            ) from e
    else:
        external_ip = axon.external_ip

    # ---- Subscribe to chain ----
    serve_success = serve_extrinsic(
        subtensor=subtensor,
        wallet=axon.wallet,
        ip=external_ip,
        port=external_port,
        netuid=netuid,
        protocol=4,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    return serve_success


# Community uses this extrinsic directly and via `subtensor.commit`
@net.ensure_connected
def publish_metadata(
    self: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    data_type: str,
    data: bytes,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Publishes metadata on the Bittensor network using the specified wallet and network identifier.

    Args:
        self (bittensor.core.subtensor.Subtensor): The subtensor instance representing the Bittensor blockchain connection.
        wallet (bittensor_wallet.Wallet): The wallet object used for authentication in the transaction.
        netuid (int): Network UID on which the metadata is to be published.
        data_type (str): The data type of the information being submitted. It should be one of the following: ``'Sha256'``, ``'Blake256'``, ``'Keccak256'``, or ``'Raw0-128'``. This specifies the format or hashing algorithm used for the data.
        data (str): The actual metadata content to be published. This should be formatted or hashed according to the ``type`` specified. (Note: max ``str`` length is 128 bytes)
        wait_for_inclusion (bool): If ``True``, the function will wait for the extrinsic to be included in a block before returning. Defaults to ``False``.
        wait_for_finalization (bool): If ``True``, the function will wait for the extrinsic to be finalized on the chain before returning. Defaults to ``True``.

    Returns:
        bool: ``True`` if the metadata was successfully published (and finalized if specified). ``False`` otherwise.

    Raises:
        MetadataError: If there is an error in submitting the extrinsic or if the response from the blockchain indicates failure.
    """

    if not (unlock := unlock_key(wallet, "hot")).success:
        logging.error(unlock.message)
        return False

    with self.substrate as substrate:
        call = substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={
                "netuid": netuid,
                "info": {"fields": [[{f"{data_type}": data}]]},
            },
        )

        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.hotkey)
        response = substrate.submit_extrinsic(
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True
        response.process_events()
        if response.is_success:
            return True
        else:
            raise MetadataError(
                format_error_message(response.error_message, substrate=self.substrate)
            )


# Community uses this function directly
@net.ensure_connected
def get_metadata(self, netuid: int, hotkey: str, block: Optional[int] = None) -> str:
    with self.substrate as substrate:
        return substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=None if block is None else substrate.get_block_hash(block),
        ).value
