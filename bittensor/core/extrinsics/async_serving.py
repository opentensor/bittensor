import asyncio
from typing import Optional, TYPE_CHECKING

from bittensor.core.errors import MetadataError
from bittensor.core.extrinsics.utils import async_submit_extrinsic
from bittensor.core.settings import version_as_int
from bittensor.utils import (
    format_error_message,
    networking as net,
    unlock_key,
    Certificate,
)
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.axon import Axon
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor.utils.substrate_interface import AsyncSubstrateInterface
    from bittensor.core.types import AxonServeCallParams
    from bittensor_wallet import Wallet


async def do_serve_axon(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    call_params: "AxonServeCallParams",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
) -> tuple[bool, Optional[dict]]:
    """
    Internal method to submit a serve axon transaction to the Bittensor blockchain. This method creates and submits a
        transaction, enabling a neuron's `Axon` to serve requests on the network.

    Args:
        subtensor: Subtensor instance object.
        wallet: The wallet associated with the neuron.
        call_params: Parameters required for the serve axon call.
        wait_for_inclusion: Waits for the transaction to be included in a block.
        wait_for_finalization: Waits for the transaction to be finalized on the blockchain.

    Returns:
        A tuple containing a success flag and an optional error message.

    This function is crucial for initializing and announcing a neuron's `Axon` service on the network, enhancing the
        decentralized computation capabilities of Bittensor.
    """

    if call_params["certificate"] is None:
        del call_params["certificate"]
        call_function = "serve_axon"
    else:
        call_function = "serve_axon_tls"

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function=call_function,
        call_params=call_params,
    )
    extrinsic = await subtensor.substrate.create_signed_extrinsic(
        call=call, keypair=wallet.hotkey
    )
    response = await async_submit_extrinsic(
        subtensor,
        extrinsic=extrinsic,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    if wait_for_inclusion or wait_for_finalization:
        if await response.is_success:
            return True, None
        else:
            return False, await response.error_message
    else:
        return True, None


async def serve_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    ip: str,
    port: int,
    protocol: int,
    netuid: int,
    placeholder1: int = 0,
    placeholder2: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization=True,
    certificate: Optional[Certificate] = None,
) -> bool:
    """Subscribes a Bittensor endpoint to the subtensor chain.

    Args:
        subtensor: Subtensor instance object.
        wallet: Bittensor wallet object.
        ip: Endpoint host port i.e., `192.122.31.4`.
        port: Endpoint port number i.e., `9221`.
        protocol: An `int` representation of the protocol.
        netuid: The network uid to serve on.
        placeholder1: A placeholder for future use.
        placeholder2: A placeholder for future use.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`,
            or returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success: Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
            finalization/inclusion, the response is `True`.
    """
    # Decrypt hotkey
    if not (unlock := unlock_key(wallet, "hotkey")).success:
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
        "certificate": certificate,
    }
    logging.debug("Checking axon ...")
    neuron = await subtensor.get_neuron_for_pubkey_and_subnet(
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
    success, error_message = await do_serve_axon(
        subtensor=subtensor,
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
            logging.error(f"Failed: {format_error_message(error_message)}")
            return False
    else:
        return True


async def serve_axon_extrinsic(
    subtensor: "AsyncSubtensor",
    netuid: int,
    axon: "Axon",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    certificate: Optional[Certificate] = None,
) -> bool:
    """
    Serves the axon to the network.

    Args:
        subtensor: Subtensor instance object.
        netuid: The `netuid` being served on.
        axon: Axon to serve.
        wait_for_inclusion: If set, waits for the extrinsic to enter a block before returning `True`, or returns
            `False` if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization: If set, waits for the extrinsic to be finalized on the chain before returning `True`,
            or returns `False` if the extrinsic fails to be finalized within the timeout.

    Returns:
        success: Flag is `True` if extrinsic was finalized or included in the block. If we did not wait for
            finalization/inclusion, the response is `True`.
    """
    if not (unlock := unlock_key(axon.wallet, "hotkey")).success:
        logging.error(unlock.message)
        return False
    external_port = axon.external_port

    # ---- Get external ip ----
    if axon.external_ip is None:
        try:
            external_ip = await asyncio.get_running_loop().run_in_executor(
                None, net.get_external_ip
            )
            logging.success(
                f":white_heavy_check_mark: [green]Found external ip:[/green] [blue]{external_ip}[/blue]"
            )
        except Exception as e:
            raise RuntimeError(
                f"Unable to attain your external ip. Check your internet connection. error: {e}"
            ) from e
    else:
        external_ip = axon.external_ip

    # ---- Subscribe to chain ----
    serve_success = await serve_extrinsic(
        subtensor=subtensor,
        wallet=axon.wallet,
        ip=external_ip,
        port=external_port,
        netuid=netuid,
        protocol=4,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        certificate=certificate,
    )
    return serve_success


# Community uses this extrinsic directly and via `subtensor.commit`
async def publish_metadata(
    subtensor: "AsyncSubtensor",
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
        subtensor: The subtensor instance representing the Bittensor blockchain connection.
        wallet: The wallet object used for authentication in the transaction.
        netuid: Network UID on which the metadata is to be published.
        data_type: The data type of the information being submitted. It should be one of the following: 'Sha256',
            'Blake256', 'Keccak256', or 'Raw0-128'. This specifies the format or hashing algorithm used for the data.
        data: The actual metadata content to be published. This should be formatted or hashed according to the `type`
            specified. (Note: max `str` length is 128 bytes)
        wait_for_inclusion: If `True`, the function will wait for the extrinsic to be included in a block before
            returning. Defaults to `False`.
        wait_for_finalization: If `True`, the function will wait for the extrinsic to be finalized on the chain before
            returning. Defaults to `True`.

    Returns:
        success: `True` if the metadata was successfully published (and finalized if specified). `False` otherwise.

    Raises:
        MetadataError: If there is an error in submitting the extrinsic or if the response from the blockchain
            indicates failure.
    """

    if not (unlock := unlock_key(wallet, "hotkey")).success:
        logging.error(unlock.message)
        return False

    substrate: "AsyncSubstrateInterface"
    async with subtensor.substrate as substrate:
        call = await substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={
                "netuid": netuid,
                "info": {"fields": [[{f"{data_type}": data}]]},
            },
        )

        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=wallet.hotkey)
        response = await async_submit_extrinsic(
            subtensor,
            extrinsic,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # We only wait here if we expect finalization.
        if not wait_for_finalization and not wait_for_inclusion:
            return True
        if await response.is_success:
            return True
        else:
            raise MetadataError(format_error_message(await response.error_message))


# Community uses this function directly
async def get_metadata(
    subtensor: "AsyncSubtensor",
    netuid: int,
    hotkey: str,
    block: Optional[int] = None,
    block_hash: Optional[str] = None,
    reuse_block: bool = False,
) -> str:
    substrate: "AsyncSubstrateInterface"
    async with subtensor.substrate as substrate:
        block_hash = await subtensor._determine_block_hash(
            block, block_hash, reuse_block
        )
        return substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )