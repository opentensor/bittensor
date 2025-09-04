import asyncio
from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.errors import MetadataError
from bittensor.core.settings import version_as_int
from bittensor.core.types import AxonServeCallParams
from bittensor.utils import (
    networking as net,
    unlock_key,
    Certificate,
)
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor.core.axon import Axon
    from bittensor.core.async_subtensor import AsyncSubtensor
    from bittensor_wallet import Wallet


async def do_serve_axon(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    call_params: "AxonServeCallParams",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Internal method to submit a serve axon transaction to the Bittensor blockchain. This method creates and submits a
        transaction, enabling a neuron's ``Axon`` to serve requests on the network.

    Args:
        subtensor (bittensor.core.async_subtensor.AsyncSubtensor): Subtensor instance object.
        wallet (bittensor_wallet.Wallet): The wallet associated with the neuron.
        call_params (bittensor.core.types.AxonServeCallParams): Parameters required for the serve axon call.
        wait_for_inclusion (bool): Waits for the transaction to be included in a block.
        wait_for_finalization (bool): Waits for the transaction to be finalized on the blockchain.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.

    Returns:
        tuple[bool, str]: A tuple containing a success flag and an optional error message.

    This function is crucial for initializing and announcing a neuron's ``Axon`` service on the network, enhancing the
        decentralized computation capabilities of Bittensor.
    """

    if call_params.certificate is None:
        call_function = "serve_axon"
    else:
        call_function = "serve_axon_tls"

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function=call_function,
        call_params=call_params.dict(),
    )
    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        sign_with="hotkey",
        period=period,
    )
    return success, message


async def serve_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    ip: str,
    port: int,
    protocol: int,
    netuid: int,
    placeholder1: int = 0,
    placeholder2: int = 0,
    certificate: Optional[Certificate] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Subscribes a Bittensor endpoint to the subtensor chain.

    Parameters:
        subtensor: Subtensor instance object.
        wallet: Bittensor wallet object.
        ip: Endpoint host port i.e., ``192.122.31.4``.
        port: Endpoint port number i.e., ``9221``.
        protocol: An ``int`` representation of the protocol.
        netuid: The network uid to serve on.
        placeholder1: A placeholder for future use.
        placeholder2: A placeholder for future use.
        certificate: Certificate to use for TLS. If ``None``, no TLS will be used.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        bool: True if the subnet registration was successful, False otherwise.
    """
    # Decrypt hotkey
    if not (unlock := unlock_key(wallet, "hotkey")).success:
        logging.error(unlock.message)
        return False

    params = AxonServeCallParams(
        **{
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
    )
    logging.debug("Checking axon ...")
    neuron = await subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid
    )
    neuron_up_to_date = not neuron.is_null and params == neuron
    if neuron_up_to_date:
        logging.debug(
            f"Axon already served on: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue]"
        )
        return True

    logging.debug(
        f"Serving axon with: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue] -> "
        f"[green]{subtensor.network}:{netuid}[/green]"
    )

    if params.certificate is None:
        call_function = "serve_axon"
    else:
        call_function = "serve_axon_tls"

    call = await subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function=call_function,
        call_params=params.dict(),
    )
    success, message = await subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        sign_with="hotkey",
        period=period,
        raise_error=raise_error,
    )

    if success:
        logging.debug(
            f"Axon served with: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue] on "
            f"[green]{subtensor.network}:{netuid}[/green]"
        )
        return True

    logging.error(f"Failed: {message}")
    return False


async def serve_axon_extrinsic(
    subtensor: "AsyncSubtensor",
    netuid: int,
    axon: "Axon",
    certificate: Optional[Certificate] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> bool:
    """
    Serves the axon to the network.

    Parameters:
        subtensor: AsyncSubtensor instance object.
        netuid (int): The ``netuid`` being served on.
        axon (bittensor.core.axon.Axon): Axon to serve.
        certificate (bittensor.utils.Certificate): Certificate to use for TLS. If ``None``, no TLS will be used.
            Defaults to ``None``.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        bool: True if the subnet registration was successful, False otherwise.
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
            raise ConnectionError(
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
        protocol=4,
        netuid=netuid,
        certificate=certificate,
        period=period,
        raise_error=raise_error,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )
    return serve_success


async def publish_metadata(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    data_type: str,
    data: Union[bytes, dict],
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    period: Optional[int] = None,
    reset_bonds: bool = False,
) -> bool:
    """
    Publishes metadata on the Bittensor network using the specified wallet and network identifier.

    Args:
        subtensor (bittensor.subtensor): The subtensor instance representing the Bittensor blockchain connection.
        wallet (bittensor.wallet): The wallet object used for authentication in the transaction.
        netuid (int): Network UID on which the metadata is to be published.
        data_type (str): The data type of the information being submitted. It should be one of the following:
            ``'Sha256'``, ``'Blake256'``, ``'Keccak256'``, or ``'Raw0-128'``. This specifies the format or hashing
            algorithm used for the data.
        data (Union[bytes, dict]): The actual metadata content to be published. This should be formatted or hashed
            according to the ``type`` specified. (Note: max ``str`` length is 128 bytes for ``'Raw0-128'``.)
        wait_for_inclusion (bool, optional): If ``True``, the function will wait for the extrinsic to be included in a
            block before returning. Defaults to ``False``.
        wait_for_finalization (bool, optional): If ``True``, the function will wait for the extrinsic to be finalized
            on the chain before returning. Defaults to ``True``.
        period (Optional[int]): The number of blocks during which the transaction will remain valid after it's submitted. If
            the transaction is not included in a block within that number of blocks, it will expire and be rejected.
            You can think of it as an expiration date for the transaction.
        reset_bonds (bool): If `True`, the function will reset the bonds for the neuron. Defaults to `False`.

    Returns:
        bool: ``True`` if the metadata was successfully published (and finalized if specified). ``False`` otherwise.

    Raises:
        MetadataError: If there is an error in submitting the extrinsic, or if the response from the blockchain indicates
            failure.
    """

    if not (unlock := unlock_key(wallet, "hotkey")).success:
        logging.error(unlock.message)
        return False

    fields = [{f"{data_type}": data}]
    if reset_bonds:
        fields.append({"ResetBondsFlag": b""})

    async with subtensor.substrate as substrate:
        call = await substrate.compose_call(
            call_module="Commitments",
            call_function="set_commitment",
            call_params={
                "netuid": netuid,
                "info": {"fields": [fields]},
            },
        )

        success, message = await subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            sign_with="hotkey",
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            period=period,
        )

        if success:
            return True
        raise MetadataError(message)


async def get_metadata(
    subtensor: "AsyncSubtensor",
    netuid: int,
    hotkey: str,
    block: Optional[int] = None,
    block_hash: Optional[str] = None,
    reuse_block: bool = False,
) -> Union[str, dict]:
    """Fetches metadata from the blockchain for a given hotkey and netuid."""
    async with subtensor.substrate:
        block_hash = await subtensor.determine_block_hash(
            block, block_hash, reuse_block
        )
        commit_data = await subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
            reuse_block_hash=reuse_block,
        )
    return commit_data


async def get_last_bonds_reset(
    subtensor: "AsyncSubtensor",
    netuid: int,
    hotkey: str,
    block: Optional[int] = None,
    block_hash: Optional[str] = None,
    reuse_block: bool = False,
) -> bytes:
    """Fetches the last bonds reset triggered at commitment from the blockchain for a given hotkey and netuid."""
    block_hash = await subtensor.determine_block_hash(block, block_hash, reuse_block)
    block = await subtensor.substrate.query(
        module="Commitments",
        storage_function="LastBondsReset",
        params=[netuid, hotkey],
        block_hash=block_hash,
    )
    return block
