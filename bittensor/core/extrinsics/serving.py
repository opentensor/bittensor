from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.errors import MetadataError
from bittensor.core.settings import version_as_int
from bittensor.core.types import AxonServeCallParams
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils import (
    get_function_name,
    networking as net,
    unlock_key,
    Certificate,
)
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.axon import Axon
    from bittensor.core.subtensor import Subtensor


def serve_extrinsic(
    subtensor: "Subtensor",
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
) -> ExtrinsicResponse:
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
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    # Decrypt hotkey
    if not (unlock := unlock_key(wallet, "hotkey")).success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

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
    neuron = subtensor.get_neuron_for_pubkey_and_subnet(
        wallet.hotkey.ss58_address, netuid=netuid
    )
    neuron_up_to_date = not neuron.is_null and params == neuron
    if neuron_up_to_date:
        message = f"Axon already served on: AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})"
        logging.debug(f"[blue]{message}[/blue]")
        return ExtrinsicResponse(True, message, extrinsic_function=get_function_name())

    logging.debug(
        f"Serving axon with: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue] -> "
        f"[green]{subtensor.network}:{netuid}[/green]"
    )

    if params.certificate is None:
        call_function = "serve_axon"
    else:
        call_function = "serve_axon_tls"

    call = subtensor.substrate.compose_call(
        call_module="SubtensorModule",
        call_function=call_function,
        call_params=params.dict(),
    )

    response = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        sign_with="hotkey",
        period=period,
        raise_error=raise_error,
        calling_function=get_function_name(),
    )

    if response.success:
        logging.debug(
            f"Axon served with: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue] on "
            f"[green]{subtensor.network}:{netuid}[/green]"
        )
        return response

    logging.error(f"Failed: {response.message}")
    return response


def serve_axon_extrinsic(
    subtensor: "Subtensor",
    netuid: int,
    axon: "Axon",
    certificate: Optional["Certificate"] = None,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Serves the axon to the network.

    Parameters:
        subtensor: Subtensor instance object.
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
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    if not (unlock := unlock_key(axon.wallet, "hotkey")).success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    external_port = axon.external_port

    # ---- Get external ip ----
    if axon.external_ip is None:
        try:
            external_ip = net.get_external_ip()
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
    response = serve_extrinsic(
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
    return response


def publish_metadata_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    data_type: str,
    data: Union[bytes, dict],
    reset_bonds: bool = False,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
) -> ExtrinsicResponse:
    """
    Publishes metadata on the Bittensor network using the specified wallet and network identifier.

    Parameters:
        subtensor: The subtensor instance representing the Bittensor blockchain connection.
        wallet: The wallet object used for authentication in the transaction.
        netuid: Network UID on which the metadata is to be published.
        data_type: The data type of the information being submitted. It should be one of the following:
            ``'Sha256'``, ``'Blake256'``, ``'Keccak256'``, or ``'Raw0-128'``. This specifies the format or hashing
            algorithm used for the data.
        data: The actual metadata content to be published. This should be formatted or hashed
            according to the ``type`` specified. (Note: max ``str`` length is 128 bytes for ``'Raw0-128'``.)
        reset_bonds: If `True`, the function will reset the bonds for the neuron. Defaults to `False`.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    Raises:
        MetadataError: If there is an error in submitting the extrinsic, or if the response from the blockchain indicates
        failure.
    """
    if not (unlock := unlock_key(wallet, "hotkey")).success:
        logging.error(unlock.message)
        return ExtrinsicResponse(
            False, unlock.message, extrinsic_function=get_function_name()
        )

    fields = [{f"{data_type}": data}]
    if reset_bonds:
        fields.append({"ResetBondsFlag": b""})

    call = subtensor.substrate.compose_call(
        call_module="Commitments",
        call_function="set_commitment",
        call_params={
            "netuid": netuid,
            "info": {"fields": [fields]},
        },
    )

    response = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        sign_with="hotkey",
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
        period=period,
        raise_error=raise_error,
        calling_function=get_function_name(),
    )

    if response.success:
        return response
    raise MetadataError(response.message)


def get_metadata(
    subtensor: "Subtensor", netuid: int, hotkey: str, block: Optional[int] = None
) -> Union[str, dict]:
    """Fetches metadata from the blockchain for a given hotkey and netuid."""
    commit_data = subtensor.substrate.query(
        module="Commitments",
        storage_function="CommitmentOf",
        params=[netuid, hotkey],
        block_hash=subtensor.determine_block_hash(block),
    )
    return commit_data


def get_last_bonds_reset(
    subtensor: "Subtensor", netuid: int, hotkey: str, block: Optional[int] = None
) -> bytes:
    """
    Fetches the last bonds reset triggered at commitment from the blockchain for a given hotkey and netuid.

    Parameters:
        subtensor: Subtensor instance object.
        netuid: The network uid to fetch from.
        hotkey: The hotkey of the neuron for which to fetch the last bonds reset.
        block: The block number to query. If ``None``, the latest block is used.

    Returns:
        bytes: The last bonds reset data for the specified hotkey and netuid.
    """
    return subtensor.substrate.query(
        module="Commitments",
        storage_function="LastBondsReset",
        params=[netuid, hotkey],
        block_hash=subtensor.determine_block_hash(block),
    )
