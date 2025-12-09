import asyncio
from typing import Optional, Union, TYPE_CHECKING

from bittensor.core.errors import MetadataError
from bittensor.core.extrinsics.asyncex.mev_shield import submit_encrypted_extrinsic
from bittensor.core.extrinsics.pallets import Commitments, SubtensorModule
from bittensor.core.extrinsics.utils import MEV_HOTKEY_USAGE_WARNING
from bittensor.core.settings import DEFAULT_MEV_PROTECTION, version_as_int
from bittensor.core.types import AxonServeCallParams, ExtrinsicResponse
from bittensor.utils import (
    networking as net,
    Certificate,
)
from bittensor.utils.btlogging import logging

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.axon import Axon
    from bittensor.core.async_subtensor import AsyncSubtensor


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
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
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
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type="hotkey"
            )
        ).success:
            return unlocked

        params = AxonServeCallParams(
            version=version_as_int,
            ip=net.ip_to_int(ip),
            port=port,
            ip_type=net.ip_version(ip),
            netuid=netuid,
            hotkey=wallet.hotkey.ss58_address,
            coldkey=wallet.coldkeypub.ss58_address,
            protocol=protocol,
            placeholder1=placeholder1,
            placeholder2=placeholder2,
            certificate=certificate,
        )

        logging.debug("Checking axon ...")
        neuron = await subtensor.get_neuron_for_pubkey_and_subnet(
            wallet.hotkey.ss58_address, netuid=netuid
        )
        neuron_up_to_date = not neuron.is_null and params == neuron
        if neuron_up_to_date:
            message = f"Axon already served on: AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})"
            logging.debug(f"[blue]{message}[/blue]")
            return ExtrinsicResponse(message=message)

        logging.debug(
            f"Serving axon with: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue] -> "
            f"[green]{subtensor.network}:{netuid}[/green]"
        )

        call_function = (
            SubtensorModule(subtensor).serve_axon_tls
            if certificate
            else SubtensorModule(subtensor).serve_axon
        )
        call = await call_function(**params.as_dict())

        if mev_protection:
            logging.warning(MEV_HOTKEY_USAGE_WARNING)
            response = await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                sign_with="hotkey",
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                sign_with="hotkey",
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            logging.debug(
                f"Axon served with: [blue]AxonInfo({wallet.hotkey.ss58_address}, {ip}:{port})[/blue] on "
                f"[green]{subtensor.network}:{netuid}[/green]"
            )
            return response

        logging.error(f"[red]{response.message}[/red]")
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def serve_axon_extrinsic(
    subtensor: "AsyncSubtensor",
    netuid: int,
    axon: "Axon",
    certificate: Optional[Certificate] = None,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
) -> ExtrinsicResponse:
    """
    Serves the axon to the network.

    Parameters:
        subtensor: AsyncSubtensor instance object.
        netuid: The ``netuid`` being served on.
        axon: Axon to serve.
        certificate: Certificate to use for TLS. If ``None``, no TLS will be used.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        external_port = axon.external_port

        # ---- Get external ip ----
        if axon.external_ip is None:
            try:
                external_ip = await asyncio.get_running_loop().run_in_executor(
                    None, net.get_external_ip
                )
                logging.debug(
                    f"[green]Found external ip:[/green] [blue]{external_ip}[/blue]"
                )
            except Exception as error:
                message = f"Unable to attain your external ip. Check your internet connection. Error: {error}"
                if raise_error:
                    raise ConnectionError(message) from error

                return ExtrinsicResponse(False, message).with_log()
        else:
            external_ip = axon.external_ip

        # ---- Subscribe to chain ----
        response = await serve_extrinsic(
            subtensor=subtensor,
            wallet=axon.wallet,
            ip=external_ip,
            port=external_port,
            protocol=4,
            netuid=netuid,
            certificate=certificate,
            mev_protection=mev_protection,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
            wait_for_revealed_execution=wait_for_revealed_execution,
        )
        response.data = {
            "external_ip": external_ip,
            "external_port": external_port,
            "axon": axon,
        }
        return response

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


async def publish_metadata_extrinsic(
    subtensor: "AsyncSubtensor",
    wallet: "Wallet",
    netuid: int,
    data_type: str,
    data: Union[bytes, dict],
    reset_bonds: bool = False,
    *,
    mev_protection: bool = DEFAULT_MEV_PROTECTION,
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    wait_for_revealed_execution: bool = True,
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
        reset_bonds: If `True`, the function will reset the bonds for the neuron.
        mev_protection: If True, encrypts and submits the transaction through the MEV Shield pallet to protect
            against front-running and MEV attacks. The transaction remains encrypted in the mempool until validators
            decrypt and execute it. If False, submits the transaction directly without encryption.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        wait_for_revealed_execution: Whether to wait for the revealed execution of transaction if mev_protection used.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.

    Raises:
        MetadataError: If there is an error in submitting the extrinsic, or if the response from the blockchain indicates
        failure.
    """
    try:
        signing_keypair = "hotkey"
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, signing_keypair
            )
        ).success:
            return unlocked

        fields = [{f"{data_type}": data}]
        if reset_bonds:
            fields.append({"ResetBondsFlag": b""})

        info = {"fields": [fields]}

        call = await Commitments(subtensor).set_commitment(netuid=netuid, info=info)

        if mev_protection:
            logging.warning(MEV_HOTKEY_USAGE_WARNING)
            response = await submit_encrypted_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                call=call,
                sign_with="hotkey",
                period=period,
                raise_error=raise_error,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                wait_for_revealed_execution=wait_for_revealed_execution,
            )
        else:
            response = await subtensor.sign_and_send_extrinsic(
                call=call,
                wallet=wallet,
                sign_with=signing_keypair,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                period=period,
                raise_error=raise_error,
            )

        if response.success:
            return response

        raise MetadataError(response.message)

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)
