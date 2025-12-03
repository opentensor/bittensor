"""Module with helper functions for extrinsics."""

import hashlib
import logging
from typing import TYPE_CHECKING, Optional, Union

from bittensor_drand import encrypt_mlkem768, mlkem_kdf_id
from scalecodec import ss58_decode

from bittensor.core.extrinsics.pallets import Sudo
from bittensor.core.types import ExtrinsicResponse
from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from bittensor_wallet import Wallet
    from bittensor.core.chain_data import StakeInfo
    from bittensor.core.subtensor import Subtensor
    from scalecodec.types import GenericCall
    from bittensor_wallet.keypair import Keypair
    from async_substrate_interface import AsyncExtrinsicReceipt, ExtrinsicReceipt


MEV_SUBMITTED_EVENT = "mevShield.EncryptedSubmitted"
MEV_EXECUTED_EVENT = "mevShield.DecryptedExecuted"
MEV_UNSUCCESSFUL_EVENTS = [
    "mevShield.DecryptedRejected",
    "mevShield.DecryptionFailed",
]

POST_SUBMIT_MEV_EVENTS = [MEV_EXECUTED_EVENT] + MEV_UNSUCCESSFUL_EVENTS


def get_old_stakes(
    wallet: "Wallet",
    hotkey_ss58s: list[str],
    netuids: list[int],
    all_stakes: list["StakeInfo"],
) -> list["Balance"]:
    """
    Retrieve the previous staking balances for a wallet's hotkeys across given netuids.

    This function searches through the provided staking data to find the stake amounts for the specified hotkeys and
    netuids associated with the wallet's coldkey. If no match is found for a particular hotkey and netuid combination,
    a default balance of zero is returned.

    Parameters:
        wallet: The wallet containing the coldkey to compare with stake data.
        hotkey_ss58s: List of hotkey SS58 addresses for which stakes are retrieved.
        netuids: List of network unique identifiers (netuids) corresponding to the hotkeys.
        all_stakes: A collection of all staking information to search through.

    Returns:
        A list of Balances, each representing the stake for a given hotkey and netuid.
    """
    stake_lookup = {
        (stake.hotkey_ss58, stake.coldkey_ss58, stake.netuid): stake.stake
        for stake in all_stakes
    }
    return [
        stake_lookup.get(
            (hotkey_ss58, wallet.coldkeypub.ss58_address, netuid),
            Balance.from_tao(0),  # Default to 0 balance if no match found
        )
        for hotkey_ss58, netuid in zip(hotkey_ss58s, netuids)
    ]


def get_transfer_fn_params(
    amount: Optional["Balance"], destination_ss58: str, keep_alive: bool
) -> tuple[str, dict[str, Union[str, int, bool]]]:
    """
    Helper function to get the transfer call function and call params, depending on the value and keep_alive flag
    provided.

    Parameters:
        amount: the amount of Tao to transfer. `None` if transferring all.
        destination_ss58: the destination SS58 of the transfer
        keep_alive: whether to enforce a retention of the existential deposit in the account after transfer.

    Returns:
        tuple[call function, call params]
    """
    call_params: dict[str, Union[str, int, bool]] = {"dest": destination_ss58}
    if amount is None:
        call_function = "transfer_all"
        if keep_alive:
            call_params["keep_alive"] = True
        else:
            call_params["keep_alive"] = False
    else:
        call_params["value"] = amount.rao
        if keep_alive:
            call_function = "transfer_keep_alive"
        else:
            call_function = "transfer_allow_death"
    return call_function, call_params


def sudo_call_extrinsic(
    subtensor: "Subtensor",
    wallet: "Wallet",
    call_function: str,
    call_params: dict,
    call_module: str = "AdminUtils",
    sign_with: str = "coldkey",
    use_nonce: bool = False,
    nonce_key: str = "hotkey",
    period: Optional[int] = None,
    raise_error: bool = False,
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = True,
    root_call: bool = False,
) -> ExtrinsicResponse:
    """Execute a sudo call extrinsic.

    Parameters:
        subtensor: The Subtensor instance.
        wallet: The wallet instance.
        call_function: The call function to execute.
        call_params: The call parameters.
        call_module: The call module.
        sign_with: The keypair to sign the extrinsic with.
        use_nonce: Whether to use a nonce.
        nonce_key: The key to use for the nonce.
        period: The number of blocks during which the transaction will remain valid after it's submitted. If the
            transaction is not included in a block within that number of blocks, it will expire and be rejected. You can
            think of it as an expiration date for the transaction.
        raise_error: Raises a relevant exception rather than returning `False` if unsuccessful.
        wait_for_inclusion: Whether to wait for the inclusion of the transaction.
        wait_for_finalization: Whether to wait for the finalization of the transaction.
        root_call: False, if the subnet owner makes a call.

    Returns:
        ExtrinsicResponse: The result object of the extrinsic execution.
    """
    try:
        if not (
            unlocked := ExtrinsicResponse.unlock_wallet(
                wallet, raise_error, unlock_type=sign_with
            )
        ).success:
            return unlocked

        call = subtensor.compose_call(
            call_module=call_module,
            call_function=call_function,
            call_params=call_params,
        )
        if not root_call:
            call = Sudo(subtensor).sudo(call)

        return subtensor.sign_and_send_extrinsic(
            call=call,
            wallet=wallet,
            sign_with=sign_with,
            use_nonce=use_nonce,
            nonce_key=nonce_key,
            period=period,
            raise_error=raise_error,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

    except Exception as error:
        return ExtrinsicResponse.from_exception(raise_error=raise_error, error=error)


def apply_pure_proxy_data(
    response: ExtrinsicResponse,
    triggered_events: list,
    block_number: int,
    extrinsic_idx: int,
    raise_error: bool,
) -> ExtrinsicResponse:
    """Apply pure proxy data to the response object.

    Parameters:
        response: The response object to update.
        triggered_events: The triggered events of the transaction.
        block_number: The block number of the transaction.
        extrinsic_idx: The index of the extrinsic in the transaction.
        raise_error: Whether to raise an error if the data cannot be applied successfully.

    Returns:
        True if the data was applied successfully, False otherwise.
    """
    # Extract pure proxy address from PureCreated triggered event if wait_for_inclusion is True.
    for event in triggered_events:
        if event.get("event_id") == "PureCreated":
            # Event structure: PureProxyCreated { disambiguation_index, proxy_type, pure, who }
            attributes = event.get("attributes", [])
            if attributes:
                response.data = {
                    "pure_account": attributes.get("pure"),
                    "spawner": attributes.get("who"),
                    "proxy_type": attributes.get("proxy_type"),
                    "index": attributes.get("disambiguation_index"),
                    "height": block_number,
                    "ext_index": extrinsic_idx,
                }
            return response

    # If triggered events are not available or event PureCreated does not exist in the response, return the response
    # with warning message ot raise the error if raise_error is True.
    message = (
        f"The ExtrinsicResponse doesn't contain pure_proxy data (`pure_account`, `spawner`, `proxy_type`, etc.) "
        f"because the extrinsic receipt doesn't have triggered events. This typically happens when "
        f"`wait_for_inclusion=False` or when `block_hash` is not available. To get this data, either pass "
        f"`wait_for_inclusion=True` when calling this function, or retrieve the data manually from the blockchain "
        f"using the extrinsic hash."
    )
    if response.extrinsic is not None and hasattr(response.extrinsic, "extrinsic_hash"):
        extrinsic_hash = response.extrinsic.extrinsic_hash
        hash_str = (
            extrinsic_hash.hex()
            if hasattr(extrinsic_hash, "hex")
            else str(extrinsic_hash)
        )
        message += f" Extrinsic hash: `{hash_str}`"

    response.success = False
    response.message = message

    if raise_error:
        raise RuntimeError(message)

    return response.with_log("warning")


def get_mev_commitment_and_ciphertext(
    call: "GenericCall",
    signer_keypair: "Keypair",
    genesis_hash: str,
    ml_kem_768_public_key: bytes,
) -> tuple[str, bytes, bytes, bytes]:
    """
    Builds MEV Shield payload and encrypts it using ML-KEM-768 + XChaCha20Poly1305.

    This function constructs the payload structure required for MEV Shield encryption and performs the encryption
    process. The payload binds the transaction to a specific key epoch using the key_hash, which replaces nonce-based
    replay protection.

    Parameters:
        call: The GenericCall object representing the inner call to be encrypted and executed.
        signer_keypair: The Keypair used for signing the inner call payload. The signer's AccountId32 (32 bytes) is
            embedded in the payload_core.
        genesis_hash: The genesis block hash as a hex string (with or without "0x" prefix). Used for chain-bound
            signature domain separation.
        ml_kem_768_public_key: The ML-KEM-768 public key bytes (1184 bytes) from NextKey storage. This key is used for
            encryption and its hash binds the transaction to the key epoch.

    Returns:
        A tuple containing:
            - commitment_hex (str): Hex string of the Blake2-256 hash of payload_core (32 bytes).
            - ciphertext (bytes): Encrypted blob containing plaintext.
            - payload_core (bytes): Raw payload bytes before encryption.
            - signature (bytes): MultiSignature (64 bytes for sr25519).
    """
    # Create payload_core: signer (32B) + key_hash (32B Blake2-256 hash) + SCALE(call)
    decoded_ss58 = ss58_decode(signer_keypair.ss58_address)
    decoded_ss58_cut = (
        decoded_ss58[2:] if decoded_ss58.startswith("0x") else decoded_ss58
    )
    signer_bytes = bytes.fromhex(decoded_ss58_cut)  # 32 bytes

    # Compute key_hash = Blake2-256(NextKey_bytes)
    # This binds the transaction to the key epoch at submission time
    key_hash_bytes = hashlib.blake2b(
        ml_kem_768_public_key, digest_size=32
    ).digest()  # 32 bytes

    scale_call_bytes = bytes(call.data.data)  # SCALE encoded call
    mev_shield_version = mlkem_kdf_id()

    # Fix genesis_hash processing
    genesis_hash_clean = (
        genesis_hash[2:] if genesis_hash.startswith("0x") else genesis_hash
    )
    genesis_hash_bytes = bytes.fromhex(genesis_hash_clean)

    payload_core = signer_bytes + key_hash_bytes + scale_call_bytes

    # Sign payload: coldkey.sign(b"mev-shield:v1" + genesis_hash + payload_core)
    message_to_sign = (
        b"mev-shield:" + mev_shield_version + genesis_hash_bytes + payload_core
    )

    signature = signer_keypair.sign(message_to_sign)

    # Create plaintext: payload_core + b"\x01" + signature
    plaintext = payload_core + b"\x01" + signature

    # Getting ciphertext (encrypting plaintext using ML-KEM-768)
    ciphertext = encrypt_mlkem768(ml_kem_768_public_key, plaintext)

    # Compute commitment: blake2_256(payload_core)
    commitment_hash = hashlib.blake2b(payload_core, digest_size=32).digest()
    commitment_hex = "0x" + commitment_hash.hex()

    return commitment_hex, ciphertext, payload_core, signature


def get_event_attributes_by_event_name(events: list, event_name: str) -> Optional[dict]:
    """
    Extracts event data from triggered events by event ID.

    Searches through a list of triggered events and returns the attributes dictionary for the first event matching the
    specified event_id.

    Parameters:
        events: List of event dictionaries, typically from ExtrinsicReceipt.triggered_events. Each event should have an
            "module_id". "event_id" key and an "attributes" key.
        event_name: The events identifier to search for (e.g. "mevShield.EncryptedSubmitted", etc).

    Returns:
        The attributes dictionary of the matching event, or None if no matching event is found."""
    for event in events:
        try:
            module_id, event_id = event_name.split(".")
        except (ValueError, AttributeError):
            logging.debug(
                "Invalid event_name. Should be string as `module_id.event_id` e.g. `mevShield.EncryptedSubmitted`."
            )
            return None
        if (
            event["module_id"].lower() == module_id.lower()
            and event["event_id"].lower() == event_id.lower()
        ):
            return event
    return None


def post_process_mev_response(
    response: "ExtrinsicResponse",
    revealed_name: str,
    revealed_extrinsic: Optional["ExtrinsicReceipt | AsyncExtrinsicReceipt"],
    raise_error: bool = False,
) -> None:
    """
    Post-processes the result of a MEV Shield extrinsic submission by updating the response object based on the revealed
    extrinsic execution status.

    This function analyzes the revealed extrinsic (execute_revealed) that was found after the initial encrypted
    submission and updates the response object accordingly. It handles cases where the revealed extrinsic was not found,
    where it failed (DecryptedRejected or DecryptionFailed events), and propagates errors if requested.

    Parameters:
        response: The ExtrinsicResponse object from the initial submit_encrypted call. This object will be modified
            in-place with the revealed extrinsic receipt and updated success/error status.
        revealed_name: The name of the event found in the revealed extrinsic (e.g., "mevShield.DecryptedExecuted",
            "mevShield.DecryptedRejected", "mevShield.DecryptionFailed").
        revealed_extrinsic: The ExtrinsicReceipt or AsyncExtrinsicReceipt object for the execute_revealed transaction,
            if found. None if the revealed extrinsic was not found in the expected blocks. This receipt contains the
            triggered events and execution details.
        raise_error: If True, raises the error immediately if the response contains an error. If False, the error is
            stored in response.error but not raised. Defaults to False.

    Returns:
        None. The function modifies the response object in-place by setting:
            - response.mev_extrinsic_receipt: The revealed extrinsic receipt
            - response.success: False if revealed extrinsic not found or failed, otherwise True.
            - response.message: Error message describing the failure if failure.
            - response.error: RuntimeError with the response.message.
    """
    # add revealed extrinsic receipt to response
    response.mev_extrinsic_receipt = revealed_extrinsic

    # when main extrinsic is successful but revealed extrinsic is not found in the chain.
    if revealed_extrinsic is None:
        response.success = False
        response.message = "Result event not found in chain."
        response.error = RuntimeError(response.message)

    # when main extrinsic is successful but revealed extrinsic is not successful.
    if revealed_name in MEV_UNSUCCESSFUL_EVENTS:
        response.success = False
        response.message = (
            f"{revealed_name}: Check `mev_extrinsic_receipt` for details."
        )
        response.error = RuntimeError(response.message)

    if response.error and raise_error:
        raise response.error
