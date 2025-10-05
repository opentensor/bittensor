"""Chain data helper functions and data."""

from enum import Enum
from typing import Optional, Union, TYPE_CHECKING

from async_substrate_interface.types import ScaleObj
from bittensor_wallet.utils import SS58_FORMAT
from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.utils.ss58 import ss58_encode

from bittensor.utils.balance import Balance

if TYPE_CHECKING:
    from async_substrate_interface.sync_substrate import QueryMapResult


class ChainDataType(Enum):
    NeuronInfo = 1
    SubnetInfo = 2
    DelegateInfo = 3
    NeuronInfoLite = 4
    DelegatedInfo = 5
    StakeInfo = 6
    IPInfo = 7
    SubnetHyperparameters = 8
    ScheduledColdkeySwapInfo = 9
    AccountId = 10
    SubnetState = 11
    DynamicInfo = 12
    SubnetIdentity = 13
    MetagraphInfo = 14
    ChainIdentity = 15
    AxonInfo = 16


def from_scale_encoding(
    input_: Union[list[int], bytes, "ScaleBytes"],
    type_name: "ChainDataType",
    is_vec: bool = False,
    is_option: bool = False,
) -> Optional[dict]:
    """
    Decodes input_ data from SCALE encoding based on the specified type name and modifiers.

    Parameters:
        input_: The input_ data to decode.
        type_name: The type of data being decoded.
        is_vec: Whether the data is a vector of the specified type.
        is_option: Whether the data is an optional value of the specified type.

    Returns:
        The decoded data as a dictionary, or ``None`` if the decoding fails.
    """
    type_string = type_name.name
    if type_name == ChainDataType.DelegatedInfo:
        # DelegatedInfo is a tuple of (DelegateInfo, Compact<u64>)
        type_string = f"({ChainDataType.DelegateInfo.name}, Compact<u64>)"
    if is_option:
        type_string = f"Option<{type_string}>"
    if is_vec:
        type_string = f"Vec<{type_string}>"

    return from_scale_encoding_using_type_string(input_, type_string)


def from_scale_encoding_using_type_string(
    input_: Union[list[int], bytes, ScaleBytes], type_string: str
) -> Optional[dict]:
    """
    Decodes SCALE encoded data to a dictionary based on the provided type string.

    Parameters:
        input_: The SCALE encoded input data.
        type_string: The type string defining the structure of the data.

    Returns:
        The decoded data as a dictionary, or ``None`` if the decoding fails.

    Raises:
        TypeError: If the input_ is not a list[int], bytes, or ScaleBytes.
    """
    if isinstance(input_, ScaleBytes):
        as_scale_bytes = input_
    else:
        if isinstance(input_, list) and all([isinstance(i, int) for i in input_]):
            vec_u8 = input_
            as_bytes = bytes(vec_u8)
        elif isinstance(input_, bytes):
            as_bytes = input_
        else:
            raise TypeError("input_ must be a list[int], bytes, or ScaleBytes")

        as_scale_bytes = ScaleBytes(as_bytes)

    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))

    obj = rpc_runtime_config.create_scale_object(type_string, data=as_scale_bytes)

    return obj.decode()


def decode_account_id(account_id_bytes: Union[bytes, str]) -> str:
    """
    Decodes an AccountId from bytes to a Base64 string using SS58 encoding.

    Parameters:
        account_id_bytes: The AccountId in bytes that needs to be decoded.

    Returns:
        str: The decoded AccountId as a Base64 string.
    """
    if isinstance(account_id_bytes, tuple) and isinstance(account_id_bytes[0], tuple):
        account_id_bytes = account_id_bytes[0]

    # Convert the AccountId bytes to a Base64 string
    return ss58_encode(bytes(account_id_bytes).hex(), SS58_FORMAT)


def process_stake_data(stake_data: list) -> dict:
    """
    Processes stake data to decode account IDs and convert stakes from rao to Balance objects.

    Parameters:
        stake_data: A list of tuples where each tuple contains an account ID in bytes and a stake in rao.

    Returns:
        dict: A dictionary with account IDs as keys and their corresponding Balance objects as values.
    """
    decoded_stake_data = {}
    for account_id_bytes, stake_ in stake_data:
        account_id = decode_account_id(account_id_bytes)
        decoded_stake_data.update({account_id: Balance.from_rao(stake_)})
    return decoded_stake_data


def decode_metadata(metadata: dict) -> str:
    commitment = metadata["info"]["fields"][0][0]
    raw_bytes = next(iter(commitment.values()))
    byte_tuple = raw_bytes[0] if raw_bytes else raw_bytes
    return bytes(byte_tuple).decode("utf-8", errors="ignore")


def decode_block(data: bytes) -> int:
    """
    Decode the block data from the given input if it is not None.

    Parameters:
        data: The block data to decode.

    Returns:
        int: The decoded block.
    """
    return int(data.value) if isinstance(data, ScaleObj) else data


def decode_revealed_commitment(encoded_data) -> tuple[int, str]:
    """
    Decode the revealed commitment data from the given input if it is not None.

    Parameters:
        encoded_data: A tuple containing the revealed message and the block number.

    Returns:
        A tuple containing the revealed block number and decoded commitment message.
    """

    def scale_decode_offset(data: bytes) -> int:
        """Decodes the scale offset from a given byte data sequence."""
        first_byte = data[0]
        mode = first_byte & 0b11
        if mode == 0:
            return 1
        elif mode == 1:
            return 2
        else:
            return 4

    com_bytes, revealed_block = encoded_data
    offset = scale_decode_offset(com_bytes)

    revealed_commitment = bytes(com_bytes[offset:]).decode("utf-8", errors="ignore")
    return revealed_block, revealed_commitment


def decode_revealed_commitment_with_hotkey(
    encoded_data: "QueryMapResult",
) -> tuple[str, tuple[tuple[int, str], ...]]:
    """
    Decode revealed commitment using a hotkey.

    Returns:
        tuple[str, tuple[tuple[int, str], ...]]: A tuple containing the hotkey (ss58 address) and a tuple of block
            numbers and their corresponding revealed commitments.
    """
    key, data = encoded_data

    ss58_address = decode_account_id(next(iter(key)))
    block_data = tuple(decode_revealed_commitment(p) for p in data.value)
    return ss58_address, block_data
