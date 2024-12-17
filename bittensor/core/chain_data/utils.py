"""Chain data helper functions and data."""

from enum import Enum
from typing import Optional, Union

from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.settings import SS58_FORMAT
from bittensor.utils.balance import Balance


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
    NeuronCertificate = 11


def from_scale_encoding(
    input_: Union[list[int], bytes, "ScaleBytes"],
    type_name: "ChainDataType",
    is_vec: bool = False,
    is_option: bool = False,
) -> Optional[dict]:
    """
    Decodes input_ data from SCALE encoding based on the specified type name and modifiers.

    Args:
        input_ (Union[List[int], bytes, ScaleBytes]): The input_ data to decode.
        type_name (ChainDataType): The type of data being decoded.
        is_vec (bool): Whether the data is a vector of the specified type. Default is ``False``.
        is_option (bool): Whether the data is an optional value of the specified type. Default is ``False``.

    Returns:
        Optional[dict]: The decoded data as a dictionary, or ``None`` if the decoding fails.
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

    Args:
        input_ (Union[List[int], bytes, ScaleBytes]): The SCALE encoded input data.
        type_string (str): The type string defining the structure of the data.

    Returns:
        Optional[dict]: The decoded data as a dictionary, or ``None`` if the decoding fails.

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
    rpc_runtime_config.update_type_registry(custom_rpc_type_registry)

    obj = rpc_runtime_config.create_scale_object(type_string, data=as_scale_bytes)

    return obj.decode()


custom_rpc_type_registry = {
    "types": {
        "NeuronCertificate": {
            "type": "struct",
            "type_mapping": [
                ["certificate", "Vec<u8>"],
            ],
        },
        "axon_info": {
            "type": "struct",
            "type_mapping": [
                ["block", "u64"],
                ["version", "u32"],
                ["ip", "u128"],
                ["port", "u16"],
                ["ip_type", "u8"],
                ["protocol", "u8"],
                ["placeholder1", "u8"],
                ["placeholder2", "u8"],
            ],
        },
        "PrometheusInfo": {
            "type": "struct",
            "type_mapping": [
                ["block", "u64"],
                ["version", "u32"],
                ["ip", "u128"],
                ["port", "u16"],
                ["ip_type", "u8"],
            ],
        },
        "IPInfo": {
            "type": "struct",
            "type_mapping": [
                ["ip", "Compact<u128>"],
                ["ip_type_and_protocol", "Compact<u8>"],
            ],
        },
        "StakeInfo": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["stake", "Compact<u64>"],
            ],
        },
        "ScheduledColdkeySwapInfo": {
            "type": "struct",
            "type_mapping": [
                ["old_coldkey", "AccountId"],
                ["new_coldkey", "AccountId"],
                ["arbitration_block", "Compact<u64>"],
            ],
        },
    }
}


def decode_account_id(account_id_bytes: Union[bytes, str]) -> str:
    """
    Decodes an AccountId from bytes to a Base64 string using SS58 encoding.

    Args:
        account_id_bytes (bytes): The AccountId in bytes that needs to be decoded.

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

    Args:
        stake_data (list): A list of tuples where each tuple contains an account ID in bytes and a stake in rao.

    Returns:
        dict: A dictionary with account IDs as keys and their corresponding Balance objects as values.
    """
    decoded_stake_data = {}
    for account_id_bytes, stake_ in stake_data:
        account_id = decode_account_id(account_id_bytes)
        decoded_stake_data.update({account_id: Balance.from_rao(stake_)})
    return decoded_stake_data
