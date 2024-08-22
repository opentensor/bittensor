"""Bittensor.chain_data.utils module provides helper functions for interacting with the described chain data types."""

from enum import Enum
from typing import Dict, List, Optional, Union

from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset

SS58_FORMAT = 42


class ChainDataType(Enum):
    NeuronInfo = 1
    SubnetInfoV2 = 2
    DelegateInfo = 3
    NeuronInfoLite = 4
    DelegatedInfo = 5
    StakeInfo = 6
    IPInfo = 7
    SubnetHyperparameters = 8
    SubstakeElements = 9
    DynamicPoolInfoV2 = 10
    DelegateInfoLight = 11
    DynamicInfo = 12
    ScheduledColdkeySwapInfo = 13
    SubnetInfo = 14
    SubnetState = 15


def from_scale_encoding(
    input_: Union[List[int], bytes, ScaleBytes],
    type_name: ChainDataType,
    is_vec: bool = False,
    is_option: bool = False,
) -> Optional[Dict]:
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
    input_: Union[List[int], bytes, ScaleBytes], type_string: str
) -> Optional[Dict]:
    if isinstance(input_, ScaleBytes):
        as_scale_bytes = input_
    else:
        if isinstance(input_, list) and all([isinstance(i, int) for i in input_]):
            vec_u8 = input_
            as_bytes = bytes(vec_u8)
        elif isinstance(input_, bytes):
            as_bytes = input_
        else:
            raise TypeError("input must be a List[int], bytes, or ScaleBytes")
        as_scale_bytes = ScaleBytes(as_bytes)
    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
    rpc_runtime_config.update_type_registry(custom_rpc_type_registry)
    obj = rpc_runtime_config.create_scale_object(type_string, data=as_scale_bytes)
    return obj.decode()


custom_rpc_type_registry = {
    "types": {
        "SubnetInfo": {
            "type": "struct",
            "type_mapping": [
                ["netuid", "Compact<u16>"],
                ["rho", "Compact<u16>"],
                ["kappa", "Compact<u16>"],
                ["difficulty", "Compact<u64>"],
                ["immunity_period", "Compact<u16>"],
                ["max_allowed_validators", "Compact<u16>"],
                ["min_allowed_weights", "Compact<u16>"],
                ["max_weights_limit", "Compact<u16>"],
                ["scaling_law_power", "Compact<u16>"],
                ["subnetwork_n", "Compact<u16>"],
                ["max_allowed_uids", "Compact<u16>"],
                ["blocks_since_last_step", "Compact<u64>"],
                ["tempo", "Compact<u16>"],
                ["network_modality", "Compact<u16>"],
                ["network_connect", "Vec<[u16; 2]>"],
                ["emission_values", "Compact<u64>"],
                ["burn", "Compact<u64>"],
                ["owner", "AccountId"],
            ],
        },
        "DynamicPoolInfoV2": {
            "type": "struct",
            "type_mapping": [
                ["netuid", "u16"],
                ["alpha_issuance", "u64"],
                ["alpha_outstanding", "u64"],
                ["alpha_reserve", "u64"],
                ["tao_reserve", "u64"],
                ["k", "u128"],
            ],
        },
        "SubnetInfoV2": {
            "type": "struct",
            "type_mapping": [
                ["netuid", "u16"],
                ["owner", "AccountId"],
                ["max_allowed_validators", "u16"],
                ["scaling_law_power", "u16"],
                ["subnetwork_n", "u16"],
                ["max_allowed_uids", "u16"],
                ["blocks_since_last_step", "Compact<u32>"],
                ["network_modality", "u16"],
                ["emission_values", "Compact<u64>"],
                ["burn", "Compact<u64>"],
                ["tao_locked", "Compact<u64>"],
                ["hyperparameters", "SubnetHyperparameters"],
                ["dynamic_pool", "Option<DynamicPoolInfoV2>"],
            ],
        },
        "DelegateInfo": {
            "type": "struct",
            "type_mapping": [
                ["delegate_ss58", "AccountId"],
                ["take", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["nominators", "Vec<(AccountId, Compact<u64>)>"],
                ["owner_ss58", "AccountId"],
                ["registrations", "Vec<Compact<u16>>"],
                ["validator_permits", "Vec<Compact<u16>>"],
                ["return_per_1000", "Compact<u64>"],
                ["total_daily_return", "Compact<u64>"],
            ],
        },
        "DelegateInfoLight": {
            "type": "struct",
            "type_mapping": [
                ["delegate_ss58", "AccountId"],
                ["owner_ss58", "AccountId"],
                ["take", "u16"],
                ["owner_stake", "Compact<u64>"],
                ["total_stake", "Compact<u64>"],
            ],
        },
        "NeuronInfo": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["uid", "Compact<u16>"],
                ["netuid", "Compact<u16>"],
                ["active", "bool"],
                ["axon_info", "axon_info"],
                ["prometheus_info", "PrometheusInfo"],
                ["stake", "Vec<(AccountId, Compact<u64>)>"],
                ["rank", "Compact<u16>"],
                ["emission", "Compact<u64>"],
                ["incentive", "Compact<u16>"],
                ["consensus", "Compact<u16>"],
                ["trust", "Compact<u16>"],
                ["validator_trust", "Compact<u16>"],
                ["dividends", "Compact<u16>"],
                ["last_update", "Compact<u64>"],
                ["validator_permit", "bool"],
                ["weights", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["bonds", "Vec<(Compact<u16>, Compact<u16>)>"],
                ["pruning_score", "Compact<u16>"],
            ],
        },
        "NeuronInfoLite": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["uid", "Compact<u16>"],
                ["netuid", "Compact<u16>"],
                ["active", "bool"],
                ["axon_info", "axon_info"],
                ["prometheus_info", "PrometheusInfo"],
                ["stake", "Vec<(AccountId, Compact<u64>)>"],
                ["rank", "Compact<u16>"],
                ["emission", "Compact<u64>"],
                ["incentive", "Compact<u16>"],
                ["consensus", "Compact<u16>"],
                ["trust", "Compact<u16>"],
                ["validator_trust", "Compact<u16>"],
                ["dividends", "Compact<u16>"],
                ["last_update", "Compact<u64>"],
                ["validator_permit", "bool"],
                ["pruning_score", "Compact<u16>"],
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
        "DynamicInfo": {
            "type": "struct",
            "type_mapping": [
                ["owner", "AccountId"],
                ["netuid", "Compact<u16>"],
                ["tempo", "Compact<u16>"],
                ["last_step", "Compact<u64>"],
                ["blocks_since_last_step", "Compact<u64>"],
                ["emission", "Compact<u64>"],
                ["alpha_in", "Compact<u64>"],
                ["alpha_out", "Compact<u64>"],
                ["tao_in", "Compact<u64>"],
                ["total_locked", "Compact<u64>"],
                ["owner_locked", "Compact<u64>"],
            ],
        },
        "SubstakeElements": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["netuid", "Compact<u16>"],
                ["stake", "Compact<u64>"],
            ],
        },
        "SubnetHyperparameters": {
            "type": "struct",
            "type_mapping": [
                ["rho", "Compact<u16>"],
                ["kappa", "Compact<u16>"],
                ["immunity_period", "Compact<u16>"],
                ["min_allowed_weights", "Compact<u16>"],
                ["max_weights_limit", "Compact<u16>"],
                ["tempo", "Compact<u16>"],
                ["min_difficulty", "Compact<u64>"],
                ["max_difficulty", "Compact<u64>"],
                ["weights_version", "Compact<u64>"],
                ["weights_rate_limit", "Compact<u64>"],
                ["adjustment_interval", "Compact<u16>"],
                ["activity_cutoff", "Compact<u16>"],
                ["registration_allowed", "bool"],
                ["target_regs_per_interval", "Compact<u16>"],
                ["min_burn", "Compact<u64>"],
                ["max_burn", "Compact<u64>"],
                ["bonds_moving_avg", "Compact<u64>"],
                ["max_regs_per_block", "Compact<u16>"],
                ["serving_rate_limit", "Compact<u64>"],
                ["max_validators", "Compact<u16>"],
                ["adjustment_alpha", "Compact<u64>"],
                ["difficulty", "Compact<u64>"],
                ["commit_reveal_weights_interval", "Compact<u64>"],
                ["commit_reveal_weights_enabled", "bool"],
                ["alpha_high", "Compact<u16>"],
                ["alpha_low", "Compact<u16>"],
                ["liquid_alpha_enabled", "bool"],
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
          "SubnetState": {
            "type": "struct",
            "type_mapping": [
                ["netuid", "Compact<u16>"],
                ["hotkeys", "Vec<AccountId>"],
                ["coldkeys", "Vec<AccountId>"],
                ["active", "Vec<bool>"],
                ["validator_permit", "Vec<bool>"],
                ["pruning_score", "Vec<Compact<u16>>"],
                ["last_update", "Vec<Compact<u64>>"],
                ["emission", "Vec<Compact<u64>>"],
                ["dividends", "Vec<Compact<u16>>"],
                ["incentives", "Vec<Compact<u16>>"],
                ["consensus", "Vec<Compact<u16>>"],
                ["trust", "Vec<Compact<u16>>"],
                ["rank", "Vec<Compact<u16>>"],
                ["block_at_registration", "Vec<Compact<u64>>"],
                ["local_stake", "Vec<Compact<u64>>"],
                ["global_stake", "Vec<Compact<u64>>"],
                ["stake_weight", "Vec<Compact<u16>>"],
                ["emission_history", "Vec<Vec<Compact<u64>>>"],
            ],
        },
        "StakeInfo": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["netuid", "Compact<u16>"],
                ["stake", "Compact<u64>"],
                ["locked", "Compact<u64>"],
                ["emission", "Compact<u64>"],
                ["drain", "Compact<u64>"],
                ["is_registered", "bool"],
            ],
        },
    }
}
