# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation
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

"""
This module provides data structures and functions for working with the Bittensor network,
including neuron and subnet information, SCALE encoding/decoding, and custom RPC type registry.
"""

import json
import typing
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, TypedDict, Union

from scalecodec.base import RuntimeConfiguration, ScaleBytes
from scalecodec.type_registry import load_type_registry_preset
from scalecodec.types import GenericCall
from scalecodec.utils.ss58 import ss58_encode

import bittensor
from .utils import networking as net, RAOPERTAO, U16_NORMALIZED_FLOAT
from .utils.balance import Balance
from .utils.registration import torch, use_torch


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
        "StakeInfo": {
            "type": "struct",
            "type_mapping": [
                ["hotkey", "AccountId"],
                ["coldkey", "AccountId"],
                ["netuid", "Compact<u16>"],
                ["stake", "Compact<u64>"],
                ["locked", "Compact<u64>"],
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
    }
}


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


def from_scale_encoding(
    input: Union[List[int], bytes, ScaleBytes],
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

    return from_scale_encoding_using_type_string(input, type_string)


def from_scale_encoding_using_type_string(
    input: Union[List[int], bytes, ScaleBytes], type_string: str
) -> Optional[Dict]:
    if isinstance(input, ScaleBytes):
        as_scale_bytes = input
    else:
        if isinstance(input, list) and all([isinstance(i, int) for i in input]):
            vec_u8 = input
            as_bytes = bytes(vec_u8)
        elif isinstance(input, bytes):
            as_bytes = input
        else:
            raise TypeError("input must be a List[int], bytes, or ScaleBytes")
        as_scale_bytes = ScaleBytes(as_bytes)
    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
    rpc_runtime_config.update_type_registry(custom_rpc_type_registry)
    obj = rpc_runtime_config.create_scale_object(type_string, data=as_scale_bytes)
    return obj.decode()


@dataclass
class DynamicInfo:
    owner: str
    netuid: int
    tempo: int
    last_step: int
    blocks_since_last_step: int
    emission: Balance
    alpha_in: Balance
    alpha_out: Balance
    tao_in: Balance
    total_locked: Balance
    owner_locked: Balance
    price: Balance
    k: float
    is_dynamic: bool
    symbol: str

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["DynamicInfo"]:
        if len(vec_u8) == 0:
            return None
        decoded = from_scale_encoding(vec_u8, ChainDataType.DynamicInfo, is_option=True)
        if decoded is None:
            return None
        return DynamicInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["DynamicInfo"]:
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.DynamicInfo, is_vec=True, is_option=True
        )
        if decoded is None:
            return []
        decoded = [DynamicInfo.fix_decoded_values(d) for d in decoded]
        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "DynamicInfo":
        netuid = int(decoded["netuid"])
        symbol = bittensor.Balance.get_unit(netuid)
        emission = Balance.from_rao(decoded["emission"]).set_unit(0)
        alpha_out = bittensor.Balance.from_rao(decoded["alpha_out"]).set_unit(netuid)
        alpha_in = bittensor.Balance.from_rao(decoded["alpha_in"]).set_unit(netuid)
        tao_in = bittensor.Balance.from_rao(decoded["tao_in"]).set_unit(0)
        total_locked = bittensor.Balance.from_rao(decoded["total_locked"]).set_unit(
            netuid
        )
        owner_locked = bittensor.Balance.from_rao(decoded["owner_locked"]).set_unit(
            netuid
        )
        price = (
            bittensor.Balance.from_tao(tao_in.tao / alpha_in.tao)
            if alpha_in.tao > 0
            else bittensor.Balance.from_tao(1)
        )
        is_dynamic = True if decoded["alpha_in"] > 0 else False
        return DynamicInfo(
            owner=ss58_encode(decoded["owner"], bittensor.__ss58_format__),
            netuid=netuid,
            tempo=decoded["tempo"],
            last_step=decoded["last_step"],
            blocks_since_last_step=decoded["blocks_since_last_step"],
            emission=emission,
            alpha_out=alpha_out,
            alpha_in=alpha_in,
            tao_in=tao_in,
            total_locked=total_locked,
            owner_locked=owner_locked,
            price=price,
            k=tao_in.rao * alpha_in.rao,
            is_dynamic=is_dynamic,
            symbol=symbol,
        )

    def tao_to_alpha(self, tao: Balance) -> Balance:
        if self.price.tao != 0:
            return Balance.from_tao(tao.tao / self.price.tao).set_unit(self.netuid)
        else:
            return Balance.from_tao(0)

    def alpha_to_tao(self, alpha: Balance) -> Balance:
        return Balance.from_tao(alpha.tao * self.price.tao)

    def tao_to_alpha_with_slippage(self, tao: Balance) -> Tuple[Balance, Balance]:
        """
        Returns an estimate of how much Alpha would a staker receive if they stake their tao
        using the current pool state

        Args:
            tao: Amount of TAO to stake.

        Returns:
            Tuple of balances where the first part is the amount of Alpha received, and the
            second part (slippage) is the difference between the estimated amount and ideal
            amount as if there was no slippage
        """
        if self.is_dynamic:
            new_tao_in = self.tao_in + tao
            if new_tao_in == 0:
                return tao, Balance.from_rao(0)
            new_alpha_in = self.k / new_tao_in

            # Amount of alpha given to the staker
            alpha_returned = Balance.from_rao(
                self.alpha_in.rao - new_alpha_in.rao
            ).set_unit(self.netuid)

            # Ideal conversion as if there is no slippage, just price
            alpha_ideal = self.tao_to_alpha(tao)

            if alpha_ideal.tao > alpha_returned.tao:
                slippage = Balance.from_tao(
                    alpha_ideal.tao - alpha_returned.tao
                ).set_unit(self.netuid)
            else:
                slippage = Balance.from_tao(0)
        else:
            alpha_returned = tao.set_unit(self.netuid)
            slippage = Balance.from_tao(0)
        return alpha_returned, slippage

    def alpha_to_tao_with_slippage(self, alpha: Balance) -> Tuple[Balance, Balance]:
        """
        Returns an estimate of how much TAO would a staker receive if they unstake their
        alpha using the current pool state

        Args:
            alpha: Amount of Alpha to stake.

        Returns:
            Tuple of balances where the first part is the amount of TAO received, and the
            second part (slippage) is the difference between the estimated amount and ideal
            amount as if there was no slippage
        """
        if self.is_dynamic:
            new_alpha_in = self.alpha_in + alpha
            new_tao_reserve = self.k / new_alpha_in
            # Amount of TAO given to the unstaker
            tao_returned = Balance.from_rao(self.tao_in - new_tao_reserve)

            # Ideal conversion as if there is no slippage, just price
            tao_ideal = self.alpha_to_tao(alpha)

            if tao_ideal > tao_returned:
                slippage = Balance.from_tao(tao_ideal.tao - tao_returned.tao)
            else:
                slippage = Balance.from_tao(0)
        else:
            tao_returned = alpha.set_unit(0)
            slippage = Balance.from_tao(0)
        return tao_returned, slippage


@dataclass
class DynamicPool:
    is_dynamic: bool
    alpha_issuance: Balance
    alpha_outstanding: Balance
    alpha_reserve: Balance
    tao_reserve: Balance
    k: int
    price: Balance
    netuid: int

    def __init__(
        self,
        is_dynamic: bool,
        netuid: int,
        alpha_issuance: typing.Union[int, Balance],
        alpha_outstanding: typing.Union[int, Balance],
        alpha_reserve: typing.Union[int, Balance],
        tao_reserve: typing.Union[int, Balance],
        k: int,
    ):
        self.is_dynamic = is_dynamic
        self.netuid = netuid
        self.alpha_issuance = (
            alpha_issuance
            if isinstance(alpha_issuance, Balance)
            else Balance.from_rao(alpha_issuance).set_unit(netuid)
        )
        self.alpha_outstanding = (
            alpha_outstanding
            if isinstance(alpha_outstanding, Balance)
            else Balance.from_rao(alpha_outstanding).set_unit(netuid)
        )
        self.alpha_reserve = (
            alpha_reserve
            if isinstance(alpha_reserve, Balance)
            else Balance.from_rao(alpha_reserve).set_unit(netuid)
        )
        self.tao_reserve = (
            tao_reserve
            if isinstance(tao_reserve, Balance)
            else Balance.from_rao(tao_reserve).set_unit(0)
        )
        self.k = k
        if is_dynamic:
            if self.alpha_reserve.tao > 0:
                self.price = Balance.from_tao(
                    self.tao_reserve.tao / self.alpha_reserve.tao
                )
            else:
                self.price = Balance.from_tao(0.0)
        else:
            self.price = Balance.from_tao(1.0)

    def __str__(self) -> str:
        return f"DynamicPool( alpha_issuance={self.alpha_issuance}, alpha_outstanding={self.alpha_outstanding}, alpha_reserve={self.alpha_reserve}, tao_reserve={self.tao_reserve}, k={self.k}, price={self.price} )"

    def __repr__(self) -> str:
        return self.__str__()

    def tao_to_alpha(self, tao: Balance) -> Balance:
        if self.price.tao != 0:
            return Balance.from_tao(tao.tao / self.price.tao).set_unit(self.netuid)
        else:
            return Balance.from_tao(0)

    def alpha_to_tao(self, alpha: Balance) -> Balance:
        return Balance.from_tao(alpha.tao * self.price.tao)

    def tao_to_alpha_with_slippage(self, tao: Balance) -> Tuple[Balance, Balance]:
        """
        Returns an estimate of how much Alpha would a staker receive if they stake their tao
        using the current pool state

        Args:
            tao: Amount of TAO to stake.

        Returns:
            Tuple of balances where the first part is the amount of Alpha received, and the
            second part (slippage) is the difference between the estimated amount and ideal
            amount as if there was no slippage
        """
        if self.is_dynamic:
            new_tao_in = self.tao_reserve + tao
            if new_tao_in == 0:
                return tao, Balance.from_rao(0)
            new_alpha_in = self.k / new_tao_in

            # Amount of alpha given to the staker
            alpha_returned = Balance.from_rao(
                self.alpha_reserve.rao - new_alpha_in.rao
            ).set_unit(self.netuid)

            # Ideal conversion as if there is no slippage, just price
            alpha_ideal = self.tao_to_alpha(tao)

            if alpha_ideal.tao > alpha_returned.tao:
                slippage = Balance.from_tao(
                    alpha_ideal.tao - alpha_returned.tao
                ).set_unit(self.netuid)
            else:
                slippage = Balance.from_tao(0)
        else:
            alpha_returned = tao.set_unit(self.netuid)
            slippage = Balance.from_tao(0)
        return alpha_returned, slippage

    def alpha_to_tao_with_slippage(self, alpha: Balance) -> Tuple[Balance, Balance]:
        """
        Returns an estimate of how much TAO would a staker receive if they unstake their
        alpha using the current pool state

        Args:
            alpha: Amount of Alpha to stake.

        Returns:
            Tuple of balances where the first part is the amount of TAO received, and the
            second part (slippage) is the difference between the estimated amount and ideal
            amount as if there was no slippage
        """
        if self.is_dynamic:
            new_alpha_in = self.alpha_reserve + alpha
            new_tao_reserve = self.k / new_alpha_in
            # Amount of TAO given to the unstaker
            tao_returned = Balance.from_rao(self.tao_reserve - new_tao_reserve)

            # Ideal conversion as if there is no slippage, just price
            tao_ideal = self.alpha_to_tao(alpha)

            if tao_ideal > tao_returned:
                slippage = Balance.from_tao(tao_ideal.tao - tao_returned.tao)
            else:
                slippage = Balance.from_tao(0)
        else:
            tao_returned = alpha.set_unit(0)
            slippage = Balance.from_tao(0)
        return tao_returned, slippage


class SubstakeElements:
    @staticmethod
    def decode(result: List[int]) -> List[Dict]:
        descaled = from_scale_encoding(
            input=result, type_name=ChainDataType.SubstakeElements, is_vec=True
        )
        result = []
        for item in descaled:
            result.append(
                {
                    "hotkey": ss58_encode(item["hotkey"], bittensor.__ss58_format__),
                    "coldkey": ss58_encode(item["coldkey"], bittensor.__ss58_format__),
                    "netuid": item["netuid"],
                    "stake": Balance.from_rao(item["stake"]),
                }
            )
        return result


@dataclass
class AxonInfo:
    version: int
    ip: str
    port: int
    ip_type: int
    hotkey: str
    coldkey: str
    protocol: int = 4
    placeholder1: int = 0
    placeholder2: int = 0

    @property
    def is_serving(self) -> bool:
        """True if the endpoint is serving."""
        return self.ip != "0.0.0.0"

    def ip_str(self) -> str:
        """Return the whole IP as string"""
        return net.ip__str__(self.ip_type, self.ip, self.port)

    def __eq__(self, other: "AxonInfo"):
        if other is None:
            return False

        if (
            self.version == other.version
            and self.ip == other.ip
            and self.port == other.port
            and self.ip_type == other.ip_type
            and self.coldkey == other.coldkey
            and self.hotkey == other.hotkey
        ):
            return True

        return False

    def __str__(self):
        return "AxonInfo( {}, {}, {}, {} )".format(
            str(self.ip_str()), str(self.hotkey), str(self.coldkey), self.version
        )

    def __repr__(self):
        return self.__str__()

    def to_string(self) -> str:
        """Converts the AxonInfo object to a string representation using JSON."""
        try:
            return json.dumps(asdict(self))
        except (TypeError, ValueError) as e:
            bittensor.logging.error(f"Error converting AxonInfo to string: {e}")
            return AxonInfo(0, "", 0, 0, "", "").to_string()

    @classmethod
    def from_string(cls, json_string: str) -> "AxonInfo":
        """
        Creates an AxonInfo object from its string representation using JSON.

        Args:
            json_string (str): The JSON string representation of the AxonInfo object.

        Returns:
            AxonInfo: An instance of AxonInfo created from the JSON string. If decoding fails, returns a default AxonInfo object with default values.

        Raises:
            json.JSONDecodeError: If there is an error in decoding the JSON string.
            TypeError: If there is a type error when creating the AxonInfo object.
            ValueError: If there is a value error when creating the AxonInfo object.
        """
        try:
            data = json.loads(json_string)
            return cls(**data)
        except json.JSONDecodeError as e:
            bittensor.logging.error(f"Error decoding JSON: {e}")
        except TypeError as e:
            bittensor.logging.error(f"Type error: {e}")
        except ValueError as e:
            bittensor.logging.error(f"Value error: {e}")
        return AxonInfo(0, "", 0, 0, "", "")

    @classmethod
    def from_neuron_info(cls, neuron_info: dict) -> "AxonInfo":
        """
        Converts a dictionary to an AxonInfo object.

        Args:
            neuron_info (dict): A dictionary containing the neuron information.

        Returns:
            instance (AxonInfo): An instance of AxonInfo created from the dictionary.
        """
        return cls(
            version=neuron_info["axon_info"]["version"],
            ip=net.int_to_ip(int(neuron_info["axon_info"]["ip"])),
            port=neuron_info["axon_info"]["port"],
            ip_type=neuron_info["axon_info"]["ip_type"],
            hotkey=neuron_info["hotkey"],
            coldkey=neuron_info["coldkey"],
        )

    def _to_parameter_dict(
        self, return_type: str
    ) -> Union[dict[str, Union[int, str]], "torch.nn.ParameterDict"]:
        if return_type == "torch":
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    def to_parameter_dict(
        self,
    ) -> Union[dict[str, Union[int, str]], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info, depending on the USE_TORCH flag set."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "AxonInfo":
        """Returns an axon_info object from a torch parameter_dict or a parameter dict."""
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)


@dataclass
class NeuronInfo:
    """Dataclass for neuron metadata."""

    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: Balance
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: Dict[str, Balance]
    total_stake: Balance
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    validator_trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    weights: List[List[int]]
    bonds: List[List[int]]
    pruning_score: int
    prometheus_info: Optional["PrometheusInfo"] = None
    axon_info: Optional[AxonInfo] = None
    is_null: bool = False

    @classmethod
    def fix_decoded_values(cls, neuron_info_decoded: Any) -> "NeuronInfo":
        """Fixes the values of the NeuronInfo object."""
        neuron_info_decoded["hotkey"] = ss58_encode(
            neuron_info_decoded["hotkey"], bittensor.__ss58_format__
        )
        neuron_info_decoded["coldkey"] = ss58_encode(
            neuron_info_decoded["coldkey"], bittensor.__ss58_format__
        )
        stake_dict = {
            ss58_encode(coldkey, bittensor.__ss58_format__): Balance.from_rao(
                int(stake)
            )
            for coldkey, stake in neuron_info_decoded["stake"]
        }
        neuron_info_decoded["stake_dict"] = stake_dict
        neuron_info_decoded["stake"] = sum(stake_dict.values())
        neuron_info_decoded["total_stake"] = neuron_info_decoded["stake"]
        neuron_info_decoded["weights"] = [
            [int(weight[0]), int(weight[1])]
            for weight in neuron_info_decoded["weights"]
        ]
        neuron_info_decoded["bonds"] = [
            [int(bond[0]), int(bond[1])] for bond in neuron_info_decoded["bonds"]
        ]
        neuron_info_decoded["rank"] = U16_NORMALIZED_FLOAT(neuron_info_decoded["rank"])
        neuron_info_decoded["emission"] = neuron_info_decoded["emission"] / RAOPERTAO
        neuron_info_decoded["incentive"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["incentive"]
        )
        neuron_info_decoded["consensus"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["consensus"]
        )
        neuron_info_decoded["trust"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["trust"]
        )
        neuron_info_decoded["validator_trust"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["validator_trust"]
        )
        neuron_info_decoded["dividends"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["dividends"]
        )
        neuron_info_decoded["prometheus_info"] = PrometheusInfo.fix_decoded_values(
            neuron_info_decoded["prometheus_info"]
        )
        neuron_info_decoded["axon_info"] = AxonInfo.from_neuron_info(
            neuron_info_decoded
        )
        return cls(**neuron_info_decoded)

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> "NeuronInfo":
        """Returns a NeuronInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return NeuronInfo.get_null_neuron()

        decoded = from_scale_encoding(vec_u8, ChainDataType.NeuronInfo)
        if decoded is None:
            return NeuronInfo.get_null_neuron()

        return NeuronInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["NeuronInfo"]:
        """Returns a list of NeuronInfo objects from a ``vec_u8``"""

        decoded_list = from_scale_encoding(
            vec_u8, ChainDataType.NeuronInfo, is_vec=True
        )
        if decoded_list is None:
            return []

        decoded_list = [
            NeuronInfo.fix_decoded_values(decoded) for decoded in decoded_list
        ]
        return decoded_list

    @staticmethod
    def get_null_neuron() -> "NeuronInfo":
        neuron = NeuronInfo(
            uid=0,
            netuid=0,
            active=0,
            stake=Balance.from_rao(0),
            stake_dict={},
            total_stake=Balance.from_rao(0),
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            weights=[],
            bonds=[],
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )
        return neuron

    @classmethod
    def from_weights_bonds_and_neuron_lite(
        cls,
        neuron_lite: "NeuronInfoLite",
        weights_as_dict: Dict[int, List[Tuple[int, int]]],
        bonds_as_dict: Dict[int, List[Tuple[int, int]]],
    ) -> "NeuronInfo":
        n_dict = neuron_lite.__dict__
        n_dict["weights"] = weights_as_dict.get(neuron_lite.uid, [])
        n_dict["bonds"] = bonds_as_dict.get(neuron_lite.uid, [])

        return cls(**n_dict)


@dataclass
class NeuronInfoLite:
    """Dataclass for neuron metadata, but without the weights and bonds."""

    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: Balance
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: Dict[str, Balance]
    total_stake: Balance
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    validator_trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    prometheus_info: Optional["PrometheusInfo"]
    axon_info: "axon_info"
    pruning_score: int
    is_null: bool = False

    @classmethod
    def fix_decoded_values(cls, neuron_info_decoded: Any) -> "NeuronInfoLite":
        """Fixes the values of the NeuronInfoLite object."""
        neuron_info_decoded["hotkey"] = ss58_encode(
            neuron_info_decoded["hotkey"], bittensor.__ss58_format__
        )
        neuron_info_decoded["coldkey"] = ss58_encode(
            neuron_info_decoded["coldkey"], bittensor.__ss58_format__
        )
        stake_dict = {
            ss58_encode(coldkey, bittensor.__ss58_format__): Balance.from_rao(
                int(stake)
            )
            for coldkey, stake in neuron_info_decoded["stake"]
        }
        neuron_info_decoded["stake_dict"] = stake_dict
        neuron_info_decoded["stake"] = sum(stake_dict.values())
        neuron_info_decoded["total_stake"] = neuron_info_decoded["stake"]
        neuron_info_decoded["rank"] = U16_NORMALIZED_FLOAT(neuron_info_decoded["rank"])
        neuron_info_decoded["emission"] = neuron_info_decoded["emission"] / RAOPERTAO
        neuron_info_decoded["incentive"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["incentive"]
        )
        neuron_info_decoded["consensus"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["consensus"]
        )
        neuron_info_decoded["trust"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["trust"]
        )
        neuron_info_decoded["validator_trust"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["validator_trust"]
        )
        neuron_info_decoded["dividends"] = U16_NORMALIZED_FLOAT(
            neuron_info_decoded["dividends"]
        )
        neuron_info_decoded["prometheus_info"] = PrometheusInfo.fix_decoded_values(
            neuron_info_decoded["prometheus_info"]
        )
        neuron_info_decoded["axon_info"] = AxonInfo.from_neuron_info(
            neuron_info_decoded
        )
        return cls(**neuron_info_decoded)

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> "NeuronInfoLite":
        """Returns a NeuronInfoLite object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return NeuronInfoLite.get_null_neuron()

        decoded = from_scale_encoding(vec_u8, ChainDataType.NeuronInfoLite)
        if decoded is None:
            return NeuronInfoLite.get_null_neuron()

        return NeuronInfoLite.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["NeuronInfoLite"]:
        """Returns a list of NeuronInfoLite objects from a ``vec_u8``."""

        decoded_list = from_scale_encoding(
            vec_u8, ChainDataType.NeuronInfoLite, is_vec=True
        )
        if decoded_list is None:
            return []

        decoded_list = [
            NeuronInfoLite.fix_decoded_values(decoded) for decoded in decoded_list
        ]
        return decoded_list

    @staticmethod
    def get_null_neuron() -> "NeuronInfoLite":
        neuron = NeuronInfoLite(
            uid=0,
            netuid=0,
            active=0,
            stake=Balance.from_rao(0),
            stake_dict={},
            total_stake=Balance.from_rao(0),
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )
        return neuron


@dataclass
class PrometheusInfo:
    """Dataclass for prometheus info."""

    block: int
    version: int
    ip: str
    port: int
    ip_type: int

    @classmethod
    def fix_decoded_values(cls, prometheus_info_decoded: Dict) -> "PrometheusInfo":
        """Returns a PrometheusInfo object from a prometheus_info_decoded dictionary."""
        prometheus_info_decoded["ip"] = net.int_to_ip(
            int(prometheus_info_decoded["ip"])
        )

        return cls(**prometheus_info_decoded)


@dataclass
class DelegateInfoLite:
    """
    Dataclass for light delegate information.

    Args:
        hotkey_ss58 (str): Hotkey of the delegate for which the information is being fetched.
        owner_ss58 (str): Coldkey of the owner.
        total_stake (int): Total stake of the delegate.
        owner_stake (int): Own stake of the delegate.
        take (float): Take of the delegate as a percentage. None if custom
    """

    hotkey_ss58: str  # Hotkey of delegate
    owner_ss58: str  # Coldkey of owner
    take: float
    total_stake: Balance  # Total stake of the delegate
    previous_total_stake: Balance  # Total stake of the delegate
    owner_stake: Balance  # Own stake of the delegate

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "DelegateInfoLite":
        """Fixes the decoded values."""

        decoded_take = decoded["take"]
        if decoded_take == 65535:
            fixed_take = None
        else:
            fixed_take = U16_NORMALIZED_FLOAT(decoded_take)

        return cls(
            hotkey_ss58=ss58_encode(
                decoded["delegate_ss58"], bittensor.__ss58_format__
            ),
            owner_ss58=ss58_encode(decoded["owner_ss58"], bittensor.__ss58_format__),
            take=fixed_take,
            total_stake=Balance.from_rao(decoded["total_stake"]),
            owner_stake=Balance.from_rao(decoded["owner_stake"]),
            previous_total_stake=None,
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["DelegateInfoLite"]:
        r"""Returns a DelegateInfoLight object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegateInfoLight)

        if decoded is None:
            return None

        decoded = DelegateInfoLite.fix_decoded_values(decoded)

        return decoded

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["DelegateInfoLite"]:
        r"""Returns a list of DelegateInfoLight objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.DelegateInfoLight, is_vec=True
        )

        if decoded is None:
            return []

        decoded = [DelegateInfoLite.fix_decoded_values(d) for d in decoded]

        return decoded


@dataclass
class DelegateInfo:
    """
    Dataclass for delegate information. For a lighter version of this class, see :func:`DelegateInfoLite`.

    Args:
        hotkey_ss58 (str): Hotkey of the delegate for which the information is being fetched.
        total_stake (int): Total stake of the delegate.
        nominators (list[Tuple[str, int]]): List of nominators of the delegate and their stake.
        take (float): Take of the delegate as a percentage.
        owner_ss58 (str): Coldkey of the owner.
        registrations (list[int]): List of subnets that the delegate is registered on.
        validator_permits (list[int]): List of subnets that the delegate is allowed to validate on.
        return_per_1000 (int): Return per 1000 TAO, for the delegate over a day.
        total_daily_return (int): Total daily return of the delegate.

    """

    hotkey_ss58: str  # Hotkey of delegate
    total_stake: Balance  # Total stake of the delegate
    nominators: List[
        Tuple[str, Balance]
    ]  # List of nominators of the delegate and their stake
    owner_ss58: str  # Coldkey of owner
    take: List[Tuple[int, float]]  # Takes of the delegate per subnet
    validator_permits: List[
        int
    ]  # List of subnets that the delegate is allowed to validate on
    registrations: List[int]  # List of subnets that the delegate is registered on
    return_per_1000: Balance  # Return per 1000 tao of the delegate over a day
    total_daily_return: Balance  # Total daily return of the delegate

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "DelegateInfo":
        """Fixes the decoded values."""

        return cls(
            hotkey_ss58=ss58_encode(
                decoded["delegate_ss58"], bittensor.__ss58_format__
            ),
            owner_ss58=ss58_encode(decoded["owner_ss58"], bittensor.__ss58_format__),
            take=U16_NORMALIZED_FLOAT(decoded["take"]),
            nominators=[
                (
                    ss58_encode(nom[0], bittensor.__ss58_format__),
                    Balance.from_rao(nom[1]),
                )
                for nom in decoded["nominators"]
            ],
            total_stake=Balance.from_rao(
                sum([nom[1] for nom in decoded["nominators"]])
            ),
            validator_permits=decoded["validator_permits"],
            registrations=decoded["registrations"],
            return_per_1000=Balance.from_rao(decoded["return_per_1000"]),
            total_daily_return=Balance.from_rao(decoded["total_daily_return"]),
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["DelegateInfo"]:
        """Returns a DelegateInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegateInfo)
        if decoded is None:
            return None

        return DelegateInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["DelegateInfo"]:
        """Returns a list of DelegateInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegateInfo, is_vec=True)

        if decoded is None:
            return []

        return [DelegateInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def delegated_list_from_vec_u8(
        cls, vec_u8: List[int]
    ) -> List[Tuple["DelegateInfo", Balance]]:
        """Returns a list of Tuples of DelegateInfo objects, and Balance, from a ``vec_u8``.

        This is the list of delegates that the user has delegated to, and the amount of stake delegated.
        """
        decoded = from_scale_encoding(vec_u8, ChainDataType.DelegatedInfo, is_vec=True)
        if decoded is None:
            return []

        return [
            (DelegateInfo.fix_decoded_values(d), Balance.from_rao(s))
            for d, s in decoded
        ]


@dataclass
class StakeInfo:
    """Dataclass for stake info."""

    hotkey_ss58: str  # Hotkey address
    coldkey_ss58: str  # Coldkey address
    netuid: int
    stake: Balance  # Stake for the hotkey-coldkey pair
    locked: Balance  # Stake which is locked.

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "StakeInfo":
        """Fixes the decoded values."""
        return cls(
            hotkey_ss58=ss58_encode(decoded["hotkey"], bittensor.__ss58_format__),
            coldkey_ss58=ss58_encode(decoded["coldkey"], bittensor.__ss58_format__),
            netuid=int(decoded["netuid"]),
            stake=Balance.from_rao(decoded["stake"]),
            locked=Balance.from_rao(decoded["locked"]),
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["StakeInfo"]:
        """Returns a StakeInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.StakeInfo)
        if decoded is None:
            return None

        return StakeInfo.fix_decoded_values(decoded)

    @classmethod
    def list_of_tuple_from_vec_u8(
        cls, vec_u8: List[int]
    ) -> Dict[str, List["StakeInfo"]]:
        """Returns a list of StakeInfo objects from a ``vec_u8``."""
        decoded: Optional[list[tuple[str, list[object]]]] = (
            from_scale_encoding_using_type_string(
                input=vec_u8, type_string="Vec<(AccountId, Vec<StakeInfo>)>"
            )
        )

        if decoded is None:
            return {}

        return {
            ss58_encode(address=account_id, ss58_format=bittensor.__ss58_format__): [
                StakeInfo.fix_decoded_values(d) for d in stake_info
            ]
            for account_id, stake_info in decoded
        }

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["StakeInfo"]:
        """Returns a list of StakeInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(vec_u8, ChainDataType.StakeInfo, is_vec=True)
        if decoded is None:
            return []

        return [StakeInfo.fix_decoded_values(d) for d in decoded]


@dataclass
class SubnetInfo:
    """Dataclass for subnet info."""

    netuid: int
    rho: int
    kappa: int
    difficulty: int
    immunity_period: int
    max_allowed_validators: int
    min_allowed_weights: int
    max_weight_limit: float
    scaling_law_power: float
    subnetwork_n: int
    max_n: int
    blocks_since_epoch: int
    tempo: int
    modality: int
    # netuid -> topk percentile prunning score requirement (u16:MAX normalized.)
    connection_requirements: Dict[str, float]
    emission_value: float
    burn: Balance
    owner_ss58: str

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["SubnetInfo"]:
        """Returns a SubnetInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfo)
        if decoded is None:
            return None

        return SubnetInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetInfo"]:
        r"""Returns a list of SubnetInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetInfo, is_vec=True, is_option=True
        )

        if decoded is None:
            return []

        return [SubnetInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetInfo":
        """Returns a SubnetInfo object from a decoded SubnetInfo dictionary."""
        return SubnetInfo(
            netuid=decoded["netuid"],
            rho=decoded["rho"],
            kappa=decoded["kappa"],
            difficulty=decoded["difficulty"],
            immunity_period=decoded["immunity_period"],
            max_allowed_validators=decoded["max_allowed_validators"],
            min_allowed_weights=decoded["min_allowed_weights"],
            max_weight_limit=decoded["max_weights_limit"],
            # adjustment_alpha=decoded["adjustment_alpha"],
            # bonds_moving_avg=decoded["bonds_moving_average"],
            scaling_law_power=decoded["scaling_law_power"],
            subnetwork_n=decoded["subnetwork_n"],
            max_n=decoded["max_allowed_uids"],
            blocks_since_epoch=decoded["blocks_since_last_step"],
            tempo=decoded["tempo"],
            modality=decoded["network_modality"],
            connection_requirements={
                str(int(netuid)): U16_NORMALIZED_FLOAT(int(req))
                for netuid, req in decoded["network_connect"]
            },
            emission_value=decoded["emission_values"],
            burn=Balance.from_rao(decoded["burn"]),
            owner_ss58=ss58_encode(decoded["owner"], bittensor.__ss58_format__),
        )

    def to_parameter_dict(self) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "SubnetInfo":
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)


@dataclass
class SubnetHyperparameters:
    """Dataclass for subnet hyperparameters."""

    rho: int
    kappa: int
    immunity_period: int
    min_allowed_weights: int
    max_weight_limit: float
    tempo: int
    min_difficulty: int
    max_difficulty: int
    weights_version: int
    weights_rate_limit: int
    adjustment_interval: int
    activity_cutoff: int
    registration_allowed: bool
    target_regs_per_interval: int
    min_burn: int
    max_burn: int
    bonds_moving_avg: int
    max_regs_per_block: int
    serving_rate_limit: int
    max_validators: int
    adjustment_alpha: int
    difficulty: int
    commit_reveal_weights_interval: int
    commit_reveal_weights_enabled: bool
    alpha_high: int
    alpha_low: int
    liquid_alpha_enabled: bool

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["SubnetHyperparameters"]:
        """Returns a SubnetHyperparameters object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetHyperparameters)
        if decoded is None:
            return None

        return SubnetHyperparameters.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetHyperparameters"]:
        """Returns a list of SubnetHyperparameters objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetHyperparameters, is_vec=True, is_option=True
        )
        if decoded is None:
            return []

        return [SubnetHyperparameters.fix_decoded_values(d) for d in decoded]

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetHyperparameters":
        """Returns a SubnetHyperparameters object from a decoded SubnetHyperparameters dictionary."""
        return SubnetHyperparameters(
            rho=decoded["rho"],
            kappa=decoded["kappa"],
            immunity_period=decoded["immunity_period"],
            min_allowed_weights=decoded["min_allowed_weights"],
            max_weight_limit=decoded["max_weights_limit"],
            tempo=decoded["tempo"],
            min_difficulty=decoded["min_difficulty"],
            max_difficulty=decoded["max_difficulty"],
            weights_version=decoded["weights_version"],
            weights_rate_limit=decoded["weights_rate_limit"],
            adjustment_interval=decoded["adjustment_interval"],
            activity_cutoff=decoded["activity_cutoff"],
            registration_allowed=decoded["registration_allowed"],
            target_regs_per_interval=decoded["target_regs_per_interval"],
            min_burn=decoded["min_burn"],
            max_burn=decoded["max_burn"],
            max_regs_per_block=decoded["max_regs_per_block"],
            max_validators=decoded["max_validators"],
            serving_rate_limit=decoded["serving_rate_limit"],
            bonds_moving_avg=decoded["bonds_moving_avg"],
            adjustment_alpha=decoded["adjustment_alpha"],
            difficulty=decoded["difficulty"],
            commit_reveal_weights_interval=decoded["commit_reveal_weights_interval"],
            commit_reveal_weights_enabled=decoded["commit_reveal_weights_enabled"],
            alpha_high=decoded["alpha_high"],
            alpha_low=decoded["alpha_low"],
            liquid_alpha_enabled=decoded["liquid_alpha_enabled"],
        )

    def to_parameter_dict(
        self,
    ) -> Union[dict[str, Union[int, float, bool]], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet hyperparameters."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "SubnetHyperparameters":
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)


@dataclass
class DynamicPoolInfoV2:
    """Dataclass for dynamic pool info."""

    netuid: int
    alpha_issuance: int
    alpha_outstanding: int
    alpha_reserve: int
    tao_reserve: int
    k: int

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["DynamicPoolInfoV2"]:
        r"""Returns a DynamicPoolInfoV2 object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        return from_scale_encoding(vec_u8, ChainDataType.DynamicPoolInfoV2)

    def _to_parameter_dict(
        self, return_type: str
    ) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        if return_type == "torch":
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    def to_parameter_dict(self) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info."""
        if use_torch():
            return self._to_parameter_dict("torch")
        else:
            return self._to_parameter_dict("numpy")

    @classmethod
    def _from_parameter_dict_torch(
        cls, parameter_dict: "torch.nn.ParameterDict"
    ) -> "DynamicPoolInfoV2":
        """Returns a DynamicPoolInfoV2 object from a torch parameter_dict."""
        return cls(**dict(parameter_dict))

    @classmethod
    def _from_parameter_dict_numpy(
        cls, parameter_dict: dict[str, Any]
    ) -> "DynamicPoolInfoV2":
        r"""Returns a DynamicPoolInfoV2 object from a parameter_dict."""
        return cls(**parameter_dict)

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "DynamicPoolInfoV2":
        if use_torch():
            return cls._from_parameter_dict_torch(parameter_dict)
        else:
            return cls._from_parameter_dict_numpy(parameter_dict)


@dataclass
class SubnetInfoV2:
    """Dataclass for subnet info."""

    netuid: int
    owner_ss58: str
    max_allowed_validators: int
    scaling_law_power: float
    subnetwork_n: int
    max_n: int
    blocks_since_epoch: int
    modality: int
    emission_value: float
    burn: Balance
    tao_locked: Balance
    hyperparameters: SubnetHyperparameters
    dynamic_pool: DynamicPool

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["SubnetInfoV2"]:
        """Returns a SubnetInfoV2 object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.SubnetInfoV2)

        if decoded is None:
            return None

        return SubnetInfoV2.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["SubnetInfoV2"]:
        """Returns a list of SubnetInfoV2 objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.SubnetInfoV2, is_vec=True, is_option=False
        )

        if decoded is None:
            return []

        decoded = [SubnetInfoV2.fix_decoded_values(d) for d in decoded]

        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "SubnetInfoV2":
        """Returns a SubnetInfoV2 object from a decoded SubnetInfoV2 dictionary."""
        # init dynamic pool object
        pool_info = decoded["dynamic_pool"]
        if pool_info:
            pool = DynamicPool(
                True,
                pool_info["netuid"],
                pool_info["alpha_issuance"],
                pool_info["alpha_outstanding"],
                pool_info["alpha_reserve"],
                pool_info["tao_reserve"],
                pool_info["k"],
            )
        else:
            pool = DynamicPool(False, decoded["netuid"], 0, 0, 0, 0, 0)

        return SubnetInfoV2(
            netuid=decoded["netuid"],
            owner_ss58=ss58_encode(decoded["owner"], bittensor.__ss58_format__),
            max_allowed_validators=decoded["max_allowed_validators"],
            scaling_law_power=decoded["scaling_law_power"],
            subnetwork_n=decoded["subnetwork_n"],
            max_n=decoded["max_allowed_uids"],
            blocks_since_epoch=decoded["blocks_since_last_step"],
            modality=decoded["network_modality"],
            emission_value=decoded["emission_values"],
            burn=Balance.from_rao(decoded["burn"]),
            tao_locked=Balance.from_rao(decoded["tao_locked"]),
            hyperparameters=decoded["hyperparameters"],
            dynamic_pool=pool,
        )

    def _to_parameter_dict(
        self, return_type: str
    ) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        if return_type == "torch":
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    def to_parameter_dict(self) -> Union[dict[str, Any], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet info."""
        if use_torch():
            return self._to_parameter_dict("torch")
        else:
            return self._to_parameter_dict("numpy")

    @classmethod
    def _from_parameter_dict_torch(
        cls, parameter_dict: "torch.nn.ParameterDict"
    ) -> "SubnetInfoV2":
        """Returns a SubnetInfoV2 object from a torch parameter_dict."""
        return cls(**dict(parameter_dict))

    @classmethod
    def _from_parameter_dict_numpy(
        cls, parameter_dict: dict[str, Any]
    ) -> "SubnetInfoV2":
        """Returns a SubnetInfoV2 object from a parameter_dict."""
        return cls(**parameter_dict)

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "SubnetInfoV2":
        if use_torch():
            return cls._from_parameter_dict_torch(parameter_dict)
        else:
            return cls._from_parameter_dict_numpy(parameter_dict)


@dataclass
class IPInfo:
    """Dataclass for associated IP Info."""

    ip: str
    ip_type: int
    protocol: int

    def encode(self) -> Dict[str, Any]:
        """Returns a dictionary of the IPInfo object that can be encoded."""
        return {
            "ip": net.ip_to_int(
                self.ip
            ),  # IP type and protocol are encoded together as a u8
            "ip_type_and_protocol": ((self.ip_type << 4) + self.protocol) & 0xFF,
        }

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["IPInfo"]:
        """Returns a IPInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.IPInfo)
        if decoded is None:
            return None

        return IPInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["IPInfo"]:
        r"""Returns a list of IPInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(vec_u8, ChainDataType.IPInfo, is_vec=True)

        if decoded is None:
            return []

        return [IPInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def fix_decoded_values(cls, decoded: Dict) -> "IPInfo":
        """Returns a IPInfo object from a decoded IPInfo dictionary."""
        return IPInfo(
            ip=net.int_to_ip(decoded["ip"]),
            ip_type=decoded["ip_type_and_protocol"] >> 4,
            protocol=decoded["ip_type_and_protocol"] & 0xF,
        )

    def to_parameter_dict(
        self,
    ) -> Union[dict[str, Union[str, int]], "torch.nn.ParameterDict"]:
        """Returns a torch tensor or dict of the subnet IP info."""
        if use_torch():
            return torch.nn.ParameterDict(self.__dict__)
        else:
            return self.__dict__

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: Union[dict[str, Any], "torch.nn.ParameterDict"]
    ) -> "IPInfo":
        if use_torch():
            return cls(**dict(parameter_dict))
        else:
            return cls(**parameter_dict)


# Senate / Proposal data
class ProposalVoteData(TypedDict):
    index: int
    threshold: int
    ayes: List[str]
    nays: List[str]
    end: int


ProposalCallData = GenericCall


@dataclass
class ScheduledColdkeySwapInfo:
    """Dataclass for scheduled coldkey swap information."""

    old_coldkey: str
    new_coldkey: str
    arbitration_block: int

    @classmethod
    def fix_decoded_values(cls, decoded: Any) -> "ScheduledColdkeySwapInfo":
        """Fixes the decoded values."""
        return cls(
            old_coldkey=ss58_encode(decoded["old_coldkey"], bittensor.__ss58_format__),
            new_coldkey=ss58_encode(decoded["new_coldkey"], bittensor.__ss58_format__),
            arbitration_block=decoded["arbitration_block"],
        )

    @classmethod
    def from_vec_u8(cls, vec_u8: List[int]) -> Optional["ScheduledColdkeySwapInfo"]:
        """Returns a ScheduledColdkeySwapInfo object from a ``vec_u8``."""
        if len(vec_u8) == 0:
            return None

        decoded = from_scale_encoding(vec_u8, ChainDataType.ScheduledColdkeySwapInfo)
        if decoded is None:
            return None

        return ScheduledColdkeySwapInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: List[int]) -> List["ScheduledColdkeySwapInfo"]:
        """Returns a list of ScheduledColdkeySwapInfo objects from a ``vec_u8``."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.ScheduledColdkeySwapInfo, is_vec=True
        )
        if decoded is None:
            return []

        return [ScheduledColdkeySwapInfo.fix_decoded_values(d) for d in decoded]

    @classmethod
    def decode_account_id_list(cls, vec_u8: List[int]) -> Optional[List[str]]:
        """Decodes a list of AccountIds from vec_u8."""
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.ScheduledColdkeySwapInfo.AccountId, is_vec=True
        )
        if decoded is None:
            return None
        return [
            ss58_encode(account_id, bittensor.__ss58_format__) for account_id in decoded
        ]
