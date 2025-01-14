"""
This module defines the `DynamicInfo` data class and associated methods for handling and decoding
dynamic information in the Bittensor network.
"""

from dataclasses import dataclass
from typing import Optional, Union

from scalecodec.utils.ss58 import ss58_encode

from bittensor.core.chain_data.utils import (
    ChainDataType,
    from_scale_encoding,
    SS58_FORMAT,
)
from bittensor.core.chain_data.subnet_identity import SubnetIdentity
from bittensor.utils.balance import Balance


@dataclass
class DynamicInfo:
    netuid: int
    owner_hotkey: str
    owner_coldkey: str
    subnet_name: str
    symbol: str
    tempo: int
    last_step: int
    blocks_since_last_step: int
    emission: Balance
    alpha_in: Balance
    alpha_out: Balance
    tao_in: Balance
    price: Balance
    k: float
    is_dynamic: bool
    alpha_out_emission: Balance
    alpha_in_emission: Balance
    tao_in_emission: Balance
    pending_alpha_emission: Balance
    pending_root_emission: Balance
    network_registered_at: int
    subnet_identity: Optional[SubnetIdentity]

    @classmethod
    def from_vec_u8(cls, vec_u8: list[int]) -> Optional["DynamicInfo"]:
        if len(vec_u8) == 0:
            return None
        decoded = from_scale_encoding(vec_u8, ChainDataType.DynamicInfo)
        if decoded is None:
            return None
        return DynamicInfo.fix_decoded_values(decoded)

    @classmethod
    def list_from_vec_u8(cls, vec_u8: Union[list[int], bytes]) -> list["DynamicInfo"]:
        decoded = from_scale_encoding(
            vec_u8, ChainDataType.DynamicInfo, is_vec=True, is_option=True
        )
        if decoded is None:
            return []
        decoded = [DynamicInfo.fix_decoded_values(d) for d in decoded]
        return decoded

    @classmethod
    def fix_decoded_values(cls, decoded: dict) -> "DynamicInfo":
        """Returns a DynamicInfo object from a decoded DynamicInfo dictionary."""

        netuid = int(decoded["netuid"])
        symbol = bytes([int(b) for b in decoded["token_symbol"]]).decode()
        subnet_name = bytes([int(b) for b in decoded["subnet_name"]]).decode()
        
        is_dynamic = (
            True if int(decoded["netuid"]) > 0 else False
        )  # Root is not dynamic

        owner_hotkey = ss58_encode(decoded["owner_hotkey"], SS58_FORMAT)
        owner_coldkey = ss58_encode(decoded["owner_coldkey"], SS58_FORMAT)

        emission = Balance.from_rao(decoded["emission"]).set_unit(0)
        alpha_in = Balance.from_rao(decoded["alpha_in"]).set_unit(netuid)
        alpha_out = Balance.from_rao(decoded["alpha_out"]).set_unit(netuid)
        tao_in = Balance.from_rao(decoded["tao_in"]).set_unit(0)
        alpha_out_emission = Balance.from_rao(decoded["alpha_out_emission"]).set_unit(
            netuid
        )
        alpha_in_emission = Balance.from_rao(decoded["alpha_in_emission"]).set_unit(
            netuid
        )
        tao_in_emission = Balance.from_rao(decoded["tao_in_emission"]).set_unit(0)
        pending_alpha_emission = Balance.from_rao(
            decoded["pending_alpha_emission"]
        ).set_unit(netuid)
        pending_root_emission = Balance.from_rao(
            decoded["pending_root_emission"]
        ).set_unit(0)

        price = (
            Balance.from_tao(1.0)
            if netuid == 0
            else Balance.from_tao(tao_in.tao / alpha_in.tao)
            if alpha_in.tao > 0
            else Balance.from_tao(1)
        )  # Root always has 1-1 price

        if decoded.get("subnet_identity"):
            subnet_identity = SubnetIdentity(
                subnet_name=decoded["subnet_identity"]["subnet_name"],
                github_repo=decoded["subnet_identity"]["github_repo"],
                subnet_contact=decoded["subnet_identity"]["subnet_contact"],
            )
        else:
            subnet_identity = None

        return cls(
            netuid=netuid,
            owner_hotkey=owner_hotkey,
            owner_coldkey=owner_coldkey,
            subnet_name=subnet_name,
            symbol=symbol,
            tempo=int(decoded["tempo"]),
            last_step=int(decoded["last_step"]),
            blocks_since_last_step=int(decoded["blocks_since_last_step"]),
            emission=emission,
            alpha_in=alpha_in,
            alpha_out=alpha_out,
            tao_in=tao_in,
            k=tao_in.rao * alpha_in.rao,
            is_dynamic=is_dynamic,
            price=price,
            alpha_out_emission=alpha_out_emission,
            alpha_in_emission=alpha_in_emission,
            tao_in_emission=tao_in_emission,
            pending_alpha_emission=pending_alpha_emission,
            pending_root_emission=pending_root_emission,
            network_registered_at=int(decoded["network_registered_at"]),
            subnet_identity=subnet_identity,
        )
