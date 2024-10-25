from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from substrateinterface.utils.ss58 import ss58_encode

from bittensor.chain_data.utils import ChainDataType, from_scale_encoding, SS58_FORMAT
from bittensor.utils.balance import Balance


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
        symbol = Balance.get_unit(netuid)
        emission = Balance.from_rao(decoded["emission"]).set_unit(0)
        alpha_out = Balance.from_rao(decoded["alpha_out"]).set_unit(netuid)
        alpha_in = Balance.from_rao(decoded["alpha_in"]).set_unit(netuid)
        tao_in = Balance.from_rao(decoded["tao_in"]).set_unit(0)
        total_locked = Balance.from_rao(decoded["total_locked"]).set_unit(netuid)
        owner_locked = Balance.from_rao(decoded["owner_locked"]).set_unit(netuid)
        price = (
            Balance.from_tao(tao_in.tao / alpha_in.tao)
            if alpha_in.tao > 0
            else Balance.from_tao(1)
        )
        is_dynamic = True if decoded["alpha_in"] > 0 else False
        return DynamicInfo(
            owner=ss58_encode(decoded["owner"], SS58_FORMAT),
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
        Returns an estimate of how much Alpha would a staker receive if they stake their tao using the current pool state.

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
        Returns an estimate of how much TAO would a staker receive if they unstake their alpha using the current pool state.

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
