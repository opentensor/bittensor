from dataclasses import dataclass
from typing import Tuple, Union

from bittensor.utils.balance import Balance


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
        alpha_issuance: Union[int, Balance],
        alpha_outstanding: Union[int, Balance],
        alpha_reserve: Union[int, Balance],
        tao_reserve: Union[int, Balance],
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
        return (
            f"DynamicPool( alpha_issuance={self.alpha_issuance}, "
            f"alpha_outstanding={self.alpha_outstanding}, "
            f"alpha_reserve={self.alpha_reserve}, "
            f"tao_reserve={self.tao_reserve}, k={self.k}, price={self.price} )"
        )

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
