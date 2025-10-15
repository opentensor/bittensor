from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from bittensor.utils.balance import Balance


@dataclass
class CrowdloanParams:
    @classmethod
    def create(
        cls,
        deposit: "Balance",
        min_contribution: "Balance",
        cap: "Balance",
        end: int,
        call: Optional[str] = None,
        target_address: Optional[str] = None,
    ) -> dict:
        """Returns the parameters for the `create`."""
        return {
            "deposit": deposit.rao,
            "min_contribution": min_contribution.rao,
            "cap": cap.rao,
            "end": end,
            "call": call,
            "target_address": target_address,
        }

    @classmethod
    def contribute(
        cls,
        crowdloan_id: int,
        amount: "Balance",
    ) -> dict:
        """Returns the parameters for the `contribute`."""
        return {
            "crowdloan_id": crowdloan_id,
            "amount": amount.rao,
        }

    @classmethod
    def withdraw(
        cls,
        crowdloan_id: int,
    ) -> dict:
        """Returns the parameters for the `withdraw`."""
        return {"crowdloan_id": crowdloan_id}

    @classmethod
    def finalize(
        cls,
        crowdloan_id: int,
    ) -> dict:
        """Returns the parameters for the `finalize`."""
        return {"crowdloan_id": crowdloan_id}

    @classmethod
    def refund(
        cls,
        crowdloan_id: int,
    ) -> dict:
        """Returns the parameters for the `refund`."""
        return {"crowdloan_id": crowdloan_id}

    @classmethod
    def dissolve(
        cls,
        crowdloan_id: int,
    ) -> dict:
        """Returns the parameters for the `dissolve`."""
        return {"crowdloan_id": crowdloan_id}

    @classmethod
    def update_min_contribution(
        cls,
        crowdloan_id: int,
        new_min_contribution: "Balance",
    ) -> dict:
        """Returns the parameters for the `update_min_contribution`."""
        return {
            "crowdloan_id": crowdloan_id,
            "new_min_contribution": new_min_contribution.rao,
        }

    @classmethod
    def update_end(
        cls,
        crowdloan_id: int,
        new_end: int,
    ) -> dict:
        """Returns the parameters for the `update_end`."""
        return {
            "crowdloan_id": crowdloan_id,
            "new_end": new_end,
        }

    @classmethod
    def update_cap(
        cls,
        crowdloan_id: int,
        new_cap: "Balance",
    ) -> dict:
        """Returns the parameters for the `update_cap`."""
        return {
            "crowdloan_id": crowdloan_id,
            "new_cap": new_cap.rao,
        }
