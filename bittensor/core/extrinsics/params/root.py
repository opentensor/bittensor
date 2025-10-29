from dataclasses import dataclass
from bittensor.core.types import UIDs


@dataclass
class RootParams:
    @classmethod
    def root_register(
        cls,
        hotkey_ss58: str,
    ) -> dict:
        """Returns the parameters for the `root_register`."""
        return {"hotkey": hotkey_ss58}

    @classmethod
    def set_root_claim_type(
        cls,
        new_root_claim_type: str,
    ) -> dict:
        """Returns the parameters for the `set_root_claim_type`."""
        return {"new_root_claim_type": new_root_claim_type}

    @classmethod
    def claim_root(
        cls,
        netuids: UIDs,
    ) -> dict:
        """Returns the parameters for the `claim_root`."""
        return {"subnets": netuids}
