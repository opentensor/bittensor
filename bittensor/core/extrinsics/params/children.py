from dataclasses import dataclass

from bittensor.utils import float_to_u64


@dataclass
class ChildrenParams:
    @classmethod
    def set_children(
        cls,
        hotkey_ss58: str,
        netuid: int,
        children: list[tuple[float, str]],
    ) -> dict:
        """Returns the parameters for the `set_children`."""
        params = {
            "children": [
                (float_to_u64(proportion), child_hotkey)
                for proportion, child_hotkey in children
            ],
            "hotkey": hotkey_ss58,
            "netuid": netuid,
        }
        return params

    @classmethod
    def set_pending_childkey_cooldown(
        cls,
        cooldown: int,
    ) -> dict:
        """Returns the parameters for the `set_pending_childkey_cooldown`."""
        return {"cooldown": cooldown}
