from dataclasses import dataclass


@dataclass
class TakeParams:
    @classmethod
    def increase_decrease_take(
        cls,
        hotkey_ss58: str,
        take: int,
    ) -> dict:
        """Returns the parameters for the `increase_take` and `decrease_take`."""
        return {"hotkey": hotkey_ss58, "take": take}
