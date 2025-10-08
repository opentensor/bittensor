from dataclasses import dataclass


@dataclass
class RootParams:
    @classmethod
    def root_register(
        cls,
        hotkey_ss58: str,
    ) -> dict:
        """Returns the parameters for the `root_register`."""
        return {"hotkey": hotkey_ss58}
