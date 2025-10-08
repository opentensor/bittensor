from dataclasses import dataclass


@dataclass
class StartCallParams:
    @classmethod
    def start_call(
        cls,
        netuid: int,
    ) -> dict:
        """Returns the parameters for the `start_call`."""
        return {"netuid": netuid}
