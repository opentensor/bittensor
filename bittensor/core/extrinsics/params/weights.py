from dataclasses import dataclass

from bittensor.core.types import UIDs, Weights, Salt


@dataclass
class WeightsParams:
    @classmethod
    def commit_timelocked_mechanism_weights(
        cls,
        netuid: int,
        mechid: int,
        commit_for_reveal: bytes,
        reveal_round: int,
        commit_reveal_version: int,
    ) -> dict:
        """Returns the parameters for the `commit_timelocked_mechanism_weights`."""
        return {
            "netuid": netuid,
            "mecid": mechid,
            "commit": commit_for_reveal,
            "reveal_round": reveal_round,
            "commit_reveal_version": commit_reveal_version,
        }

    @classmethod
    def commit_mechanism_weights(
        cls,
        netuid: int,
        mechid: int,
        commit_hash: str,
    ) -> dict:
        """Returns the parameters for the `commit_mechanism_weights`."""
        return {
            "netuid": netuid,
            "mecid": mechid,
            "commit_hash": commit_hash,
        }

    @classmethod
    def reveal_mechanism_weights(
        cls,
        netuid: int,
        mechid: int,
        uids: UIDs,
        weights: Weights,
        salt: Salt,
        version_key: int,
    ) -> dict:
        """Returns the parameters for the `reveal_mechanism_weights`."""
        return {
            "netuid": netuid,
            "mecid": mechid,
            "uids": uids,
            "values": weights,
            "salt": salt,
            "version_key": version_key,
        }

    @classmethod
    def set_mechanism_weights(
        cls,
        netuid: int,
        mechid: int,
        uids: UIDs,
        weights: Weights,
        version_key: int,
    ) -> dict:
        """Returns the parameters for the `set_mechanism_weights`."""
        return {
            "netuid": netuid,
            "mecid": mechid,
            "dests": uids,
            "weights": weights,
            "version_key": version_key,
        }
