from dataclasses import dataclass
from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class ChainIdentity(InfoBase):
    """Dataclass for chain identity information."""

    name: str
    url: str
    github: str
    image: str
    discord: str
    description: str
    additional: str

    @classmethod
    def _from_dict(cls, decoded: dict) -> "ChainIdentity":
        """Returns a ChainIdentity object from decoded chain data."""
        return cls(
            name=decoded["name"],
            url=decoded["url"],
            github=decoded["github_repo"],
            image=decoded["image"],
            discord=decoded["discord"],
            description=decoded["description"],
            additional=decoded["additional"],
        )
