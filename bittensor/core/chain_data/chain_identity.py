from dataclasses import dataclass
from bittensor.core.chain_data.info_base import InfoBase


@dataclass
class ChainIdentity(InfoBase):
    """Dataclass for chain identity information."""

    name: str
    github: str
    contact: str
    url: str
    discord: str
    description: str
    additional: str

    @classmethod
    def _from_dict(cls, decoded: dict) -> "ChainIdentity":
        return cls(
            name=decoded["subnet_name"],
            github=decoded["github_repo"],
            contact=decoded["subnet_contact"],
            url=decoded["subnet_url"],
            discord=decoded["discord"],
            description=decoded["description"],
            additional=decoded["additional"],
        )
