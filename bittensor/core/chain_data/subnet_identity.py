from dataclasses import dataclass
from typing import Optional


@dataclass
class SubnetIdentity:
    """Dataclass for subnet identity information."""

    subnet_name: str
    github_repo: str
    subnet_contact: str
    subnet_url: str
    logo_url: str
    discord: str
    description: str
    additional: str
    x_handle: Optional[str] = (
        None
    )

    @classmethod
    def _from_dict(cls, decoded: dict) -> "SubnetIdentity":
        """Returns a SubnetIdentity object from decoded chain data."""
        return cls(
            subnet_name=decoded["subnet_name"],
            github_repo=decoded["github_repo"],
            subnet_contact=decoded["subnet_contact"],
            subnet_url=decoded["subnet_url"],
            logo_url=decoded["logo_url"],
            discord=decoded["discord"],
            description=decoded["description"],
            additional=decoded["additional"],
            x_handle=decoded.get("x_handle"),
        )
