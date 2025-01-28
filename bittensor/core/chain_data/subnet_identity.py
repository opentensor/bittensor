from dataclasses import dataclass


@dataclass
class SubnetIdentity:
    """Dataclass for subnet identity information."""

    subnet_name: str
    github_repo: str
    subnet_contact: str

    # TODO: Add other methods when fetching from chain
