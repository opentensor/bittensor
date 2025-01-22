from dataclasses import dataclass


@dataclass
class ChainIdentity:
    """Dataclass for chain identity information."""

    # In `bittensor.core.chain_data.utils.custom_rpc_type_registry` represents as `ChainIdentityOf` structure.

    name: str
    url: str
    image: str
    discord: str
    description: str
    additional: list[str, int]
