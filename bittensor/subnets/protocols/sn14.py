import typing
import bittensor as bt
import pydantic


class LLMDefenderProtocol(bt.Synapse):
    """
    This class implements the protocol definition for the the
    llm-defender subnet.

    The protocol is a simple request-response communication protocol in
    which the validator sends a request to the miner for processing
    activities.
    """

    # Parse variables
    prompt: typing.Optional[str] = None
    engine: typing.Optional[str] = None
    output: typing.Optional[dict] = None

    synapse_uuid: str = pydantic.Field(
        ...,
        description="Synapse UUID",
        allow_mutation=False
    )

    subnet_version: int = pydantic.Field(
        ...,
        description="Current subnet version",
        allow_mutation=False,
    )

    roles: typing.List[str] = pydantic.Field(
        ...,
        title="Roles",
        description="An immutable list depicting the roles",
        allow_mutation=False,
        regex=r"^(internal|external)$",
    )

    analyzer: typing.List[str] = pydantic.Field(
        ...,
        title="analyzer",
        description="An immutable list depicting the analyzers to execute",
        allow_mutation=False,
        regex=r"^(Prompt Injection)$",
    )

    def get_analyzers(self) -> list:
        """Returns the analyzers associated with the synapse"""

        return self.analyzer

    def deserialize(self) -> bt.Synapse:
        """Deserialize the instance of the protocol"""
        return self
