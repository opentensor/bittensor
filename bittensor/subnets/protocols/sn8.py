# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Taoshidev
# Copyright © 2023 Taoshi Inc

import typing
import bittensor as bt
from pydantic import Field

from typing import List


class BaseProtocol(bt.Synapse):
    request_uuid: str = Field(..., allow_mutation=False)
    stream_id: str = Field(..., allow_mutation=False)
    samples: typing.Optional[bt.Tensor] = None
    topic_id: typing.Optional[int] = Field(..., allow_mutation=False)


class Forward(BaseProtocol):
    feature_ids: List[float]
    prediction_size: int = Field(..., allow_mutation=False)
    schema_id: typing.Optional[int] = Field(..., allow_mutation=False)
    predictions: typing.Optional[bt.Tensor] = None

    # def deserialize(self) -> bt.Tensor:
    #     return self.predictions


class Backward(BaseProtocol):
    received: bool = None

    # def deserialize(self) -> bool:
    #     return self.received


class TrainingForward(Forward):
    pass


class LiveForward(Forward):
    pass


class TrainingBackward(Backward):
    pass


class LiveBackward(Backward):
    pass



