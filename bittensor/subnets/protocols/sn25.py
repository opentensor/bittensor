# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Optional

import bittensor as bt
import numpy as np
import pydantic

class IsAlive( bt.Synapse ):
    answer: Optional[str] = None
    completion: str = pydantic.Field(
        "",
        title="Completion",
        description="Completion status of the current StreamPrompting object. "
                    "This attribute is mutable and can be updated.",
    )

class Train( bt.Synapse ):
    """
    A simple Train protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling request and response communication between
    the miner and the validator.

    Attributes:
    """

    # class Config:
    #     """
    #     Pydantic model configuration class for Prompting. This class sets validation of attribute assignment as True.
    #     validate_assignment set to True means the pydantic model will validate attribute assignments on the class.
    #     """

    #     validate_assignment = False
    #     arbitrary_types_allowed = True

    # Required request input, filled by sending dendrite caller.
    dataset_indices: list = [0, 1]

    # Initial peers
    initial_peers: str = "/ip4/127.0.0.1/tcp/8008/p2p/12D3KooWPVy8joVQgKe2o3LYncfFvHN1VCEZNV5UZhmzj45dSs1z"

    # Required request input hash, filled automatically when dendrite creates the request.
    # This allows for proper data validation and messages are signed with the hashes of the
    # required body fields. Ensure you have a {field}_hash field for each required field.
    # dummy_input_hash: str = ""

    # Required run_id
    # allows peers to set run_id for DHT connection
    run_id: str = "s25_test_run"

    # Optional request output, filled by recieving axon.
    # gradients: List[ bt.Tensor ] = []
    # gradients: list = None
    
    # Optional model name
    model_name: str = "kmfoda/tiny-random-gpt2"

    # # Optional learning rate
    lr: float = 1e-5
    
    # # Optional dataset name
    dataset_name: str = 'wikitext'

    # # Required optimizer
    # optimizer_name: str = "adam"

    # # Required batch size
    batch_size: int = 1

    # # Optional score
    loss: float = 0.0
    
    # # Training Steps
    # steps: int = 10