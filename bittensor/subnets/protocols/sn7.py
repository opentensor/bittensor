# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 salahawk <tylermcguy@gmail.com>

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

import typing
import bittensor as bt
from bittensor.synapse import Synapse


class Store(bt.Synapse):
    # Key of Data.
    key: int = -1
    # String encoded data.
    data: str = ""

    required_hash_fields: typing.List[str] = ["key", "data"]
    
    # Deserialize responses.
    def deserialize(self) -> int:
        return self.key

class Ping(bt.Synapse):
    data: str = ""
    def deserialize(self) -> str:
        return self.data

class Retrieve(bt.Synapse):
    # Key of data.
    key_list: dict = {}
    key: str = ""
    # String encoded data.
    data: typing.Optional[str] = None

    required_hash_fields: typing.List[str] = ["key", "data"]
    
    # Deserialize responses.
    def deserialize(self) -> str:
        return self.data