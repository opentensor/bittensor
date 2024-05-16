# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

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

from typing import Optional

from pydantic import BaseModel

import bittensor


class Config:
    netuid = 0
    wallet = None
    initialized = False
    json_encrypted_path = None
    json_encrypted_pw = None
    netuids = []  # TODO implement

    def __bool__(self):
        return self.initialized

    def setup(self, conf: "ConfigBody"):
        self.initialized = True
        self.netuid = conf.netuid
        self.wallet = bittensor.wallet(
            name=conf.wallet["name"],
            hotkey=conf.wallet["hotkey"],
            path=conf.wallet["path"],
        )  # maybe config
        self.json_encrypted_path = conf.json_encrypted_path
        self.json_encrypted_pw = conf.json_encrypted_pw

    def as_dict(self):
        return {
            "initialized": self.initialized,
            "netuid": self.netuid,
            "wallet": {
                "name": self.wallet.name,
                "hotkey": self.wallet.hotkey.ss58_address,
                "path": self.wallet.path,
            },
            "json_encrypted_path": self.json_encrypted_path,
            "json_encrypted_pw": self.json_encrypted_pw,
        }


class ConfigBody(BaseModel):
    netuid: int = 0
    wallet: dict
    json_encrypted_path: Optional[str] = None
    json_encrypted_pw: Optional[str] = None


class Password(BaseModel):
    # TODO maybe encrypt this?
    password: str
