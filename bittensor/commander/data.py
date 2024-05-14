from typing import Optional

from pydantic import BaseModel

import bittensor


class Config:
    netuid = 0
    wallet = None
    initialized = False
    json_encrypted_path = None
    json_encrypted_pw = None

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


class ConfigBody(BaseModel):
    netuid: int = 0
    wallet: dict
    json_encrypted_path: Optional[str] = None
    json_encrypted_pw: Optional[str] = None
