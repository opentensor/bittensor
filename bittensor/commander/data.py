from pydantic import BaseModel

import bittensor


class Config:
    netuid = 0
    wallet = None
    initialized = False

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


class ConfigBody(BaseModel):
    netuid: int = 0
    wallet: dict
