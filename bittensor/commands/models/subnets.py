from typing import List

from pydantic import BaseModel


class Datum(BaseModel):
    value: str


class Stake(BaseModel):
    data: List[Datum]
    uid: int


class Rank(BaseModel):
    uid: int
    data: List[Datum]


class Trust(BaseModel):
    data: List[Datum]
    uid: int


class Consensus(BaseModel):
    uid: int
    data: List[Datum]


class Incentive(BaseModel):
    data: List[Datum]
    uid: int


class Dividend(BaseModel):
    uid: int
    data: List[Datum]


class Emission(BaseModel):
    data: List[Datum]
    uid: int


class ValidatorTrust(BaseModel):
    data: List[Datum]
    uid: int


class Axon(BaseModel):
    data: List[Datum]
    uid: int


class BoolData(BaseModel):
    value: bool


class Active(BaseModel):
    data: List[BoolData]
    uid: int


class LastUpdate(BaseModel):
    data: List[Datum]
    uid: int


class Coldkey(BaseModel):
    data: List[Datum]
    uid: int


class ValidatorPermit(BaseModel):
    data: List[BoolData]
    uid: int


class HotKey(BaseModel):
    key: str
    uid: int


class Uids(BaseModel):
    stake: List[Stake]
    rank: List[Rank]
    trust: List[Trust]
    consensus: List[Consensus]
    incentive: List[Incentive]
    dividends: List[Dividend]
    emission: List[Emission]
    validatorTrust: List[ValidatorTrust]
    axons: List[Axon]
    active: List[Active]
    lastUpdate: List[LastUpdate]
    coldkey: List[Coldkey]
    validatorPermit: List[ValidatorPermit]
    hotkey: List[HotKey]


class Difficulty(BaseModel):
    value: str


class Subnet(BaseModel):
    uids: Uids
    difficulty: List[Difficulty]


class TotalIssuance(BaseModel):
    value: int
    blockNumber: int


class Data(BaseModel):
    subnets: List[Subnet]
    totalIssuance: List[TotalIssuance]


class FetchMetagraphData(BaseModel):
    data: Data
