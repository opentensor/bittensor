from typing import Optional, List, Dict
import bittensor as bt
from pydantic import BaseModel

# Model types
MODEL_TYPE_FUNDS_FLOW = "funds_flow"
MODEL_TYPE_FUNDS_FLOW_ID = 1
def get_model_id(model_type):
    if model_type == MODEL_TYPE_FUNDS_FLOW:
        return MODEL_TYPE_FUNDS_FLOW_ID

# Networks
NETWORK_BITCOIN = "bitcoin"
NETWORK_BITCOIN_ID = 1
NETWORK_DOGE = "doge"
NETWORK_DOGE_ID = 2

def get_network_by_id(id):
    if id == NETWORK_BITCOIN_ID:
        return NETWORK_BITCOIN
    if id == NETWORK_DOGE_ID:
        return NETWORK_DOGE
    return None
def get_network_id(network):
    if network == NETWORK_BITCOIN:
        return NETWORK_BITCOIN_ID
    if network == NETWORK_DOGE:
        return NETWORK_DOGE_ID
    return None

# Default settings for miners
MAX_MULTIPLE_RUN_ID = 9
MAX_MULTIPLE_IPS = 9








class MinerDiscoveryMetadata(BaseModel):
    network: str = None
    model_type: str = None
    graph_schema: Optional[Dict] = None
    #TODO: implement method for getting graph schema from miner


class MinerDiscoveryOutput(BaseModel):
    metadata: MinerDiscoveryMetadata = None
    data_samples: List[Dict] = None
    block_height: int = None
    start_block_height: int = None
    run_id: str = None
    version: Optional[int] = None

class MinerDiscovery(bt.Synapse):
    output: MinerDiscoveryOutput = None

    def deserialize(self):
        return self

class MinerRandomBlockCheckOutput(BaseModel):
    data_samples: List[Dict] = None

class MinerRandomBlockCheck(bt.Synapse):
    blocks_to_check: List[int] = None
    output: MinerRandomBlockCheckOutput = None

class MinerQuery(bt.Synapse):
    network: str = None
    model_type: str = None
    query: str = None
    output: Optional[List[Dict]] = None

    def deserialize(self) -> List[Dict]:
        return self.output