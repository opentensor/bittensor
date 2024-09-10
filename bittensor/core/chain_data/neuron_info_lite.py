from dataclasses import dataclass
from typing import Dict, Optional

import bt_decode
import netaddr

from bittensor.core.chain_data.axon_info import AxonInfo
from bittensor.core.chain_data.prometheus_info import PrometheusInfo
from bittensor.core.chain_data.utils import decode_account_id, process_stake_data
from bittensor.utils import u16_normalized_float
from bittensor.utils.balance import Balance


@dataclass
class NeuronInfoLite:
    """Dataclass for neuron metadata, but without the weights and bonds."""

    hotkey: str
    coldkey: str
    uid: int
    netuid: int
    active: int
    stake: "Balance"
    # mapping of coldkey to amount staked to this Neuron
    stake_dict: Dict[str, "Balance"]
    total_stake: "Balance"
    rank: float
    emission: float
    incentive: float
    consensus: float
    trust: float
    validator_trust: float
    dividends: float
    last_update: int
    validator_permit: bool
    prometheus_info: Optional["PrometheusInfo"]
    axon_info: Optional["AxonInfo"]
    pruning_score: int
    is_null: bool = False

    @staticmethod
    def get_null_neuron() -> "NeuronInfoLite":
        neuron = NeuronInfoLite(
            uid=0,
            netuid=0,
            active=0,
            stake=Balance.from_rao(0),
            stake_dict={},
            total_stake=Balance.from_rao(0),
            rank=0,
            emission=0,
            incentive=0,
            consensus=0,
            trust=0,
            validator_trust=0,
            dividends=0,
            last_update=0,
            validator_permit=False,
            prometheus_info=None,
            axon_info=None,
            is_null=True,
            coldkey="000000000000000000000000000000000000000000000000",
            hotkey="000000000000000000000000000000000000000000000000",
            pruning_score=0,
        )
        return neuron

    @classmethod
    def list_from_vec_u8(cls, vec_u8: bytes) -> list["NeuronInfoLite"]:
        decoded = bt_decode.NeuronInfoLite.decode_vec(vec_u8)
        results = []
        for item in decoded:
            active = item.active
            axon_info = item.axon_info
            coldkey = decode_account_id(item.coldkey)
            consensus = item.consensus
            dividends = item.dividends
            emission = item.emission
            hotkey = decode_account_id(item.hotkey)
            incentive = item.incentive
            last_update = item.last_update
            netuid = item.netuid
            prometheus_info = item.prometheus_info
            pruning_score = item.pruning_score
            rank = item.rank
            stake_dict = process_stake_data(item.stake)
            stake = sum(stake_dict.values()) if stake_dict else Balance(0)
            trust = item.trust
            uid = item.uid
            validator_permit = item.validator_permit
            validator_trust = item.validator_trust
            results.append(
                NeuronInfoLite(
                    active=active,
                    axon_info=AxonInfo(
                        version=axon_info.version,
                        ip=str(netaddr.IPAddress(axon_info.ip)),
                        port=axon_info.port,
                        ip_type=axon_info.ip_type,
                        placeholder1=axon_info.placeholder1,
                        placeholder2=axon_info.placeholder2,
                        protocol=axon_info.protocol,
                        hotkey=hotkey,
                        coldkey=coldkey,
                    ),
                    coldkey=coldkey,
                    consensus=u16_normalized_float(consensus),
                    dividends=u16_normalized_float(dividends),
                    emission=emission / 1e9,
                    hotkey=hotkey,
                    incentive=u16_normalized_float(incentive),
                    last_update=last_update,
                    netuid=netuid,
                    prometheus_info=PrometheusInfo(
                        version=prometheus_info.version,
                        ip=str(netaddr.IPAddress(prometheus_info.ip)),
                        port=prometheus_info.port,
                        ip_type=prometheus_info.ip_type,
                        block=prometheus_info.block,
                    ),
                    pruning_score=pruning_score,
                    rank=u16_normalized_float(rank),
                    stake_dict=stake_dict,
                    stake=stake,
                    total_stake=stake,
                    trust=u16_normalized_float(trust),
                    uid=uid,
                    validator_permit=validator_permit,
                    validator_trust=u16_normalized_float(validator_trust),
                )
            )
        return results
