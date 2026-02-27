"""
This module provides data structures and functions for working with the Bittensor network, including neuron and subnet
information, SCALE encoding/decoding, and custom RPC type registry.
"""

from scalecodec.types import GenericCall

from .axon_info import AxonInfo
from .chain_identity import ChainIdentity
from .coldkey_swap import (
    ColdkeySwapAnnouncementInfo,
    ColdkeySwapConstants,
    ColdkeySwapDisputeInfo,
)
from .crowdloan_info import CrowdloanConstants, CrowdloanInfo
from .delegate_info import DelegatedInfo, DelegateInfo
from .delegate_info_lite import DelegateInfoLite
from .dynamic_info import DynamicInfo
from .ip_info import IPInfo
from .metagraph_info import (
    MetagraphInfo,
    MetagraphInfoEmissions,
    MetagraphInfoParams,
    MetagraphInfoPool,
    SelectiveMetagraphIndex,
)
from .neuron_info import NeuronInfo
from .neuron_info_lite import NeuronInfoLite
from .prometheus_info import PrometheusInfo
from .proposal_vote_data import ProposalVoteData
from .proxy import ProxyAnnouncementInfo, ProxyConstants, ProxyInfo, ProxyType
from .root_claim import RootClaimType
from .scheduled_coldkey_swap_info import ScheduledColdkeySwapInfo
from .sim_swap import SimSwapResult
from .stake_info import StakeInfo
from .subnet_hyperparameters import SubnetHyperparameters
from .subnet_identity import SubnetIdentity
from .subnet_info import SubnetInfo
from .subnet_state import SubnetState
from .utils import decode_account_id, process_stake_data
from .weight_commit_info import WeightCommitInfo

ProposalCallData = GenericCall

__all__ = [
    "AxonInfo",
    "ChainIdentity",
    "ColdkeySwapAnnouncementInfo",
    "ColdkeySwapConstants",
    "ColdkeySwapDisputeInfo",
    "CrowdloanInfo",
    "CrowdloanConstants",
    "DelegateInfo",
    "DelegatedInfo",
    "DelegateInfoLite",
    "DynamicInfo",
    "IPInfo",
    "MetagraphInfo",
    "MetagraphInfoEmissions",
    "MetagraphInfoParams",
    "MetagraphInfoPool",
    "NeuronInfo",
    "NeuronInfoLite",
    "PrometheusInfo",
    "ProposalCallData",
    "ProposalVoteData",
    "ProxyConstants",
    "ProxyAnnouncementInfo",
    "ProxyInfo",
    "ProxyType",
    "RootClaimType",
    "ScheduledColdkeySwapInfo",
    "SelectiveMetagraphIndex",
    "SimSwapResult",
    "StakeInfo",
    "SubnetHyperparameters",
    "SubnetIdentity",
    "SubnetInfo",
    "SubnetState",
    "WeightCommitInfo",
    "decode_account_id",
    "process_stake_data",
]
