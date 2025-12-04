"""
This module provides data structures and functions for working with the Bittensor network, including neuron and subnet
information, SCALE encoding/decoding, and custom RPC type registry.
"""

from scalecodec.types import GenericCall

from .axon_info import AxonInfo
from .chain_identity import ChainIdentity
from .crowdloan_info import CrowdloanInfo, CrowdloanConstants
from .delegate_info import DelegateInfo, DelegatedInfo
from .delegate_info_lite import DelegateInfoLite
from .dynamic_info import DynamicInfo
from .ip_info import IPInfo
from .metagraph_info import (
    MetagraphInfo,
    MetagraphInfoEmissions,
    MetagraphInfoPool,
    MetagraphInfoParams,
    SelectiveMetagraphIndex,
)
from .neuron_info import NeuronInfo
from .neuron_info_lite import NeuronInfoLite
from .prometheus_info import PrometheusInfo
from .proposal_vote_data import ProposalVoteData
from .proxy import ProxyConstants, ProxyInfo, ProxyType, ProxyAnnouncementInfo
from .root_claim import RootClaimType
from .scheduled_coldkey_swap_info import ScheduledColdkeySwapInfo
from .stake_info import StakeInfo
from .sim_swap import SimSwapResult
from .subnet_hyperparameters import SubnetHyperparameters
from .subnet_identity import SubnetIdentity
from .subnet_info import SubnetInfo
from .subnet_state import SubnetState
from .weight_commit_info import WeightCommitInfo
from .utils import decode_account_id, process_stake_data

ProposalCallData = GenericCall

__all__ = [
    "AxonInfo",
    "ChainIdentity",
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
