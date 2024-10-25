"""
This module provides data structures and functions for working with the Bittensor network, including neuron and subnet
information, SCALE encoding/decoding, and custom RPC type registry.
"""

from scalecodec.types import GenericCall

from .axon_info import AxonInfo
from .delegate_info import DelegateInfo
from .delegate_info_lite import DelegateInfoLite
from .ip_info import IPInfo
from .neuron_info import NeuronInfo
from .neuron_info_lite import NeuronInfoLite
from .prometheus_info import PrometheusInfo
from .proposal_vote_data import ProposalVoteData
from .scheduled_coldkey_swap_info import ScheduledColdkeySwapInfo
from .stake_info import StakeInfo
from .subnet_hyperparameters import SubnetHyperparameters
from .subnet_info import SubnetInfo
from .subnet_state import SubnetState
from .utils import custom_rpc_type_registry

ProposalCallData = GenericCall
