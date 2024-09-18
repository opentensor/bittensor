# The MIT License (MIT)
# Copyright © 2023 Opentensor Technologies Inc

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

from munch import Munch, munchify

defaults: Munch = munchify(
    {
        "netuid": 1,
        "subtensor": {"network": "finney", "chain_endpoint": None, "_mock": False},
        "pow_register": {
            "num_processes": None,
            "update_interval": 50000,
            "output_in_place": True,
            "verbose": False,
            "cuda": {"dev_id": [0], "use_cuda": False, "tpb": 256},
        },
        "axon": {
            "port": 8091,
            "ip": "[::]",
            "external_port": None,
            "external_ip": None,
            "max_workers": 10,
            "maximum_concurrent_rpcs": 400,
        },
        "priority": {"max_workers": 5, "maxsize": 10},
        "prometheus": {"port": 7091, "level": "INFO"},
        "wallet": {
            "name": "default",
            "hotkey": "default",
            "path": "~/.bittensor/wallets/",
        },
        "dataset": {
            "batch_size": 10,
            "block_size": 20,
            "num_workers": 0,
            "dataset_names": "default",
            "data_dir": "~/.bittensor/data/",
            "save_dataset": False,
            "max_datasets": 3,
            "num_batches": 100,
        },
        "logging": {
            "debug": False,
            "trace": False,
            "record_log": False,
            "logging_dir": "~/.bittensor/miners",
        },
    }
)

from .stake import (
    StakeCommand,
    StakeShow,
    SetChildrenCommand,
    GetChildrenCommand,
    SetChildKeyTakeCommand,
    GetChildKeyTakeCommand,
)
from .unstake import UnStakeCommand, RevokeChildrenCommand
from .overview import OverviewCommand
from .register import (
    PowRegisterCommand,
    RegisterCommand,
    RunFaucetCommand,
    SwapHotkeyCommand,
)
from .delegates import (
    NominateCommand,
    ListDelegatesCommand,
    DelegateStakeCommand,
    DelegateUnstakeCommand,
    MyDelegatesCommand,
    SetTakeCommand,
)
from .wallets import (
    NewColdkeyCommand,
    NewHotkeyCommand,
    RegenColdkeyCommand,
    RegenColdkeypubCommand,
    RegenHotkeyCommand,
    UpdateWalletCommand,
    WalletCreateCommand,
    WalletBalanceCommand,
    GetWalletHistoryCommand,
)
from .weights import CommitWeightCommand, RevealWeightCommand
from .transfer import TransferCommand
from .inspect import InspectCommand
from .metagraph import MetagraphCommand
from .list import ListCommand
from .misc import UpdateCommand, AutocompleteCommand
from .senate import (
    SenateCommand,
    ProposalsCommand,
    ShowVotesCommand,
    SenateRegisterCommand,
    SenateLeaveCommand,
    VoteCommand,
)
from .network import (
    RegisterSubnetworkCommand,
    SubnetLockCostCommand,
    SubnetListCommand,
    SubnetSudoCommand,
    SubnetHyperparamsCommand,
    SubnetGetHyperparamsCommand,
)
from .root import (
    RootRegisterCommand,
    RootList,
    RootSetWeightsCommand,
    RootGetWeightsCommand,
    RootSetBoostCommand,
    RootSetSlashCommand,
)
from .identity import GetIdentityCommand, SetIdentityCommand
from .check_coldkey_swap import CheckColdKeySwapCommand
