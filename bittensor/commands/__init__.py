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


from .stake import ( 
    StakeCommand,   # noqa: F401
    StakeShow,    # noqa: F401
    SetChildCommand,    # noqa: F401
    SetChildrenCommand,     # noqa: F401
    GetChildrenCommand,     # noqa: F401
)
from .unstake import UnStakeCommand    # noqa: F401
from .overview import OverviewCommand    # noqa: F401
from .register import (
    PowRegisterCommand,    # noqa: F401
    RegisterCommand,    # noqa: F401
    RunFaucetCommand,    # noqa: F401
    SwapHotkeyCommand,    # noqa: F401
)
from .delegates import (
    NominateCommand,    # noqa: F401
    ListDelegatesCommand,    # noqa: F401
    DelegateStakeCommand,    # noqa: F401
    DelegateUnstakeCommand,    # noqa: F401
    MyDelegatesCommand,    # noqa: F401
    SetTakeCommand,    # noqa: F401
)
from .wallets import (
    NewColdkeyCommand,    # noqa: F401
    NewHotkeyCommand,    # noqa: F401
    RegenColdkeyCommand,    # noqa: F401
    RegenColdkeypubCommand,    # noqa: F401
    RegenHotkeyCommand,    # noqa: F401
    UpdateWalletCommand,    # noqa: F401
    WalletCreateCommand,    # noqa: F401
    WalletBalanceCommand,    # noqa: F401
    GetWalletHistoryCommand,    # noqa: F401
)
from .weights import CommitWeightCommand, RevealWeightCommand    # noqa: F401
from .transfer import TransferCommand    # noqa: F401
from .inspect import InspectCommand    # noqa: F401
from .metagraph import MetagraphCommand    # noqa: F401
from .list import ListCommand    # noqa: F401
from .misc import UpdateCommand, AutocompleteCommand    # noqa: F401
from .senate import (
    SenateCommand,    # noqa: F401
    ProposalsCommand,    # noqa: F401
    ShowVotesCommand,    # noqa: F401
    SenateRegisterCommand,    # noqa: F401
    SenateLeaveCommand,    # noqa: F401
    VoteCommand,    # noqa: F401
)
from .network import (
    RegisterSubnetworkCommand,    # noqa: F401
    SubnetLockCostCommand,    # noqa: F401
    SubnetListCommand,    # noqa: F401
    SubnetSudoCommand,    # noqa: F401
    SubnetHyperparamsCommand,    # noqa: F401
    SubnetGetHyperparamsCommand,    # noqa: F401
)
from .root import (
    RootRegisterCommand,    # noqa: F401
    RootList,    # noqa: F401
    RootSetWeightsCommand,    # noqa: F401
    RootGetWeightsCommand,    # noqa: F401
    RootSetBoostCommand,    # noqa: F401
    RootSetSlashCommand,    # noqa: F401
)
from .identity import GetIdentityCommand, SetIdentityCommand    # noqa: F401

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

