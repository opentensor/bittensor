from .stake import StakeCommand
from .unstake import UnStakeCommand
from .overview import OverviewCommand
from .register import RegisterCommand, RecycleRegisterCommand
from .delegates import NominateCommand, ListDelegatesCommand, DelegateStakeCommand, DelegateUnstakeCommand, MyDelegatesCommand
from .wallets import NewColdkeyCommand, NewHotkeyCommand, RegenColdkeyCommand, RegenColdkeypubCommand, RegenHotkeyCommand
from .transfer import TransferCommand
from .inspect import InspectCommand
from .metagraph import MetagraphCommand
from .list import ListCommand
from .misc import UpdateCommand, ListSubnetsCommand
from .senate import SenateCommand, ProposalsCommand, ShowVotesCommand, SenateRegisterCommand, SenateLeaveCommand, VoteCommand

from bittensor_config import Config


defaults: Config = Config.fromDict(
    {
        "netuid": 1,
        "subtensor": {
            "network": "finney",
            "chain_endpoint": None,
            "_mock": False,
        },
        "register": {
            "num_processes": None,
            "update_interval": 50000,
            "output_in_place": True,
            "verbose": False,
            "cuda": {"dev_id": [0], "use_cuda": False, "TPB": 256},
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
