# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

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

import asyncio
import time
from typing import Union, Optional, List

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse

import bittensor
from bittensor.commands import (
    network,
    metagraph,
    register,
    overview,
    transfer,
    inspect,
    wallets,
    identity,
    senate,
    delegates,
    unstake,
)
from bittensor.commands import list as list_commands
from bittensor.commands import root as root_commands
from bittensor.commands import stake as stake_commands
from bittensor.commander import data

# TODO app-wide error-handling

sub = bittensor.subtensor("test")
app = FastAPI(debug=True)
config = data.Config()
event_loop = asyncio.get_event_loop()


async def check_config():
    if not config:
        raise HTTPException(status_code=401, detail="Config missing")


async def is_cold_key_unlocked():
    if not config.initialized and not config.wallet.cold_key_unlocked:
        raise HTTPException(
            status_code=401,
            detail="Cold key needs to be unlocked before performing this operation.",
        )


@app.post("/setup")
async def setup(conf: data.ConfigBody):
    config.setup(conf)
    return JSONResponse(status_code=200, content={"success": True})


@app.get("/setup", dependencies=[Depends(check_config)])
async def get_setup():
    return JSONResponse(status_code=200, content=config.as_dict())


@app.post("/unlock-cold-key", dependencies=[Depends(check_config)])
async def unlock_cold_key(password: data.Password):
    result = await event_loop.run_in_executor(
        None, config.wallet.unlock_coldkey, password.password
    )
    if result is True:
        config.wallet.cold_key_unlocked = True
    return JSONResponse({"success": result})


async def run_fn(command_class, params=None):
    start = time.time()
    try:
        if hasattr(command_class, "commander_run"):
            response_content = await command_class.commander_run(
                sub, config=config, params=params
            )
            print(command_class, time.time() - start)
            return JSONResponse(content=response_content)
        else:
            raise HTTPException(
                status_code=501, detail="Command implementation missing"
            )
    except Exception as e:
        raise
        # raise HTTPException(status_code=500, detail=str(e))


# Subnets #######################
@app.get("/subnets/create", dependencies=[Depends(check_config)])
async def subnets_create(set_identity: bool):
    return await run_fn(
        network.RegisterSubnetworkCommand, params={"set_identity": set_identity}
    )


@app.get("/subnets/{sub_cmd}", dependencies=[Depends(check_config)])
async def get_subnet(sub_cmd: str):
    routing_list = {
        "list": network.SubnetListCommand,
        "metagraph": metagraph.MetagraphCommand,
        "lock_cost": network.SubnetLockCostCommand,
        # "pow_register": register.PowRegisterCommand,  # Not yet working
        "register": register.RegisterCommand,
        "hyperparameters": network.SubnetHyperparamsCommand,
    }
    return await run_fn(routing_list[sub_cmd])


# Wallet #######################
@app.get("/wallet/new_key/{key_type}", dependencies=[Depends(check_config)])
async def wallet_new_key(
    key_type: str, n_words: int, use_password: bool, overwrite: bool
):
    routing_list = {
        "hotkey": wallets.NewHotkeyCommand,
        "coldkey": wallets.NewColdkeyCommand,
    }
    try:
        return await run_fn(
            routing_list[key_type],
            params={
                "n_words": n_words,
                "use_password": use_password,
                "overwrite": overwrite,
            },
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Key type not found")


@app.get("/wallet/{key_type}/regen_key", dependencies=[Depends(check_config)])
async def wallet_regen_key(
    key_type: str,
    mnemonic: Union[str, None],
    seed: Union[str, None],
    use_password: bool = False,
    overwrite: bool = False,
):
    routing_list = {
        "hotkey": wallets.RegenHotkeyCommand,
        "coldkey": wallets.RegenColdkeyCommand,
    }
    try:
        return await run_fn(
            routing_list[key_type],
            params={
                "mnemonic": mnemonic,
                "seed": seed,
                "use_password": use_password,
                "overwrite": overwrite,
            },
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Key type not found")


@app.get("/wallet/hotkey/swap", dependencies=[Depends(check_config)])
async def wallet_hotkey_swap():
    return await run_fn(register.SwapHotkeyCommand)  # TODO


@app.get("/wallet/coldkey/regen_coldkey_pub", dependencies=[Depends(check_config)])
async def wallet_coldkey(
    ss58_address: Optional[str],
    public_key: Optional[
        str
    ],  # This differs from the CLI implementation that also allows bytes
    overwrite: Optional[bool] = False,
):
    return await run_fn(
        wallets.RegenColdkeypubCommand,
        params={
            "ss58_address": ss58_address,
            "public_key": public_key,
            "overwrite": overwrite,
        },
    )


@app.get("/wallet/overview", dependencies=[Depends(check_config)])
async def wallet_overview(
    all_coldkeys: bool = True, hotkeys: List[str] = None, all_hotkeys: bool = True
):
    # Hotkeys is List[str] I think
    return await run_fn(
        overview.OverviewCommand,
        params={
            "all_coldkeys": all_coldkeys,
            "hotkeys": hotkeys,
            "all_hotkeys": all_hotkeys,
        },
    )


@app.get("/wallet/transfer", dependencies=[Depends(check_config)])
async def wallet_transfer(dest: str, amount: float):
    # Be sure to unlock the key first, if the key is encrypted
    return await run_fn(
        transfer.TransferCommand, params={"dest": dest, "amount": amount}
    )


@app.get("/wallet/identity/get", dependencies=[Depends(check_config)])
async def wallet_identity_get(key: str):
    return await run_fn(identity.GetIdentityCommand, params={"key": key})


@app.get(
    "/wallet/identity/set",
    dependencies=[Depends(check_config), Depends(is_cold_key_unlocked)],
)
async def wallet_identity_set(
    operating_hotkey_identity: bool,
    display: str = "",
    legal: str = "",
    web: str = "",
    pgp_fingerprint: str = "",
    riot: str = "",
    email: str = "",
    image: str = "",
    twitter: str = "",
    info: str = "",
):
    return await run_fn(
        identity.SetIdentityCommand,
        params={
            "display": display,
            "legal": legal,
            "web": web,
            "pgp_fingerprint": pgp_fingerprint,
            "riot": riot,
            "email": email,
            "image": image,
            "twitter": twitter,
            "info": info,
            "operating_hotkey_identity": operating_hotkey_identity,
        },
    )


@app.get("/wallet/{sub_cmd}", dependencies=[Depends(check_config)])
async def wallet(
    sub_cmd: str,
    n_words: int = 12,
    use_password: bool = False,
    overwrite: bool = False,
    all_wallets: bool = False,
):
    routing_list = {
        "list": list_commands.ListCommand,
        "balance": wallets.WalletBalanceCommand,
        "inspect": inspect.InspectCommand,
        "create": wallets.WalletCreateCommand,
        "faucet": register.RunFaucetCommand,
        "update": wallets.UpdateWalletCommand,
        "history": wallets.GetWalletHistoryCommand,
    }
    return await run_fn(
        routing_list[sub_cmd],
        params={
            "n_words": n_words,
            "use_password": use_password,
            "overwrite": overwrite,
            "all_wallets": all_wallets,
        },
    )


# Root
@app.get(
    "/root/nominate",
    dependencies=[Depends(check_config), Depends(is_cold_key_unlocked)],
)
async def root_nominate():
    return await run_fn(delegates.NominateCommand)


@app.get("/root/weights/set", dependencies=[Depends(check_config)])
async def root_weights(netuids: list[int], weights: list[float]):
    return await run_fn(
        root_commands.RootSetWeightsCommand,
        params={"netuids": netuids, "weights": weights},
    )


@app.get("/root/delegates/{sub_cmd}", dependencies=[Depends(check_config)])
async def root_delegates(sub_cmd: str):
    routing_list = {
        "delegate": delegates.DelegateStakeCommand,
        "undelegate": delegates.DelegateUnstakeCommand,
        "my": delegates.MyDelegatesCommand,
        "list": delegates.ListDelegatesCommand,
        "list_lite": delegates.ListDelegatesLiteCommand,
    }


@app.get("/root/{sub_cmd}", dependencies=[Depends(check_config)])
async def root(sub_cmd: str, amount: int = 0):
    routing_list = {
        "list": root_commands.RootList,
        "boost": root_commands.RootSetBoostCommand,
        "slash": root_commands.RootSetSlashCommand,
        "register": root_commands.RootRegisterCommand,
        "proposals": senate.ProposalsCommand,
        "weights": root_commands.RootGetWeightsCommand,
    }
    return await run_fn(
        routing_list[sub_cmd],
        params={
            "amount": amount,
        },
    )


# Sudo
@app.get("/sudo/get", dependencies=[Depends(check_config)])
async def sudo_get():
    return await run_fn(network.SubnetGetHyperparamsCommand)


@app.get("/sudo/set", dependencies=[Depends(check_config)])
async def sudo_set(param: str, value: Union[str, int, bool, float]):
    return await run_fn(
        network.SubnetSudoCommand, params={"param": param, "value": value}
    )


# Stake
@app.post("/stake/add", dependencies=[Depends(check_config)])
async def stake_add(params: data.StakeAdd):
    return await run_fn(stake_commands.StakeCommand, params=params.model_dump())


@app.get("/stake/{sub_cmd}", dependencies=[Depends(check_config)])
async def stake(sub_cmd: str, all_wallets: bool = False):
    routing_list = {
        "show": stake_commands.StakeShow,
        "remove": unstake.UnStakeCommand,
    }
    return await run_fn(routing_list[sub_cmd], params={"all_wallets": all_wallets})
