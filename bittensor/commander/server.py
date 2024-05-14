import asyncio
from functools import partial
import time

# import sys

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# import uvicorn

import bittensor
from bittensor.commands import network
from bittensor.commands import metagraph
from bittensor.commands import register
from bittensor.commands import list as list_commands
from bittensor.commands import overview
from bittensor.commands import transfer
from bittensor.commands import inspect
from bittensor.commands import wallets
from bittensor.commander import data


# TODO app-wide error-handling

sub = bittensor.subtensor("test")
app = FastAPI(debug=True)
config = data.Config()


@app.post("/setup")
async def setup(conf: data.ConfigBody):
    config.setup(conf)
    return JSONResponse(status_code=200, content={"success": True})


async def check_config():
    if not config:
        raise HTTPException(status_code=401, detail="Config missing")


async def run_fn(command_class, *params):
    start = time.time()
    try:
        if hasattr(command_class, "commander_run"):
            # Offload synchronous execution to the threadpool
            response_content = await command_class.commander_run(sub, config=config)
            print(command_class, time.time() - start)
            return JSONResponse(content=response_content)
        else:
            raise HTTPException(
                status_code=501, detail="Command implementation missing"
            )
    except Exception as e:
        raise
        # raise HTTPException(status_code=500, detail=str(e))


# Subnets
@app.get("/subnets/{sub_cmd}", dependencies=[Depends(check_config)])
async def get_subnet(sub_cmd: str):
    routing_list = {
        "list": network.SubnetListCommand,
        "metagraph": metagraph.MetagraphCommand,
        "lock_cost": network.SubnetLockCostCommand,
        "create": network.RegisterSubnetworkCommand,
        "pow_register": register.PowRegisterCommand,
        "register": register.RegisterCommand,
        "hyperparameters": network.SubnetHyperparamsCommand,
    }
    return await run_fn(routing_list[sub_cmd])


# Wallet
@app.get("/wallet/hotkey/{sub_cmd}", dependencies=[Depends(check_config)])
async def wallet_hotkey(sub_cmd: str):
    routing_list = {
        "new": wallets.NewHotkeyCommand,
        "regen": wallets.RegenHotkeyCommand,
        "swap": register.SwapHotkeyCommand,
    }
    return await run_fn(routing_list[sub_cmd])


@app.get("/wallet/coldkey/{sub_cmd}", dependencies=[Depends(check_config)])
async def wallet_coldkey(sub_cmd: str):
    routing_list = {
        "new": wallets.NewColdkeyCommand,
        "regen": wallets.RegenColdkeyCommand,
        "regen/pub": wallets.RegenColdkeypubCommand,
    }
    return await run_fn(routing_list[sub_cmd])


@app.get("/wallet/{sub_cmd}", dependencies=[Depends(check_config)])
async def wallet(sub_cmd: str):
    routing_list = {
        "list": list_commands.ListCommand,
        "overview": overview.OverviewCommand,
        "transfer": transfer.TransferCommand,
        "inspect": inspect.InspectCommand,
        "balance": wallets.WalletBalanceCommand,
        "create": wallets.WalletCreateCommand,
        "faucet": register.RunFaucetCommand,
        "update": wallets.UpdateWalletCommand,
        "history": wallets.GetWalletHistoryCommand,
    }
    return await run_fn(routing_list[sub_cmd])


# Root

# Sudo

# Stake
