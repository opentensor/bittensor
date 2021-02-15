import pytest
from loguru import logger

from bittensor.subtensor.client import WSClient
from bittensor.subtensor.interface import Keypair
from bittensor.utils import asyncio


def connect(keypair, port):
    socket = "localhost:%i" % port
    client = WSClient(socket, keypair)

    client.connect()
    return client

@pytest.mark.asyncio
async def test_one_off():
    hotkeypair = Keypair.create_from_uri("//Alice")
    client = connect(hotkeypair, 9944)
    result = await client.is_connected()

    logger.info("Done")


@pytest.mark.asyncio
async def test_subscribe_success():
    hotkeypair = Keypair.create_from_uri("//Alice")
    client = connect(hotkeypair, 9944)
    coldkeypair = Keypair.create_from_uri("//Alice")

    await client.is_connected()

    await client.subscribe("8.8.8.8", 6666, 0, coldkeypair.public_key)

    await asyncio.sleep(10) # Sleep for at least one block to give extrinsic the chance to be put onto the blockchain
    uid = await client.get_uid_for_pubkey(hotkeypair.public_key)
    assert uid is not None