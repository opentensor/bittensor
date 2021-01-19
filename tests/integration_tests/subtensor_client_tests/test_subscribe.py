from bittensor.subtensor.client import WSClient
from bittensor.subtensor.interface import Keypair
from loguru import logger
import pytest
import asyncio

logger.remove() # Shut up loguru


socket = "localhost:9944"
keypair = Keypair.create_from_uri('//Alice')
client = WSClient(socket, keypair)


@pytest.mark.asyncio
async def test_subscribe():
    client.connect()
    await client.is_connected()

    await client.subscribe("8.8.8.8", 666, 0, keypair.public_key)
    await asyncio.sleep(10)
    uid = await client.get_uid_for_pubkey(keypair.public_key)

    assert uid is not None


@pytest.mark.asyncio
async def get_uid_for_pubkey__does_not_exist():
    client.connect()
    await client.is_connected()

    random = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    uid = await client.get_uid_for_pubkey(random.public_key)
    assert uid is None






