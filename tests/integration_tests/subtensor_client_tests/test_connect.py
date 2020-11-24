from bittensor.subtensor import WSClient, Keypair
from loguru import logger
import pytest

logger.remove() # Shut up loguru




@pytest.mark.asyncio
async def test_connect_success():
    socket = "localhost:9944"
    keypair = Keypair.create_from_uri('//Alice')
    client = WSClient(socket, keypair)

    client.connect()
    result = await client.is_connected()
    assert result == True

@pytest.mark.asyncio
async def test_connect_failed():
    socket = 'localhost:9999'
    keypair = Keypair.create_from_uri('//Alice')
    client = WSClient(socket, keypair)

    client.connect()
    result = await client.is_connected()
    assert result == False
