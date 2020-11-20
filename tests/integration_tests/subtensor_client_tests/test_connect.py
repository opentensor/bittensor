from bittensor.subtensor import WSClient
from subtensorinterface import Keypair
from loguru import logger
import pytest
import unittest

logger.remove() # Shut up loguru


socket = "pop.bittensor.com:9944"
keypair = Keypair.create_from_uri('//Alice')
client = WSClient(socket, keypair)


@pytest.mark.asyncio
async def test_connect():
    client.connect()
    result = await client.is_connected()
    assert result == True
