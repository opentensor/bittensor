from bittensor.subtensor import WSClient
from subtensorinterface import Keypair
from loguru import logger
import pytest
import unittest

# logger.remove() # Shut up loguru


socket = "localhost:9944"
keypair = Keypair.create_from_uri('//Alice')
client = WSClient(socket, keypair)


@pytest.mark.asyncio
async def test_subscribe():
    client.connect()
    await client.is_connected()

    await client.subscribe("127.0.0.1", 666)



