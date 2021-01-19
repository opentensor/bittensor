from bittensor.subtensor.client import WSClient
from bittensor.subtensor.interface import Keypair
from loguru import logger
import pytest

logger.remove() # Shut up loguru

@pytest.mark.asyncio
async def test_weight_uids_for_uid():
    socket = "localhost:9944"
    keypair = Keypair.create_from_uri('//Alice')
    client = WSClient(socket, keypair)

    client.connect()
    await client.is_connected()

    result = await client.weight_uids_for_uid(88)

    assert result == "FSFD"
