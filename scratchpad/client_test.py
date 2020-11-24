from bittensor.subtensor import SubstrateWSInterface
from bittensor.subtensor import Keypair
from bittensor.subtensor import WSClient
import asyncio


kp = Keypair.create_from_uri('//Alice')
client = WSClient("localhost:9944", kp)


async def test():
    client.connect()

    await client.is_connected()

    print("DONE")


loop = asyncio.get_event_loop()
# loop.create_task(test())
loop.run_until_complete(test())

