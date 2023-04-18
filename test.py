import asyncio
import bittensor
wallet = bittensor.wallet()
meta = bittensor.subtensor().metagraph( 1 )
dendrites = [ bittensor.text_prompting( keypair = wallet, endpoint = meta.endpoint_objs[uid] ) for uid in meta.uids ]
bittensor.logging.set_trace()


async def run():
    calls = []
    for uid in meta.uids:
        calls.append( dendrites[uid].async_forward( roles = ['user'], messages = ['what is the capital of Austin'], timeout = 1) ) 
    return await asyncio.gather(*calls)

loop = asyncio.get_event_loop()
results = loop.run_until_complete( run() )
print( [res.completion for res in results] )