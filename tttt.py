import bittensor
import torch
import time 
import psutil
import argparse



parser = argparse.ArgumentParser( description=f"speed", usage="python3 speed <command args>", add_help=True)


graph = bittensor.metagraph().load()
wallet = bittensor.wallet(name = 'const', hotkey = 'Tiberius')
dend = bittensor.dendrite( wallet = wallet ) 
endpoints = graph.endpoint_objs[:10]
inputs = torch.tensor( [10, 20], dtype = torch.int64) 


import time 
import psutil
import tqdm 
import random
start_time = time.time()
io_1 = psutil.net_io_counters()
start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

n_steps = 100
chunk_size = 10
n_queried = 10
timeout = 12

a, b, c = dend.text( endpoints=endpoints, synapses=[bittensor.synapse.TextCausalLMNext()], inputs=inputs, timeout = 4)
print (sum([bb.item() == 1 for bb in b])/len(b))

io_2 = psutil.net_io_counters()
total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv
end_time = time.time()

total_success = sum([sum(ri) for ri in results])
total_sent = n_queried n_steps
total_failed = total_sent - total_success
total_seconds =  end_time - start_time


print ('\nTotal:', total_sent, 
       '\nSteps:', n_steps, 
       '\nQueries:', n_queried,
       '\nTimeout:', timeout,
       '\nSuccess:', total_success, 
       '\nFailures:', total_failed, 
       '\nRate:', total_success/total_sent, 
       '\nSize:', list(inputs.shape), 
       '\nSeconds:', total_seconds, '/s',
       '\nQ/sec:', total_success/total_seconds, '/s',
       '\nTotal Upload:', get_size( total_bytes_sent ),
       '\nTotal Download:', get_size( total_bytes_recved ),
       '\nUpload Speed:', get_size( total_bytes_sent / total_seconds), "/s",
       '\nDownload Speed:', get_size( total_bytes_recved / total_seconds), "/s")
