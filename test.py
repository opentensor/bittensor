import bittensor
import threading
import os
import torch
import time
import torch.multiprocessing as mp 
import queue
from bittensor.experimental import axon_multiprocessing

axon = axon_multiprocessing.Axon()
axon.start()


wallet = bittensor.wallet.Wallet(
    path = "~/.bittensor/wallets/",
    name = "test",
    hotkey = "default"
)

tensor = torch.tensor([[1,2]], dtype=torch.int64)
print (tensor)

neuron = bittensor.proto.Neuron(
    address = '127.0.0.1',
    port = 8091,
    public_key = 'asdlmasskdmlakmsda'
)
print(neuron)

config = bittensor.dendrite.Dendrite.default_config()
config.receptor.timeout = 10
config.receptor.do_backoff = False
dendrite = bittensor.dendrite.Dendrite(
    config = config,
    wallet = wallet
)
print (bittensor.config.Config.toString(config))

def producer():
  while True:
    time.sleep(1)
    print ('Sending message...')
    responses, codes = dendrite.forward_text(
        [neuron],
        [tensor]
    )
    print ('Response code:', codes[0])
    break

def consumer():
  while True:
    try:
      print ('waiting on axon queue')
      pong, pubkey, inputs, modality = axon.forward_queue.get(block=True, timeout=3.0)
      print ('pipe', pong, 'key',  pubkey, 'inputs', inputs, 'modality', modality)
      print (inputs.shape)
      pong.send( torch.zeros([1,2, 512]) )
    except queue.Empty:
      print ('done waiting')
    break

print ('Create consumer...')
x = threading.Thread( target=consumer, daemon=True)
x.start()
print ('Started consumer.')

print ('Started producer.')
y = threading.Thread( target=producer, daemon=True)
y.start()
print ('Started producer.')

print ('join consumer')
x.join()

print ('join producer')
y.join()

print ('done')