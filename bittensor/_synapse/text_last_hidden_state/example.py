import torch
import bittensor
import asyncio
import grpc

channel = grpc.aio.insecure_channel(
    '127.0.0.1:9090',
    options=[('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
            ('grpc.keepalive_time_ms', 100000)])

#loop = asyncio.get_event_loop()
#loop.run_until_complete ( channel.close() )
# stub = bittensor.grpc.BittensorStub( channel )
del channel
# del stub

# Create a mock wallet.
wallet = bittensor.wallet( name = 'floppy', hotkey = '3')

# Create a local endpoint.
local_endpoint = bittensor.endpoint(
    version = bittensor.__version_as_int__,
    uid = 0,
    ip = '127.0.0.1',
    ip_type = 4,
    port = 9090,
    hotkey = wallet.hotkey.ss58_address,
    coldkey = wallet.coldkeypub.ss58_address,
    modality = 0
)    

receptor = bittensor.receptor( endpoint = local_endpoint, wallet = wallet )
# del receptor
# # Create a synapse that returns zeros.
# class Synapse(bittensor.TextLastHiddenStateSynapse):
#     def priority(self, hotkey: str, text_inputs: torch.FloatTensor, request: bittensor.ForwardTextLastHiddenStateRequest) -> float:
#         return 0.0
    
#     def blacklist(self, hotkey: str, text_inputs: torch.FloatTensor, request: bittensor.ForwardTextLastHiddenStateRequest) -> torch.FloatTensor:
#         return False
    
#     def forward(self, hotkey: str, text_inputs: torch.FloatTensor, request: bittensor.ForwardTextLastHiddenStateRequest) -> torch.FloatTensor:
#         return torch.zeros( text_inputs.shape[0], text_inputs.shape[1], bittensor.__network_dim__ )
    
# # Create a synapse and attach it to an axon.
# synapse = Synapse()
# axon = bittensor.axon( wallet = wallet, port = 9090, ip = '127.0.0.1' )
# axon.attach( synapse = synapse )

# Create a text_last_hidden_state module and call it.
#module = bittensor.text_last_hidden_state( endpoint = local_endpoint, wallet = wallet )
#response = module( text_inputs = torch.ones( ( 1, 1 ), dtype = torch.long ), timeout = 1 )
#print( response )

