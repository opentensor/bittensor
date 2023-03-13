import torch
import bittensor
import call

bittensor.logging(debug=True)

# Create a mock wallet.
wallet = bittensor.wallet( name = 'floppy', hotkey = '3')

# Create a local endpoint receptor grpc connection.
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

# Create a synapse that returns zeros.
class Synapse(bittensor.TextLastHiddenStateSynapse):
    def priority(self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> float:
        return 0.0
    
    def blacklist(self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> torch.FloatTensor:
        return False
    
    def forward(self, forward_call: 'bittensor.TextLastHiddenStateBittensorCall' ) -> bittensor.TextLastHiddenStateBittensorCall:
        forward_call.hidden_states = torch.zeros( forward_call.text_inputs.shape[0], forward_call.text_inputs.shape[1], bittensor.__network_dim__ )
        return forward_call
    
# Create a synapse and attach it to an axon.
synapse = Synapse()
axon = bittensor.axon( wallet = wallet, port = 9090, ip = '127.0.0.1' )
axon.attach( synapse = synapse )
axon.start()

# Create a text_last_hidden_state module and call it.

module = bittensor.text_last_hidden_state( endpoint = local_endpoint, wallet = wallet )
response = module.forward( text_inputs = torch.ones( ( 3, 4 ), dtype = torch.long ), mask = torch.rand( ( 3, 4 ) ) > 0.5, timeout = 1 )

# Delete objects.
del axon
del synapse