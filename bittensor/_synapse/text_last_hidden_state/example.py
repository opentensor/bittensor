import torch
import bittensor
import call

bittensor.logging(debug=True)

# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()

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

    def priority(self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> float:
        return 0.0
    
    def blacklist(self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> bool:
        return False
    
    def forward( 
            self, 
            text_inputs: torch.LongTensor,
            hotkey: str,
        ) -> torch.FloatTensor:
        """ fills in the hidden_states of the forward call.
            Args:
                text_inputs (:obj:`torch.LongTensor`, `required`):
                    tokenized text inputs.
                hotkey (:obj:`str`, `required`):
                    hotkey of the calling neuron
            Returns:
                hidden_states (:obj:`torch.FloatTensor`, `required`):
                    hidden states of the last layer of the model.
        """    
        return torch.zeros( text_inputs.shape[0], text_inputs.shape[1], bittensor.__network_dim__ )
    
    def backward( 
            self, 
            text_inputs: torch.LongTensor,
            hidden_states: torch.FloatTensor,
            hidden_states_grads: torch.FloatTensor,
        ):
        pass
    
# Create a synapse and attach it to an axon.
synapse = Synapse( wallet = wallet )
axon = bittensor.axon( wallet = wallet, port = 9090, ip = '127.0.0.1' )
axon.attach( synapse = synapse )
axon.start()

# Create a text_last_hidden_state module and call it.
module = bittensor.text_last_hidden_state( wallet = wallet, endpoint = local_endpoint )
response = module.forward( 
    text_inputs = torch.ones( ( 3, 4 ), dtype = torch.long ), 
    mask = torch.rand( ( 3, 4 ) ) > 0.5, 
    timeout = 1000
)
response = module.backward( 
    text_inputs = torch.ones( ( 3, 4 ), dtype = torch.long ), 
    hidden_states = response.hidden_states,
    hidden_states_grads = response.hidden_states,
    mask = torch.rand( ( 3, 4 ) ) > 0.5, 
)

# Delete objects.
del axon
del synapse
del module
