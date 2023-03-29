import torch
from transformers import AutoModelForCausalLM

import bittensor

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
class Synapse( bittensor.TextLastHiddenStateSynapse ):

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    def _priority(self, forward_call: "bittensor.TextLastHiddenStateBackwardCall") -> float:
        return 0.0

    def _blacklist( self, forward_call: "bittensor.TextLastHiddenStateForwardCall") -> bool:
        return False

    def forward( 
            self, 
            forward_call: "bittensor.TextLastHiddenStateForwardCall"
        ) -> torch.Tensor:
        """ fills in the hidden_states of the forward call.
            Args:
                forward_call (:obj:`torch.LongTensor`, `required`):
                    .
            Returns:
                hidden_states (:obj:`torch.FloatTensor`, `required`):
                    hidden states of the last layer of the model.
        """
        print("in terminal child forward()")
        outputs = self.model(input_ids=forward_call.text_inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, :, :bittensor.__network_dim__]
        return last_hidden_state

    def backward( 
            self, 
            text_inputs: torch.LongTensor,
            hidden_states: torch.FloatTensor,
            hidden_states_grads: torch.FloatTensor,
        ):
        pass


# subtensor = bittensor.subtensor( )
# metagraph = bittensor.metagraph ().sync( netuid = 3, subtensor = subtensor )

# Create axon.
metagraph = None #bittensor.metagraph()
axon = bittensor.axon(
    wallet = wallet,
    metagraph = metagraph,
    port = 9090,
    ip = '127.0.0.1'
)

# Create a synapse and attach it to an axon.
synapse = Synapse()
axon.attach( synapse = synapse )
axon.start()

# Create a text_last_hidden_state module and call it.
module = bittensor.text_last_hidden_state( wallet = wallet, endpoint = local_endpoint ) # Dendrite
response = module.forward( 
    text_inputs = torch.ones( ( 3, 4 ), dtype = torch.long ), 
    mask = torch.rand( ( 3, 4 ) ) > 0.5, 
    timeout = 1000
)
# response = module.backward(
#     text_inputs = torch.ones( ( 3, 4 ), dtype = torch.long ),
#     hidden_states = response.hidden_states,
#     hidden_states_grads = response.hidden_states,
#     mask = torch.rand( ( 3, 4 ) ) > 0.5,
# )

# Delete objects.
del axon
del synapse
del module
