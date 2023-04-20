import torch
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
class TS2SSynapse( bittensor.TextSeq2SeqSynapse ):
    def priority(self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> float:
        return 0.0
    
    def blacklist(self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> torch.FloatTensor:
        return False
    
    def forward(self, forward_call: 'bittensor.TextSeq2SeqBittensorCall' ) -> 'bittensor.TextSeq2SeqBittensorCall':
        forward_call.generations = torch.zeros( forward_call.text_prompt.shape[0], forward_call.text_prompt.shape[1], bittensor.__network_dim__ )
        return forward_call
    
# Create a synapse that returns zeros.
class TLHSSynapse( bittensor.TextLastHiddenStateSynapse ):
    def priority(self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> float:
        return 0.0
    
    def blacklist(self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> torch.FloatTensor:
        return False
    
    def forward(self, forward_call: 'bittensor.TextLastHiddenStateForwardCall' ) -> bittensor.TextLastHiddenStateForwardCall:
        forward_call.hidden_states = torch.zeros( forward_call.text_inputs.shape[0], forward_call.text_inputs.shape[1], bittensor.__network_dim__ )
        return forward_call
    
config = bittensor.config()
config.merge( TLHSSynapse.config() )
config.merge( TS2SSynapse.config() )
print ( config )

# Create a synapss.
synapse_s2s = TS2SSynapse( config = config )
synapse_tlhs = TLHSSynapse( config = config )

# Create an axon and attach the synapse.
axon = bittensor.axon( wallet = wallet, port = 9090, ip = '127.0.0.1' )
axon.attach( synapse = synapse_s2s )
axon.attach( synapse = synapse_tlhs )
axon.start()

# Create a modules.
module_s2s = bittensor.text_seq2seq( endpoint = local_endpoint, wallet = wallet )
module_tlhs = bittensor.text_last_hidden_state( endpoint = local_endpoint, wallet = wallet )

# Call the s2s module.
s2s_response = module_s2s( 
    text_prompt = torch.ones( ( 1, 1 ), dtype = torch.long ),
    timeout = 1,
    topk = 50, 
    num_to_generate = 256,
    num_beams = 5,
    no_repeat_ngram_size  = 2,
    early_stopping = False,
    num_return_sequences = 1,
    do_sample = False,
    top_p = 0.95, 
    temperature = 1.0,
    repetition_penalty = 1.0,
    length_penalty = 1.0,
    max_time = 150,
    num_beam_groups = 1,
    text_prompt_serializer_type = bittensor.proto.Serializer.MSGPACK,
    generations_serializer_type = bittensor.proto.Serializer.MSGPACK
)

# Call the tlhs module.
tlhs_response = module_tlhs( 
    text_inputs = torch.ones( ( 3, 4 ), dtype = torch.long ), 
    mask = torch.rand( ( 3, 4 ) ) > 0.5, 
    timeout = 1 
)

# Delete objects.
del axon
del synapse_s2s
del synapse_tlhs
del module_s2s
del module_tlhs