import torch
import bittensor

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
class Synapse( bittensor.TextSeq2SeqSynapse ):
    def priority(self, forward_call: 'bittensor.TextSeq2SeqForwardCall' ) -> float:
        return 0.0
    
    def blacklist(self, forward_call: 'bittensor.TextSeq2SeqForwardCall' ) -> torch.FloatTensor:
        return False
    
    def forward(self, forward_call: 'bittensor.TextSeq2SeqForwardCall' ) -> 'bittensor.TextSeq2SeqForwardCall':
        forward_call.generations = torch.zeros( forward_call.text_prompt.shape[0], forward_call.text_prompt.shape[1], bittensor.__network_dim__ )
        return forward_call
    
# Create a synapse and attach it to an axon.
synapse = Synapse()
axon = bittensor.axon( wallet = wallet, port = 9090, ip = '127.0.0.1' )
axon.attach( synapse = synapse )
axon.start()

# Create a text_last_hidden_stsate module and call it.
module = bittensor.text_seq2seq( endpoint = local_endpoint, wallet = wallet )
response = module( 
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

# Delete objects.
del axon
del synapse