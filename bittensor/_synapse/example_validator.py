import torch

import bittensor

bittensor.logging(debug=True)

# Create a mock wallet.
wallet = bittensor.wallet(_mock=True)

# Create a local endpoint receptor grpc connection.
local_endpoint = bittensor.endpoint(
    version=bittensor.__version_as_int__,
    uid=0,
    ip="127.0.0.1",
    ip_type=4,
    port=9090,
    hotkey=wallet.hotkey.ss58_address,
    coldkey=wallet.coldkeypub.ss58_address,
    modality=0, # text
)

batch_size = 4
sequence_length=32

# Create a text_causallm_next module and call it.
dendrite = bittensor.text_causal_lm_next(endpoint=local_endpoint, wallet=wallet)
response = dendrite.forward(
    text_inputs=torch.ones((batch_size, sequence_length), dtype=torch.long),
    timeout=12
)
# # Delete objects.
del dendrite
