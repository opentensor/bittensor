import bittensor
import json

bittensor.logging(debug=True)
# Create a mock wallet.
wallet = bittensor.wallet(name='prompting_testing', hotkey='default')

# Create a local endpoint receptor grpc connection.
local_endpoint = bittensor.endpoint(
    version=bittensor.__version_as_int__,
    uid=2,
    ip="127.0.0.1",
    ip_type=4,
    port=8091,
    hotkey=wallet.hotkey.ss58_address,
    coldkey=wallet.coldkeypub.ss58_address,
    modality=0,
)


# Create a text_prompting module and call it.
module = bittensor.text_prompting( endpoint = local_endpoint, wallet = wallet )
response = module.forward(
    messages = [json.dumps({ "role": "user", "content": "hello"})],
    timeout=1e6
)
print(response)


# # Delete objects.
del module
