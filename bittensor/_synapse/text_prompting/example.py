from typing import List, Dict

import bittensor
bittensor.logging(debug=True)
# import openai


# openai.api_key = "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"

# Create a synapse that returns zeros.
class Synapse(bittensor.TextPromptingSynapse):
    def _priority(self, forward_call: "bittensor.TextPromptingForwardCall") -> float:
        return 0.0

    def _blacklist(self, forward_call: "bittensor.TextPromptingForwardCall") -> bool:
        return False

    def forward(self, messages: List[Dict[str, str]]) -> str:

        import openai
        openai.api_key = "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"
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"},
                {"role": "assistant", "content": "The World Series was played in Arlington, Texas."},
                {"role": "user", "content": "Who was the MVP?"}
            ]
        )
        return resp['choices'][0]['message']['content']

# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()

# Create a local endpoint receptor grpc connection.
local_endpoint = bittensor.endpoint(
    version=bittensor.__version_as_int__,
    uid=0,
    ip="127.0.0.1",
    ip_type=4,
    port=9090,
    hotkey=wallet.hotkey.ss58_address,
    coldkey=wallet.coldkeypub.ss58_address,
    modality=0,
)

metagraph = None # Allow offline testing with unregistered keys.
axon = bittensor.axon(wallet=wallet, port=9090, ip="127.0.0.1", metagraph=metagraph)

synapse = Synapse()
axon.attach(synapse=synapse)
axon.start()

batch_size = 4
sequence_length = 32
# Create a text_prompting module and call it.
module = bittensor.text_prompting( endpoint = local_endpoint, wallet = wallet )
response = module.forward(
    roles=['user', 'assistant'],
    messages = [{ "user": "Human", "content": "hello"}],
    timeout=1e6
)


# # Delete objects.
del axon
del synapse
del module
