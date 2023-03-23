import torch
from transformers import AutoModelForCausalLM

import bittensor

bittensor.logging(debug=True)


# Create a synapse that returns zeros.
class Synapse(bittensor.TextCausalLMNextSynapse):

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    def _priority(self, forward_call: "bittensor.TextCausalLMNextForwardCall") -> float:
        return 0.0

    def _blacklist(self, forward_call: "bittensor.TextCausalLMNextForwardCall") -> bool:
        return False

    def forward(
        self, forward_call: "bittensor.TextCausalLMNextForwardCall"
    ) -> torch.Tensor:
        outputs = self.model(input_ids=forward_call.text_inputs, output_hidden_states=False)
        # TODO: topk here? (or in post_process_...?
        return outputs.logits


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
# Create a text_causallm_next module and call it.
module = bittensor.text_causal_lm_next(endpoint=local_endpoint, wallet=wallet)
response = module.forward(
    text_inputs=torch.ones((batch_size, sequence_length), dtype=torch.long),
    timeout=1e6
)


# # Delete objects.
del axon
del synapse
del module
