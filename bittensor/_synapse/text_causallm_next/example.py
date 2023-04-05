import torch
from transformers import AutoModelForCausalLM

from typing import Any

import bittensor

bittensor.logging(debug=True)


# Create a synapse that returns zeros.
class Synapse(bittensor.TextCausalLMNextSynapse):

    # TODO: replace with an unset variable for user-defined model?
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

    def _priority(self, forward_call: "bittensor.TextCausalLMNextForwardCall") -> float:
        return 0.0

    def _blacklist(self, forward_call: "bittensor.TextCausalLMNextForwardCall") -> bool:
        return False

    def forward(self, inputs: Any) -> torch.Tensor:
        outputs = self.model(input_ids=inputs, output_hidden_states=False)
        return outputs.logits

    def backward(
        self, backward_call: "bittensor.TextCausalLMNextBackwardCall"
     ) -> torch.Tensor:
        return torch.zeros((16, 1, bittensor.__network_dim__))

"""
    # TODO: Call this properly and/or get correct function to get and then compress topk.
    # Option 1 Naive topk
    topk = forward_call.topk
    topk_values, indices = torch.topk(last_logits, topk)

    # Option 2 Unbiased topk (selected index k out of range...)
    # topk_values, indices = bittensor.unbiased_topk(last_logits, k=topk)

    # Option 3 Eugene's func
    # Where to get tokenizer req for this function? Do we need one?
    # topk_values = bittensor.topk_token_phrases(logits=last_logits, topk=topk)

    # Compact topk for sending over wire. (returns empty tensor...)
    # compact_topk_values = bittensor.compact_topk_token_phrases(topk_values)

    # Not sure if we need to return the logits or the activations.
    # topk_probs = torch.softmax(topk_values, dim=-1)

    # TODO: (De)compact on other end.
"""

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
response1 = module.forward(
    text_inputs=torch.ones((batch_size, sequence_length), dtype=torch.long),
    timeout=1e6
)
# import pdb
# pdb.set_trace()
# response2 = module.backward(
#
# )



# # Delete objects.
del axon
del synapse
del module
