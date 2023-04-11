from time import sleep

import torch
from transformers import AutoModelForCausalLM

import bittensor

bittensor.logging(debug=True)


class Synapse(bittensor.TextCausalLMNextSynapse): # Create and inherit from the Synapse class

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    def _priority(self, forward_call: "bittensor.TextCausalLMNextForwardCall") -> float:
        return 0.0

    def _blacklist(self, forward_call: "bittensor.TextCausalLMNextForwardCall") -> bool:
        return False

    def forward(
        self, text_inputs: torch.Tensor
    ) -> torch.Tensor:

        outputs = self.model(input_ids=text_inputs, output_hidden_states=False)

        return outputs.logits

wallet = bittensor.wallet(_mock=True)
# Use a mock metagraph because we want to test a local endpoint and not register on the network.
metagraph = bittensor.metagraph(_mock=True)

axon = bittensor.axon(wallet=wallet, metagraph=metagraph, port=9090, ip="127.0.0.1")

synapse = Synapse()
axon.attach(synapse=synapse)
axon.start()

print("Serving axon")
while True: # Adjust this loop to write a miner
    sleep(0.25)

#
# ## Delete objects.
# del axon
# del synapse
