# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel


class ChatGLMMiner(bittensor.HuggingFaceMiner):
    arg_prefix: str = "chat_glm"
    assistant_label: str = ""
    user_label: str = ""
    system_label: str = ""

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    def load_model(self):
        return AutoModel.from_pretrained(
            "THUDM/chatglm-6b", trust_remote_code=True, torch_dtype=torch.float16
        )

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history(messages)
        prompt = history[-1][-1]
        if len(history) == 1:
            history = []
        generation, history = self.model.chat(
            self.tokenizer,
            prompt,
            history,
            max_length=self.config.chat_glm.max_new_tokens,
            temperature=self.config.chat_glm.temperature,
            do_sample=self.config.chat_glm.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        bittensor.logging.debug(
            "Message: " + str(messages).replace("<", "-").replace(">", "-")
        )
        bittensor.logging.debug(
            "Generation: " + str(generation).replace("<", "-").replace(">", "-")
        )
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    ChatGLMMiner().run()
