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


# General.
import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StabilityAIMiner(bittensor.HuggingFaceMiner):
    arg_prefix: str = "stabilityai"
    system_label: str = "<|SYSTEM|>:"
    assistant_label: str = "<|ASSISTANT|>:"
    user_label: str = "<|USER|>:"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        super(StabilityAIMiner, cls).add_args(parser)
        parser.add_argument(
            "--stabilityai.model_size",
            type=int,
            choices=[3, 7],
            default=7,
            help="Run the 3B or 7B model.",
        )
        parser.add_argument(
            "--stabilityai.suffix",
            type=str,
            default=None,
            help="The suffix that comes after a completion of inserted text.",
        )
        parser.add_argument(
            "--stabilityai.num_return_sequences",
            type=int,
            default=1,
            help="Description of num_return_sequences",
        )
        parser.add_argument(
            "--stabilityai.num_beams",
            type=int,
            default=1,
            help="Description of num_beams",
        )
        parser.add_argument(
            "--stabilityai.stopping_criteria",
            type=str,
            default="stop",
            help="Description of stopping_criteria",
        )

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-{}b".format(
                self.config.stabilityai.model_size
            ),
            use_auth_token=self.config.stabilityai.api_key,
        )

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-{}b".format(
                self.config.stabilityai.model_size
            ),
            use_auth_token=self.config.stabilityai.api_key,
            torch_dtype=torch.float16,
        ).cuda()

        return pipeline(
            "text-generation",
            model,
            tokenizer=self.tokenizer,
            device=0,
            max_new_tokens=self.config.stabilityai.max_new_tokens,
            num_return_sequences=self.config.stabilityai.num_return_sequences,
            num_beams=self.config.stabilityai.num_beams,
            do_sample=self.config.stabilityai.do_sample,
            temperature=self.config.stabilityai.temperature,
            top_p=self.config.stabilityai.top_p,
            top_k=self.config.stabilityai.top_k,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history(messages)
        return (
            self.model(history)[0]["generated_text"]
            .split(":")[-1]
            .replace(str(history), "")
        )


if __name__ == "__main__":
    bittensor.utils.version_checking()
    StabilityAIMiner().run()
