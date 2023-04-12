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

#### The code is modified from trlX
import json
import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List


class RewardModel(nn.Module):
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = self.model.config
        self.neox = "neox" in self.config.model_type
        # gpt-neo models have hidden_size instead of n_embd
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = self.model.gpt_neox if hasattr(self.model, "gpt_neox") else self.model.transformer
        dtype = self.config.torch_dtype if hasattr(self.config, "torch_dtype") is not None else torch.float32
        dtype = torch.float16 if dtype == "float16" else torch.float32
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False, dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.PAD_ID = self.tokenizer.pad_token_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def reward( self, completions: List[str] ) -> torch.FloatTensor:
        def reward_fn( samples ):
            samples = [s + self.tokenizer.eos_token for s in samples]
            input = self.tokenizer(samples, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(
                self.device
            )

            mbs = 24
            out = []
            for i in range(math.ceil(len(samples) / mbs)):
                batch_ixs = slice(i * mbs, (i + 1) * mbs)
                input_ids = input.input_ids[batch_ixs]
                rewards = self.forward(input_ids)
                out.extend(rewards.cpu().tolist())

            return out
        
        with torch.no_grad():
            rewards = [reward_fn([completion]) for completion in completions]
            for completion, reward in zip(completions, rewards):
                print(completion)
                print(reward)
            return torch.tensor(rewards, dtype=torch.float32)
        
    def forward(
        self,
        input_ids=None,
    ):
        states = self.model.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(rewards, 1, ends).squeeze(-1)
        return returns