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

#### NOTE(carro): This code is modified from trlX

import torch
import argparse
import bittensor

from torch import nn
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class RewardModel(nn.Module):

    def __init__( self, model_path: str, device: str, config: 'bittensor.config' = None):
        super().__init__()
        config = AutoConfig.from_pretrained( model_path )
        self.model = AutoModelForCausalLM.from_config( config )
        self.config = self.model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        if config is None: config = RewardModel.config()

        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.device = torch.device( device )
        self.transformer = self.model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def reward( self, full_completions: List[str],  comp: List[str], difference=False, shift =3) -> torch.FloatTensor:
        def reward_fn( samples ):
            if samples is None: return 0
            scores_list = []
            batch_size = 1
            for i in range(0, len(samples), batch_size):
                sub_samples = samples[i : i + batch_size]
                sub_samples = [
                    "<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples
                ]
                encodings_dict = self.tokenizer(
                    sub_samples,
                    truncation=False,
                    max_length=550,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = encodings_dict["input_ids"].to( self.device )
                attn_masks = encodings_dict["attention_mask"].to( self.device )
                input_ids = input_ids.repeat(2, 1)
                attn_masks = attn_masks.repeat(2, 1)
                with torch.no_grad():
                    sub_scores = self.forward(input_ids=input_ids.to( self.device ), attention_mask=attn_masks.to( self.device ))
                scores_list.append(sub_scores["chosen_end_scores"])
            scores = torch.cat(scores_list, dim=0).mean().item()
            return scores
        
        with torch.no_grad():
            full_rewards = [reward_fn([completion]) for completion in full_completions]
            if difference:
                comp_rewards = [reward_fn([completion]) for completion in comp]
                return torch.nn.functional.relu(torch.tensor(full_rewards, dtype=torch.float32)+shift) - torch.nn.functional.relu(torch.tensor(comp_rewards, dtype=torch.float32)+shift)
            else:
                for completion, f_reward in zip(full_completions, full_rewards):
                    print(completion)
                    print(f_reward)
                return torch.tensor(full_rewards, dtype=torch.float32)
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss based on truncated rewards (ignore padding)
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }
