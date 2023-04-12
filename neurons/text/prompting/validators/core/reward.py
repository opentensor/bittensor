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
from torch import nn
from typing import List,Dict
from transformers import AutoModel, AutoTokenizer, LlamaConfig
import math

import transformers


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


def prepare_llama_tokenizer_and_embedding(
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        special_tokens_dict: Dict = dict(pad_token=DEFAULT_PAD_TOKEN),
):
    """prepare llama tokenizer and embedding.

    """

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    return tokenizer


def smart_tokenizer_and_embedding_resize(
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        special_tokens_dict: Dict = dict(pad_token=DEFAULT_PAD_TOKEN),
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

    if tokenizer.pad_token is None:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

        if isinstance(model, RewardModel):
            model = model.get_base_model()

        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            # output_embeddings = model.model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            # output_embeddings[-num_new_tokens:] = output_embeddings_avg


class RewardModel(nn.Module):
    def __init__(self, model_path=None, config=None, lora_rank=0, lora_train_bias: str = 'none') -> None:
        super().__init__()
        if model_path is not None:
            self.model = AutoModel.from_pretrained(model_path)
        elif config is not None:
            self.model = AutoModel.from_config(config)
        else:
            self.model = AutoModel.from_config(LlamaConfig())

        self.value_head = nn.Linear(self.model.config.hidden_size, 1)
        self.value_head.weight.data.normal_(mean=0.0, std=1 / (self.model.config.hidden_size + 1))

        self.tokenizer = AutoTokenizer.from_pretrained('./llama_tokenizer')
        self.tokenizer = prepare_llama_tokenizer_and_embedding(self.tokenizer, self.model)
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reward(self, completions: List[str]) -> torch.FloatTensor:
        def reward_fn(samples):
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
                out.extend(rewards)

            return out

        with torch.no_grad():
            rewards = [reward_fn([completion]) for completion in completions]
            for completion, reward in zip(completions, rewards):
                print(completion)
                print(reward)
            # Convert the list of single-element lists containing torch tensors to a 1D torch tensor.
            rewards_tensor = torch.cat(rewards).view(-1)
            return rewards_tensor
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask
        )

        hidden_states = transformer_outputs['last_hidden_state']

        values = self.value_head(hidden_states)[:, :-1]
        value = values.mean(dim=1).squeeze(1)    # ensure shape is (B)
        return value
