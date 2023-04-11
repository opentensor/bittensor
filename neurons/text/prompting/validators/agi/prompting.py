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

import copy
import torch
import argparse
import bittensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def __call__(self, input_ids, logits):
        input_ids_set = set(tuple(x) for x in input_ids.tolist())
        for token_id in input_ids_set:
            logits[:, token_id] /= self.penalty
        return logits

class PromptingModel(torch.nn.Module):

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--prompting.model_name', type=str, default='gpt2', help='Name of the model to use')
        parser.add_argument('--prompting.min_tokens', type=int, default=1, help='Minimum number of tokens to generate')
        parser.add_argument('--prompting.max_tokens', type=int, default=256, help='Maximum number of tokens to generate')
        parser.add_argument('--prompting.temperature', type=float, default=1.0, help='Temperature for sampling')
        parser.add_argument('--prompting.top_p', type=float, default=1.0, help='Top p for sampling')
        parser.add_argument('--prompting.logprobs', type=int, default=10, help='Number of logprobs to return')
        parser.add_argument('--prompting.repetition_penalty', type=float, default=1.0, help='Repetition penalty for sampling')

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def __init__(
            self,
            config: 'bittensor.config' = None,
            model_name: str = None,
            min_tokens = 0, 
            max_tokens = 256, 
            temperature = 1.0, 
            top_p = 1.0, 
            logprobs = 0, 
            repetition_penalty = 1.0
        ):
        """ Initializes the model.
            Args:
                config (:obj:`bittensor.Config`, `optional`):
                    bittensor config object.
                model_name (:obj:`str`, `optional`):
                    Name of the model to use.
                min_tokens (:obj:`int`, `optional`, defaults to 0):
                    The minimum number of tokens to generate.
                max_tokens (:obj:`int`, `optional`, defaults to 16):
                    The maximum number of tokens to generate.
                temperature (:obj:`float`, `optional`, defaults to 1.0):
                    The value used to module the next token probabilities.
                top_p (:obj:`float`, `optional`, defaults to 1.0):
                    If set to float < 1, only the most probable tokens with probabilities that add up to
                    :obj:`top_p` or higher are kept for generation.
                logprobs (:obj:`int`, `optional`, defaults to 0):
                    If set to 1, the model returns the log probabilities of the generated tokens.
                repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                    The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                    <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details."""
        super(PromptingModel, self).__init__()

        # Setup config.
        if config is None: config = PromptingModel.config()
        if model_name is not None: config.prompting.model_name = model_name
        if min_tokens is not None: config.prompting.min_tokens = min_tokens
        if max_tokens is not None: config.prompting.max_tokens = max_tokens
        if temperature is not None: config.prompting.temperature = temperature
        if top_p is not None: config.prompting.top_p = top_p
        if logprobs is not None: config.prompting.logprobs = logprobs
        if repetition_penalty is not None: config.prompting.repetition_penalty = repetition_penalty
        PromptingModel.check_config( config )
        self.config = copy.deepcopy( config )

        # Setup local vars.
        self.model_name = self.config.prompting.model_name
        self.min_tokens = self.config.prompting.min_tokens
        self.max_tokens = self.config.prompting.max_tokens
        self.temperature = self.config.prompting.temperature
        self.top_p = self.config.prompting.top_p
        self.logprobs = self.config.prompting.logprobs
        self.repetition_penalty = self.config.prompting.repetition_penalty

        # Setup model.
        self.model = AutoModelForCausalLM.from_pretrained( self.model_name ) 
        self.tokenizer = AutoTokenizer.from_pretrained( self.model_name )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(
            self, 
            prompt: str 
        ):
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate tokens using the generate_tokens method
        tokens_tensor, token_logprobs_tensor, top_tokens_tensor, top_logprobs_tensor, status_list = self.generate_tokens(
            input_ids,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            logprobs=self.logprobs,
            repetition_penalty=1.12,
        )

        # Decode the generated tokens back to text
        generated_text = self.tokenizer.batch_decode(tokens_tensor, skip_special_tokens=True)

        return generated_text, tokens_tensor, token_logprobs_tensor, top_tokens_tensor, top_logprobs_tensor, status_list
    
    def _logits_processor(self, config, input_length, repetition_penalty=1.0):
        processor = LogitsProcessorList()

        # if (
        #     config.min_new_tokens is not None
        #     and config.min_new_tokens > 0
        #     and config.eos_token_id is not None
        # ):
        #     processor.append(
        #         MinNewTokensLengthLogitsProcessor(
        #             prompt_length_to_skip=input_length,
        #             min_new_tokens=config.min_new_tokens,
        #             eos_token_id=config.eos_token_id,
        #         )
        #     )

        if (
            config.temperature is not None
            and config.temperature > 0
            and config.temperature != 1.0
        ):
            processor.append(TemperatureLogitsWarper(config.temperature))

        if config.top_p is not None and config.top_p > 0 and config.top_p < 1:
            processor.append(TopPLogitsWarper(config.top_p))

                # Add processor for repetition penalty.
        if repetition_penalty != 1.0:
            processor.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))


        return processor

    def generate_tokens(
        self,
        input_ids,
        min_tokens=0,
        max_tokens=16,
        temperature=1.0,
        top_p=1.0,
        logprobs=0,
        repetition_penalty=1.0,
    ):
        """Generates tokens using the model.
            Args:
                input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                    The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                    it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
                min_tokens (:obj:`int`, `optional`, defaults to self.min_tokens):
                    The minimum number of tokens to generate.
                max_tokens (:obj:`int`, `optional`, defaults to self.max_tokens):
                    The maximum number of tokens to generate.
                temperature (:obj:`float`, `optional`, defaults to self.temperature):
                    The value used to module the next token probabilities.
                top_p (:obj:`float`, `optional`, defaults to self.top_p):
                    If set to float < 1, only the most probable tokens with probabilities that add up to
                    :obj:`top_p` or higher are kept for generation.
                logprobs (:obj:`int`, `optional`, defaults to self.logprobs):
                    If set to 1, the model returns the log probabilities of the generated tokens.
                repetition_penalty (:obj:`float`, `optional`, defaults to self.repetition_penalty):
                    The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                    <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            Return:
                tokens_tensor (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
                    The generated tokens. The second dimension (sequence_length) is either equal to :obj:`max_length` or
                    shorter if all batches finished early due to the :obj:`eos_token_id`.
                token_logprobs_tensor (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: 
                    The log probabilities of the generated tokens.
                top_tokens_tensor (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: 
                    The top tokens of the generated tokens.
                top_logprobs_tensor (:obj:`torch.FloatTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: 
                    The top log probabilities of the generated tokens.
                status_list (:obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: 
                    The status of the generation. 0: continue, 1: stop.
        """
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[-1]

        config = self.model.config
        config = copy.deepcopy(config)

        processor = self._logits_processor(config, input_length, repetition_penalty=repetition_penalty)

        tokens_list = []
        token_logprobs_list = []
        top_tokens_list = []
        top_logprobs_list = []
        status_list = []

        for token_index in range(max_tokens):
            inputs = self.model.prepare_inputs_for_generation(input_ids)
            outputs = self.model(**inputs, return_dict=True)

            logits = outputs.logits[:, -1, :]
            with torch.inference_mode():
                logits = processor(input_ids, logits)
            probs = torch.nn.functional.softmax(logits, dim=-1)

            tokens = torch.argmax(probs, dim=-1)[:, None]
            token_logprobs = torch.gather(probs, 1, tokens)
            #token_logprobs = torch.log(tokens, token_logprobs)
            token_logprobs = torch.log(tokens)

            tokens_list.append(tokens)
            token_logprobs_list.append(token_logprobs)

            if logprobs > 0:
                top_logprobs, top_tokens = torch.topk(probs, logprobs, dim=-1)
                top_logprobs = torch.log(top_tokens, top_logprobs)
                top_tokens_list.append(top_tokens)
                top_logprobs_list.append(top_logprobs)
            
            input_ids = torch.cat([input_ids, tokens], dim=-1)

            # Check if any input has generated the minimum number of tokens
            if token_index + 1 >= min_tokens:
                eos_mask = tokens == config.eos_token_id
                if eos_mask.any():
                    status_list.append(eos_mask)
                    status = torch.full((batch_size,), False, dtype=torch.bool)
                    status = torch.where(eos_mask.squeeze(1), True, status)
                    status_list.append(status)

                    if status.all():
                        break

            tokens_tensor = torch.cat(tokens_list, dim=-1)
            token_logprobs_tensor = torch.cat(token_logprobs_list, dim=-1)

            if logprobs > 0:
                top_tokens_tensor = torch.cat(top_tokens_list, dim=-1)
                top_logprobs_tensor = torch.cat(top_logprobs_list, dim=-1)
            else:
                top_tokens_tensor = None
                top_logprobs_tensor = None

            return tokens_tensor, token_logprobs_tensor, top_tokens_tensor, top_logprobs_tensor, status_list