# system
import copy
import argparse

# NLP
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
import bittensor



class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.penalty = penalty

    def __call__(self, input_ids, logits):
        input_ids_set = set(tuple(x) for x in input_ids.tolist())
        for token_id in input_ids_set:
            logits[:, token_id] /= self.penalty
        return logits

class PromptingValidator(torch.nn.Module):

    def __init__(self,
                config: 'bittensor.config' = None,
                model_name: str = None,
                min_tokens=0, 
                max_tokens=16, 
                temperature=1.0, 
                top_p=1.0, 
                logprobs=0, 
                repetition_penalty=1.0
                ):
        super(PromptingValidator, self).__init__()
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.logprobs = logprobs
        self.repetition_penalty = repetition_penalty


    def forward(self, prompt):
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

        if (
            config.min_new_tokens is not None
            and config.min_new_tokens > 0
            and config.eos_token_id is not None
        ):
            processor.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=input_length,
                    min_new_tokens=config.min_new_tokens,
                    eos_token_id=config.eos_token_id,
                )
            )

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
            token_logprobs = torch.log(tokens, token_logprobs)

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

    # model_name: str = None,
    # min_tokens=0, 
    # max_tokens=16, 
    # temperature=1.0, 
    # top_p=1.0, 
    # logprobs=0, 
    # repetition_penalty=1.0
    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.model_name', type=str, default='gpt2', help='Name of the model to use')
        parser.add_argument('--nucleus.min_tokens', type=int, default=1, help='Minimum number of tokens to generate')
        parser.add_argument('--nucleus.max_tokens', type=int, default=256, help='Maximum number of tokens to generate')
        parser.add_argument('--nucleus.temperature', type=float, default=1.0, help='Temperature for sampling')
        parser.add_argument('--nucleus.top_p', type=float, default=1.0, help='Top p for sampling')
        parser.add_argument('--nucleus.logprobs', type=int, default=10, help='Number of logprobs to return')
        parser.add_argument('--nucleus.repetition_penalty', type=float, default=1.0, help='Repetition penalty for sampling')


    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass