import argparse
import bittensor
from typing import List, Dict
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LlamaCppMiner(bittensor.BasePromptingMiner):
    @classmethod
    def check_config(cls, config: 'bittensor.Config'):
        pass

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        parser.add_argument( '--llama.model_path', type=str, required=True, help='Path of LlamaCpp model to load' )
        parser.add_argument( '--llama.lora_base', type=str, help='Path to the Llama LoRA base model.' )
        parser.add_argument( '--llama.lora_path', type=str, help='Path to the Llama LoRA.' )
        parser.add_argument( '--llama.n_ctx', type=int, default=512, help='Token context window.' )
        parser.add_argument( '--llama.n_parts', type=int, default=-1, help='Number of parts to split the model into.' )
        parser.add_argument( '--llama.seed', type=int, default=-1, help='Seed for model.' )
        parser.add_argument( '--llama.f16_kv', action='store_true', default=True, help='Use half-precision for key/value cache.' )
        parser.add_argument( '--llama.logits_all', action='store_true', default=False, help='Return logits for all tokens.' )
        parser.add_argument( '--llama.vocab_only', action='store_true', default=False, help='Only load the vocabulary, no weights.' )
        parser.add_argument( '--llama.use_mlock', action='store_true', default=False, help='Force system to keep model in RAM.')
        parser.add_argument( '--llama.n_threads', type=int, help='Number of threads to use.' )
        parser.add_argument( '--llama.n_batch', type=int, default=8, help='Number of tokens to process in parallel.' )
        parser.add_argument( '--llama.suffix', type=str, help='A suffix to append to the generated text.' )
        parser.add_argument( '--llama.max_tokens', type=int, default=100, help='The maximum number of tokens to generate.' )
        parser.add_argument( '--llama.temperature', type=float, default=0.8, help='The temperature to use for sampling.' )
        parser.add_argument( '--llama.top_p', type=float, default=0.95, help='The top-p value to use for sampling.' )
        parser.add_argument( '--llama.logprobs', type=int, help='The number of logprobs to return.' )
        parser.add_argument( '--llama.echo', action='store_true', default=False, help='Whether to echo the prompt.' )
        parser.add_argument( '--llama.stop', type=str, nargs='+', default=[], help='A list of strings to stop generation when encountered.' )
        parser.add_argument( '--llama.repeat_penalty', type=float, default=1.1, help='The penalty to apply to repeated tokens.' )
        parser.add_argument( '--llama.top_k', type=int, default=40, help='The top-k value to use for sampling.' )
        parser.add_argument( '--llama.last_n_tokens_size', type=int, default=64, help='The number of tokens to look back when applying the repeat_penalty.' )
        parser.add_argument( '--llama.use_mmap', action='store_true', default=True, help='Whether to keep the model loaded in RAM.' )
        parser.add_argument( '--llama.streaming', action='store_true', default=False, help='Whether to stream the results, token by token.' )
        parser.add_argument( '--llama.verbose', action='store_true', default=False, help='Verbose output for LlamaCpp model.' )
        parser.add_argument( '--llama.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( '--llama.system_prompt', type=str, help='What prompt to replace the system prompt with' )

    def __init__(self):
        super(LlamaCppMiner, self).__init__()

        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()] )
        self.llm = LlamaCpp(
            model_path=self.config.llama.model_path,
            lora_base=self.config.llama.lora_base,
            lora_path=self.config.llama.lora_path,
            n_ctx=self.config.llama.n_ctx,
            n_parts=self.config.llama.n_parts,
            seed=self.config.llama.seed,
            f16_kv=self.config.llama.f16_kv,
            logits_all=self.config.llama.logits_all,
            vocab_only=self.config.llama.vocab_only,
            use_mlock=self.config.llama.use_mlock,
            n_threads=self.config.llama.n_threads,
            n_batch=self.config.llama.n_batch,
            suffix=self.config.llama.suffix,
            max_tokens=self.config.llama.max_tokens,
            temperature=self.config.llama.temperature,
            top_p=self.config.llama.top_p,
            logprobs=self.config.llama.logprobs,
            echo=self.config.llama.echo,
            stop=self.config.llama.stop,
            repeat_penalty=self.config.llama.repeat_penalty,
            top_k=self.config.llama.top_k,
            last_n_tokens_size=self.config.llama.last_n_tokens_size,
            use_mmap=self.config.llama.use_mmap,
            streaming=self.config.llama.streaming,
            callback_manager=CallbackManager( [ StreamingStdOutCallbackHandler() ] ),
            verbose=self.config.llama.verbose
        )

    def _process_history( self, history: List[Dict[str, str]] ) -> str:
        processed_history = ''

        if self.config.llama.do_prompt_injection:
            processed_history += self.config.llama.system_prompt

        for message in history:
            if message['role'] == 'system':
                if not self.config.llama.do_prompt_injection or message != history[0]:
                    processed_history += 'SYSTEM: ' + message['content'].strip() + ' '

            if message['role'] == 'assistant':
                processed_history += 'ASSISTANT: ' + message['content'].strip() + '</s>'
            if message['role'] == 'user':
                processed_history += 'USER: ' + message['content'].strip() + ' '
        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self._process_history(messages)
        prompt = history + "ASSISTANT:"
        response = self.llm( prompt )
        return response


if __name__ == "__main__":
    bittensor.utils.version_checking()
    LlamaCppMiner().run()