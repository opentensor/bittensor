import time
import torch
import argparse
import bittensor
from typing import List, Dict, Union, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

def config():       
    parser = argparse.ArgumentParser( description = 'Whisper Text to Speech Miner' )
    parser.add_argument( '--vicuna.model_name', type=str, required=True, help='Name/path of model to load' )
    parser.add_argument( '--vicuna.device', type=str, help='Device to load model', default="cuda" )
    parser.add_argument( '--vicuna.max_new_tokens', type=int, help='Max tokens for model output.', default=256 ) 
    parser.add_argument( '--vicuna.temperature', type=float, help='Sampling temperature of model', default=0.5 )
    parser.add_argument( '--vicuna.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
    parser.add_argument( '--vicuna.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
    parser.add_argument( '--vicuna.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. " )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )


# create config
configuration = config()
configuration.vicuna.model_name = "eachadea/vicuna-7b-1.1" # model from huggingface, you can also place a local model file path here
configuration.vicuna.device = "cuda" # or "cpu"
configuration.vicuna.max_new_tokens = 256 # max number of tokens to generate
configuration.vicuna.temperature = 0.5 # sampling temperature
configuration.vicuna.do_sample = True # use sampling instead of greedy decoding
configuration.vicuna.do_prompt_injection = True # use a custom prompt instead of the one sent by bittensor
configuration.vicuna.system_prompt = "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. " # custom prompt


# load base miner

base_miner = bittensor.base_miner_neuron( netuid = 1, config = configuration ) 

# load tokenizer and model 