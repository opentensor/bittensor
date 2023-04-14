
## Vicuna Delta Miner
Vicuna Chat Model miner for bittensor's prompting network.
See: https://huggingface.co/lmsys/vicuna-13b-delta-v1.1 


# Example Usage
```
python3 -m pip install -r neurons/text/prompting/miners/vicuna/requirements.txt
python3 neurons/text/prompting/miners/vicuna/neuron.py --vicuna.model_path </path/to/vicuna-delta/weights>
```

## NOTE: This "delta model" cannot be used directly.
Users have to apply it on top of the original LLaMA weights to get actual Vicuna weights.
See https://github.com/lm-sys/FastChat#vicuna-weights for instructions. (shown below also)


## Vicuna Weights
We release [Vicuna](https://vicuna.lmsys.org/) weights as delta weights to comply with the LLaMA model license.
You can add our delta to the original LLaMA weights to obtain the Vicuna weights. Instructions:

1. Get the original LLaMA weights in the huggingface format by following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get Vicuna weights by applying our delta. They will automatically download delta weights from our Hugging Face [account](https://huggingface.co/lmsys).

**NOTE**:
Weights v1.1 are only compatible with the latest main branch of huggingface/transformers and ``fschat >= 0.2.0``.
Please update your local packages accordingly. If you follow the above commands to do a fresh install, then you should get all the correct versions.


### Install FastChat

```bash
# Install FastChat
pip3 install fschat

# Install the latest main branch of huggingface/transformers
pip3 install git+https://github.com/huggingface/transformers
```

### Vicuna-7B
This conversion command needs around 30 GB of CPU RAM.
If you do not have enough memory, you can create a large swap file that allows the operating system to automatically utilize the disk as virtual memory.
```bash
python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-7b \
    # target is `model_path` argument for neuron.py
    --target /output/path/to/vicuna-7b \
    --delta lmsys/vicuna-7b-delta-v1.1 
```

### Vicuna-13B
This conversion command needs around 60 GB of CPU RAM.
If you do not have enough memory, you can create a large swap file that allows the operating system to automatically utilize the disk as virtual memory.
```bash
python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-13b \
    # target is `model_path` argument for neuron.py
    --target /output/path/to/vicuna-13b \
    --delta lmsys/vicuna-13b-delta-v1.1 
```