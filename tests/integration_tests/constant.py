from munch import Munch

dataset = Munch().fromDict(
  {
    'dataset_name': ["ArXiv"],
    'num_batches': 10, 
    'max_hash_size': 10000,
    'buffer_size': 1000
  }
)

synapse = Munch().fromDict(
  {
    'num_to_generate': 70,
  }
)