from munch import Munch

dataset = Munch().fromDict(
  {
    'dataset_names': ["Books3"],
    'num_batches': 10
  }
)

synapse = Munch().fromDict(
  {
    'num_to_generate': 70,
  }
)