"""
Implementation of the config class, which manages the config of different bittensor modules.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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

import yaml
import json
import pandas
import bittensor
from munch import Munch
from prometheus_client import Info

class Config ( Munch ):
    """
    Implementation of the config class, which manages the config of different bittensor modules.
    """
    def __init__(self, loaded_config = None ):
        super().__init__()
        if loaded_config:
            raise NotImplementedError('Function load_from_relative_path is not fully implemented.')

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self) -> str:
        return "\n" + yaml.dump(self.toDict())

    def to_string(self, items) -> str:
        """ Get string from items
        """
        return "\n" + yaml.dump(items.toDict())

    def update_with_kwargs( self, kwargs ):
        """ Add config to self
        """
        for key,val in kwargs.items():
            self[key] = val

    def to_prometheus(self):
        """
            Sends the config to the inprocess prometheus server if it exists.
        """
        try:
            prometheus_info = Info('config', 'Config Values')
            config_info = pandas.json_normalize(json.loads(json.dumps(self)), sep='.').to_dict(orient='records')[0]
            formatted_info = {}
            for key in config_info:
                config_info[key] = str(config_info[key])
                formatted_info[key.replace('.', '_')] = str(config_info[key])
            prometheus_info.info(formatted_info)
        except ValueError:
            # The user called this function twice in the same session.
            # TODO(const): need a way of distinguishing the various config items.
            bittensor.__console__.print("The config has already been added to prometheus.", highlight=True)

    def to_defaults(self):
        try: 
            if 'axon' in self.keys():
                bittensor.defaults.axon.port = self.axon.port
                bittensor.defaults.axon.ip = self.axon.ip
                bittensor.defaults.axon.external_port = self.axon.external_port
                bittensor.defaults.axon.external_ip = self.axon.external_ip
                bittensor.defaults.axon.max_workers = self.axon.max_workers
                bittensor.defaults.axon.maximum_concurrent_rpcs = self.axon.maximum_concurrent_rpcs
            
            if 'dataset' in self.keys():
                bittensor.defaults.dataset.batch_size = self.dataset.batch_size
                bittensor.defaults.dataset.block_size = self.dataset.block_size
                bittensor.defaults.dataset.num_batches = self.dataset.num_batches
                bittensor.defaults.dataset.num_workers = self.dataset.num_workers
                bittensor.defaults.dataset.dataset_names = self.dataset.dataset_names
                bittensor.defaults.dataset.data_dir = self.dataset.data_dir
                bittensor.defaults.dataset.save_dataset = self.dataset.save_dataset
                bittensor.defaults.dataset.max_datasets = self.dataset.max_datasets

            if 'dendrite' in self.keys():
                bittensor.defaults.dendrite.timeout = self.dendrite.timeout
                bittensor.defaults.dendrite.max_active_receptors = self.dendrite.max_active_receptors
                bittensor.defaults.dendrite.requires_grad = self.dendrite.requires_grad

            if  'logging' in self.keys():
                bittensor.defaults.logging.debug = self.logging.debug
                bittensor.defaults.logging.trace = self.logging.trace
                bittensor.defaults.logging.record_log = self.logging.record_log
                bittensor.defaults.logging.logging_dir = self.logging.logging_dir
            
            if 'subtensor' in self.keys():
                bittensor.defaults.subtensor.network = self.subtensor.network
                bittensor.defaults.subtensor.chain_endpoint = self.subtensor.chain_endpoint
            
            if 'threadpool' in self.keys():
                bittensor.defaults.threadpool.max_workers = self.threadpool.max_workers
                bittensor.defaults.threadpool.maxsize = self.threadpool.maxsize 

            if 'wallet' in self.keys():
                bittensor.defaults.wallet.name = self.wallet.name
                bittensor.defaults.wallet.hotkey = self.wallet.hotkey
                bittensor.defaults.wallet.path = self.wallet.path
            
            if 'wandb' in self.keys():
                bittensor.defaults.wandb.name = self.wandb.name
                bittensor.defaults.wandb.project = self.wandb.project
                bittensor.defaults.wandb.tags = self.wandb.tags
                bittensor.defaults.wandb.run_group = self.wandb.run_group
                bittensor.defaults.wandb.directory = self.wandb.directory
                bittensor.defaults.wandb.offline = self.wandb.offline

        except Exception as e:
            print('Error when loading config into defaults {}'.format(e))