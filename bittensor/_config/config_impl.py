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
from pandas import json_normalize
from typing import Dict
import copy
import bittensor

class Config ( Munch ):
    """
    Implementation of the config class, which manages the config of different bittensor modules.
    """
    __is_set: Dict[str, bool]

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

    @classmethod
    def _merge( cls, a, b ):
        """Merge two configurations recursively.
        If there is a conflict, the value from the second configuration will take precedence.
        """
        for key in b:
            if key in a:
                if isinstance( a[key], dict ) and isinstance( b[key], dict ):
                    a[key] = cls._merge( a[key], b[key] )
                else:
                    a[key] = b[key]
            else:
                a[key] = b[key]
        return a

    def merge(self, b):
        """ Merge two configs
        """
        self = self._merge( self, b )

    def to_prometheus(self):
        """
            Sends the config to the inprocess prometheus server if it exists.
        """
        try:
            prometheus_info = Info('config', 'Config Values')
            # Make copy, remove __is_set map
            config_copy = copy.deepcopy(self)

            del config_copy['__is_set']

            config_info = json_normalize(json.loads(json.dumps(config_copy)), sep='.').to_dict(orient='records')[0]
            formatted_info = {}
            for key in config_info:
                config_info[key] = str(config_info[key])
                formatted_info[key.replace('.', '_')] = str(config_info[key])
            prometheus_info.info(formatted_info)
        except ValueError:
            # The user called this function twice in the same session.
            # TODO(const): need a way of distinguishing the various config items.
            bittensor.__console__.print("The config has already been added to prometheus.", highlight=True)

    def is_set(self, param_name: str) -> bool:
        """
        Returns a boolean indicating whether the parameter has been set or is still the default.
        """
        if param_name not in self.get('__is_set'):
            return False
        else:
            return self.get('__is_set')[param_name]

    def __fill_with_defaults__(self, is_set_map: Dict[str, bool], defaults: 'Config') -> None:
        """
        Recursively fills the config with the default values using is_set_map
        """
        defaults_filtered = {}
        for key in self.keys():
            if key in defaults.keys():
                defaults_filtered[key] = getattr(defaults, key)
        # Avoid erroring out if defaults aren't set for a submodule
        if defaults_filtered == {}: return

        flat_defaults = json_normalize(defaults_filtered, sep='.').to_dict('records')[0]
        for key, val in flat_defaults.items():
            if key not in is_set_map:
                continue
            elif not is_set_map[key]:
                # If the key is not set, set it to the default value
                # Loop through flattened key to get leaf
                a = self
                keys = key.split('.')
                for key_ in keys[:-1]:
                    if key_ not in a:
                        a[key_] = {}
                    a = a[key_]
                # Set leaf to default value
                a[keys[-1]] = val

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