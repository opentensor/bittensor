'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
import argparse
import munch
import os
import pathlib
import requests
import stat
import validators
import yaml

from loguru import logger
from munch import Munch
from importlib.machinery import SourceFileLoader

from bittensor.crypto import KeyError
from bittensor.crypto.keyfiles import KeyFileError

class InvalidConfigFile(Exception):
    pass

class ValidationError(Exception):
    pass

class PostProcessingError(Exception):
    pass

class InvalidConfigError(Exception):
    pass

class MustPassNeuronPath(Exception):
    pass


class Config:
    @staticmethod
    def toString(items) -> str:
        return "\n" + yaml.dump(items.toDict())

    @staticmethod
    def to_config(parser: argparse.ArgumentParser) -> Munch:
    
        params = parser.parse_known_args()[0]
        config_file = None
        config = Munch()
        if 'session.config_file' in vars(params).keys():
            config_file = vars(params)['session.config_file']
        
        if config_file:
            config = Config.load_from_relative_path(config_file)

        # 3. Splits params on dot syntax i.e neuron.axon_port
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split('.')
            
            if len(split_keys) == 1:
                config[arg_key] = arg_val
            else:
                if hasattr(config, split_keys[0]):
                    section = getattr(config, split_keys[0])
                
                    if not hasattr(section, split_keys[1]):
                        head = config
                        for key in split_keys[:-1]:
                            if key not in config:
                                head[key] = Munch()
                            head = head[key] 
                        head[split_keys[-1]] = arg_val
                else:
                    head = config
                    for key in split_keys[:-1]:
                        if key not in config:
                            head[key] = Munch()
                        head = head[key] 
                    head[split_keys[-1]] = arg_val

        return config

    @staticmethod
    def check_and_create_config_dir():
        path = "~/.bittensor"
        path = os.path.expanduser(path)

        if not os.path.isdir(path):
            Config.create_config_dir(path)


    @staticmethod
    def create_config_dir(path):
        logger.info("Creating {} config dir", path)
        os.makedirs(path, exist_ok=True)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


    @staticmethod
    def load_from_relative_path(path: str)  -> Munch:
        r""" Loads and returns a Munched config object from a relative path.

            Args:
                path (str, `required`): 
                    Path to config.yaml file. full_path = cwd() + path
    
            Returns:
                config  (:obj:`Munch` `required`):
                    Python Munch object with values from config under path.
        """
        # Load yaml items from relative path.
        path_items = munch.Munch()
        if path != None:
            path = os.getcwd() + '/' + path
            if not os.path.isfile(path):
                logger.error('CONFIG: cannot find passed configuration file at {}', path)
                raise FileNotFoundError('Cannot find a configuration file at', path)
            with open(path, 'r') as f:
                try:
                    path_items = yaml.safe_load(f)
                    path_items = munch.munchify(path_items)
                except yaml.YAMLError as exc:
                    logger.error('CONFIG: cannot parse passed configuration file at {}', path)
                    raise InvalidConfigFile
        return path_items

    
    @staticmethod   
    def load_from_yaml_string(yaml_str: str)  -> Munch:
        r""" Loads and returns a Munched config object from a passed string

            Args:
                yaml_str (str, `required`): 
                    String representation of yaml file.
    
            Returns:
                config  (:obj:`Munch` `required`):
                    Python Munch object with values from parsed string.
        """
        # Load items yaml string.
        yaml_items = munch.Munch()
        if yaml_str != None:
            try:
                yaml_items = yaml.safe_load(yaml_str)
                yaml_items = munch.munchify(yaml_items)
            except Exception as e:
                logger.error('CONFIG: failed parsing passed yaml with input {}. Exception: {}'.format(yaml_str, e))
                raise InvalidConfigFile
        return yaml_items

    @staticmethod
    def post_process(items):
        try:
            # 7.1 Optain remote ip.
            Config.obtain_ip_address(items)

            # 7.2 Fix paths.
            Config.fix_paths(items)

        except PostProcessingError:
            logger.debug("CONFIG: post processing error.")
            raise InvalidConfigError

    @staticmethod
    def validate_socket(key, value: str):
        def error():
            message = "CONFIG: Validation error: {} for option {} is not a valid socket definition : <ip/hostname>:<port>"
            logger.error(message, value, key)
            raise ValidationError

        if ':' not in value:
            error()

        elems = value.split(":")
        if len(elems) != 2:
            error()

        ip, port = elems
        if not validators.ipv4(ip) and not validators.domain(ip) and ip != "localhost":
            error()

        if not validators.between(int(port), min=1, max=65535):
            error()

    @staticmethod
    def validate_ip(key, value):
        if not validators.ipv4(value):
            logger.error("CONFIG: Validation error: {} for option {} is not a valid ip address", value, key)
            raise ValidationError
    
    @staticmethod
    def validate_int_range(key, value, min, max):
        """
        Validates if a specifed integer falls in the specified range
        """
        if not validators.between(value, min=min, max=max):
            logger.error(
                "CONFIG: Validation error: {} should be between {} and {}.", key, min, max)
            raise ValidationError

    @staticmethod
    def validate_path_create(key, value):
        try:
            full_neuron_path = os.getcwd() + '/' + value
            pathlib.Path(full_neuron_path).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error("CONFIG: Validation error: no permission to create path {} for option {}", value, key)
            raise ValidationError
        except Exception as e:
            logger.error("CONFIG: Validation error: An undefined error occured while trying to create path {} for option {}. Exception: {}", value, key, e)
            raise ValidationError

    @staticmethod
    def obtain_ip_address(items):
        if items.axon.external_ip:
            return
            
        try:
            value = requests.get('https://api.ipify.org').text
        except Exception as e:
            logger.error("CONFIG: Could not retrieve public facing IP from IP API. Exception: {}", e)
            raise PostProcessingError

        if not validators.ipv4(value):
            logger.error("CONFIG: Response from IP API is not a valid IP.")
            raise PostProcessingError
        items.axon.external_ip = value
        return items

    @staticmethod
    def overwrite_add(items_a, items_b):
        r""" Overwrites and adds values from items A with items B. Returns 

            Args:
                items_a (obj:Munch, `required`): 
                    Items to overrite into

                items_b (obj:Munch, `required`): 
                    Items to overrite from
    
            Returns:
                items  (:obj:`Munch` `Optional`):
                    Overritten items or None if both are None.
        """
        if items_b == None:
            return items_a
        if items_a == None:
            return items_b
        items_a = Config.overwrite(items_a, items_b)
        items_a = Config.add(items_a, items_b)
        return items_a

    @staticmethod
    def overwrite(items_a, items_b):
        r""" Overwrites values from items_b into items b.

            Args:
                items_a (obj:Munch, `required`): 
                    Items to overrite into

                items_b (obj:Munch, `required`): 
                    Items to overrite from
    
            Returns:
                items  (:obj:`Munch` `Optional`):
                    Items with overwrite from B to A.
        """
        for k, v in items_a.items():
            if k in items_b:
                if isinstance(v, dict):
                    Config.overwrite(v, items_b[k])
                else:
                    if items_b != None:
                        items_a[k] = items_b[k]
        return items_a

    @staticmethod
    def add(items_a, items_b):
        r""" Adds values from items b into items b.

            Args:
                items_a (obj:Munch, `required`): 
                    Items to overrite into

                items_b (obj:Munch, `required`): 
                    Items to overrite from
    
            Returns:
                items  (:obj:`Munch` `Optional`):
                    Items with union from A and B.
        """
        for k, v in items_b.items():
            if k not in items_a:
                items_a[k] = items_b[k]
            if k in items_a:
                if isinstance(v, dict):
                    Config.add(items_a[k], items_b[k])
        return items_a

    @staticmethod
    def update_replicate_yaml(neuron_name):
        with open("replicate.yaml") as replicate_file:
            replicate_yaml = yaml.safe_load(replicate_file)

        replicate_yaml['repository'] = "file://data/{}/.replicate".format(neuron_name)
        
        with open("replicate.yaml", "w") as f:
            yaml.dump(replicate_yaml, f)