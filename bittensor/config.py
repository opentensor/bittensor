import requests
from loguru import logger
import argparse
from importlib.machinery import SourceFileLoader
import munch
import os
import pathlib
import validators
import yaml
from munch import Munch

from bittensor.axon import Axon
from bittensor.session import Session, KeyError
from bittensor.dendrite import Dendrite
from bittensor.metagraph import Metagraph
from bittensor.metadata import Metadata
from bittensor.subtensor.interface import Keypair, KeypairRepresenter



from bittensor.session import KeyFileError

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
    def load(parser: argparse.ArgumentParser = None)  -> Munch:
        r""" Loads and return the bittensor Munched config.

            Args:
                parser (str, `required`): 
                    argument parser.    
    
            Returns:
                config  (:obj:`Munch` `required`):
                    Python Munch object with values from parsed string.
        """
        if parser == None:
            parser = argparse.ArgumentParser()
            
        # 1. Load args from bittensor backend components.
        Axon.add_args(parser)
        Dendrite.add_args(parser)
        Metagraph.add_args(parser)
        Metadata.add_args(parser)
        Session.add_args(parser)
       
        # 2. Parse.
        params = parser.parse_known_args()[0]

        # 3. Splits params on dot synatax i.e session.axon_port
        config = Munch()
        for arg_key, arg_val in params.__dict__.items():
            split_keys = arg_key.split('.')
            if len(split_keys) == 1:
                config[arg_key] = arg_val
            else:
                head = config
                for key in split_keys[:-1]:
                    if key not in config:
                        head[key] = Munch()
                    head = head[key] 
                head[split_keys[-1]] = arg_val

        # 4. Run session checks.
        try:
            Dendrite.check_config(config)
            Session.check_config(config)
            Metagraph.check_config(config)
            Metadata.check_config(config)
            Axon.check_config(config)
        except KeyFileError:
            quit()


        #5. Load key
        try:
            Session.load_keypair(config)
        except KeyError:
            quit()

        return config

            
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
            path = os.getcwd() + '/' + path + '/config.yaml'
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
    def validate(config, neuron_path) -> Munch:
        # 1. Run session checks.
        config = Dendrite.check_config(config)
        config = Session.check_config(config)
        config = Metagraph.check_config(config)
        config = Axon.check_config(config)

        # 2. Run neuron checks.
        neuron_module = SourceFileLoader("Neuron", os.getcwd() + '/' + neuron_path + '/neuron.py').load_module()
        config = neuron_module.Neuron.check_config( config )
        return config

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
        if items.axon.remote_ip:
            return
            
        try:
            value = requests.get('https://api.ipify.org').text
        except Exception as e:
            logger.error("CONFIG: Could not retrieve public facing IP from IP API. Exception: {}", e)
            raise PostProcessingError

        if not validators.ipv4(value):
            logger.error("CONFIG: Response from IP API is not a valid IP.")
            raise PostProcessingError
        items.axon.remote_ip = value
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