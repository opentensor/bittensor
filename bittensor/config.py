from loguru import logger
import argparse
import munch
import os
import pathlib
import requests
import time
import validators
import yaml
from munch import Munch

class InvalidConfigFile(Exception):
    pass

class ValidationError(Exception):
    pass

class PostProcessingError(Exception):
    pass

class InvalidConfigError(Exception):
    pass

class Config:

    @staticmethod
    def load (config_path: str = None, from_yaml: str = None) -> Munch:
        r""" Loads and returns a Munched config object from args/defaults.

            Args:
                config_path (str, `optional`): 
                    Relative path to a config.yaml file which overrides defaults.

                from_yaml (str, `optional`)
                    String yaml parsed by yaml.safe_load(str).
                
            Returns:
                config  (:obj:`Munch` `required`):
                    Python Munch object with dot syntax config items e.g. config.neuron.neuron_path.
        """
        config = Config.load_from_defaults()

        if config_path != None:
            passed_items = Config.load_from_relative_path(config_path)
            config = Config.overwrite_add(config, passed_items)

        neuron_config = Config.load_from_neuron_path(config)
        config = Config.overwrite_add(config, neuron_config)

        if from_yaml != None:
            yaml_items = Config.load_from_yaml_string(from_yaml)
            config = Config.overwrite_add(config, yaml_items)

        Config.post_process(config)
        Config.validate(config)

        return config

    @staticmethod   
    def load_from_defaults() -> Munch:
        r""" Loads and returns a Munched config object from defaults.
    
            Returns:
                defaults  (:obj:`Munch` `required`):
                    Python Munch object with values from bittensor/neurons/defaults.yaml
        """
        # 1. Load defaults from defaults.yaml.
        defaults = munch.Munch()
        defaults_path = 'bittensor/neurons/defaults.yaml'
        if not os.path.isfile(defaults_path):
                raise FileNotFoundError('Cannot find default configuration at defaults.yaml')
        with open(defaults_path, 'r') as f:
            try:
                defaults = yaml.safe_load(f)
                defaults = munch.munchify(defaults)
            except yaml.YAMLError as exc:
                logger.error('CONFIG: cannot parse default configuration file at defaults.yaml')
                raise InvalidConfigFile
        return defaults

    @staticmethod   
    def load_from_neuron_path(items: Munch = None) -> Munch:
        r""" Loads and returns a Munched config object from --neuron_path arg on command line.
    
            Returns:
                defaults  (:obj:`Munch` `required`):
                    Python Munch object with values from bittensor/neurons/defaults.yaml
        """
        # 1. Load args or defaults.yaml neuron path.
        params_path = None
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--neuron_path', default=None, help='path to a neuron configuration file')
            params = parser.parse_args()
            params_path = params.neuron_path
        except:
            pass

        neuron_items = None
        try:
            neuron_items = items.neuron.neuron_path
        except:
            pass

        # If neither has neuron path throw error.
        if params_path == None and neuron_items == None:
            logger.error('CONFIG: must specify a neuron path either through --neuron_path or in the passed item.neuron.neuron_path')
            raise InvalidConfigFile

        if params_path != None:
            return Config.load_from_relative_path(params_path)
        
        else:
            return Config.load_from_relative_path(neuron_items + '/config.yaml')

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
            except:
                logger.error('CONFIG: failed parsing passed yaml with input {}', yaml_str)
                raise InvalidConfigFile
        return yaml_items

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
        # Load items from relative path.
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
    def validate(items):
        try:
            try:
                items.session_settings.remote_ip
            except:
                logger.error("CONFIG: An error occured while parsing configuration. {} is required", 'session_settings.remote_ip')
            try:
                items.neuron.datapath
            except:
                logger.error("CONFIG: An error occured while parsing configuration. {} is required", 'neuron.datapath')
            try:
                items.session_settings.logdir
            except:
                logger.error("CONFIG: An error occured while parsing configuration. {} is required", 'session_settings.logdir')
            try:
                items.neuron.neuron_path
            except:
                logger.error("CONFIG: An error occured while parsing configuration. {} is required", 'neuron.neuron_path)')
            try:
                items.session_settings.axon_port
            except:
                logger.error("CONFIG: An error occured while parsing configuration. {} is required", 'neuron.datapath')
            Config.validate_path_create('neuron.datapath', items.neuron.datapath)
            Config.validate_path_create('session_settings.logdir', items.session_settings.logdir)
            Config.validate_neuron_file('neuron.neuron_path', items.neuron.neuron_path)
            Config.validate_ip('session_settings.remote_ip', items.session_settings.remote_ip)
            Config.validate_int_range('session_settings.axon_port', items.session_settings.axon_port, min=1024, max=65535)
        
        except ValidationError:
            logger.debug("CONFIG: Validation error")
            raise InvalidConfigError

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
        except:
            logger.error("CONFIG: Validation error: An undefined error occured while trying to create path {} for option {}", value, key)
            raise ValidationError
    
    @staticmethod
    def validate_neuron_file(key, value):
        full_path = os.getcwd() + value + '/neuron.py'
        if not os.path.isfile(full_path):
            logger.error("CONFIG: Neuron path does not point to valid neuron at {} and derived path {}", value, full_path)
            raise ValidationError

    @staticmethod
    def obtain_ip_address(items):
        try:
            items.session_settings.remote_ip
            return
        except:
            pass
        try:
            value = requests.get('https://api.ipify.org').text
        except:
            logger.error("CONFIG: Could not retrieve public facing IP from IP API.")
            raise PostProcessingError

        if not validators.ipv4(value):
            logger.error("CONFIG: Response from IP API is not a valid IP.")
            raise PostProcessingError
        items.session_settings.remote_ip = value

    @staticmethod
    def fix_paths(items):
        if items.neuron.datapath and items.neuron.datapath[-1] != '/':
            items.neuron.datapath = items.neuron.datapath + '/'

        if items.session_settings.logdir and items.session_settings.logdir[-1] != '/':
            items.session_settings.logdir = items.session_settings.logdir + '/'

    @staticmethod
    def toString(config, depth=0):
        for k, v in config.items():
            if isinstance(v, dict):
                print ('\t' * depth, '{}:'.format(k))
                Config.toString(v, depth = depth + 1)
            else:
                print ('\t' * depth, '{}: {}'.format(k, v) )

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