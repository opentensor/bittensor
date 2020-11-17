from loguru import logger
import argparse
import munch
import os
import pathlib
import requests
import time
import validators
import yaml


class InvalidConfigFile(Exception):
    pass

class ValidationError(Exception):
    pass

class PostProcessingError(Exception):
    pass

class InvalidConfigError(Exception):
    pass

# Overwrites the values from nested dictionary A with nested dictionary B
def overwrite(items_a, items_b):
    for k, v in items_a.items():
        if k in items_b:
            if isinstance(v, dict):
                overwrite(v, items_b[k])
            else:
                if items_b != None:
                    items_a[k] = items_b[k]
    return items_a

def add(items_a, items_b):
    for k, v in items_b.items():
        if k not in items_a:
            items_a[k] = items_b[k]
        if k in items_a:
            if isinstance(v, dict):
                add(items_a[k], items_b[k])
    return items_a

def overwrite_add(items_a, items_b):
    items_a = overwrite(items_a, items_b)
    items_a = add(items_a, items_b)
    return items_a

class Config:
    def toString(config, depth=0):
        for k, v in config.items():
            if isinstance(v, dict):
                print ('\t' * depth, '{}:'.format(k))
                Config.toString(v, depth = depth + 1)
            else:
                print ('\t' * depth, '{}: {}'.format(k, v) )

    def load (config_path: str = None, from_yaml: str = None):
        # 1. Load defaults from defaults.yaml.
        if not os.path.isfile('defaults.yaml'):
                raise FileNotFoundError('Cannot find default configuration at defaults.yaml')
        with open('defaults.yaml', 'r') as f:
            try:
                config_items = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                logger.error('CONFIG: cannot parse default configuration file at defaults.yaml')
                raise InvalidConfigFile

        # 2. Load items from optional passed config_path
        passed_items = None
        if config_path != None:
            passed_config_path = os.getcwd() + '/' + config_path
            if not os.path.isfile(passed_config_path):
                logger.error('CONFIG: cannot find passed configuration file at {}', passed_config_path)
                raise FileNotFoundError('Cannot find a configuration file at', passed_config_path)
            with open(passed_config_path, 'r') as f:
                try:
                    passed_items = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    logger.error('CONFIG: cannot parse passed configuration file at {}', passed_config_path)
                    raise InvalidConfigFile

        # Overwrite defaults with passed items.
        if passed_items != None:
            config_items = overwrite_add(config_items, passed_items)

        # 3. Load items from optional passed config_path
        neuron_items = None
        if config_items['neuron']['neuron_path'] != None:
            neuron_config_path = os.getcwd() + str(config_items['neuron']['neuron_path']) + '/config.yaml'
            if not os.path.isfile(neuron_config_path):
                logger.error('CONFIG: failed parsing neuron_path config file at {}', neuron_config_path)
                raise FileNotFoundError('Cannot find a neuron configuration file at', neuron_config_path)
            with open(neuron_config_path, 'r') as f:
                try:
                    neuron_items = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    logger.error('CONFIG: cannot parse neuron configuration file at {}', neuron_config_path)
                    raise InvalidConfigFile
                    
        # Overwrite items with neuron items.
        if neuron_items != None:
            config_items = overwrite_add(config_items, neuron_items)

        # 4. Load items from optional passed --config_path
        commandline_items = None
        params = None
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config_file', default=None, help='path to a configuration file')
            params = parser.parse_args()
        except:
            pass
        if params and params.config_file != None:
            if not os.path.isfile(params.config_file):
                logger.error('CONFIG: failed parsing command line config file at {}', params.config_file)
                raise FileNotFoundError('Cannot find a configuration file at', params.config_file)
            else:
                with open(params.config_file, 'r') as f:
                    try:
                        commandline_items = yaml.safe_load(f)
                    except yaml.YAMLError as exc:
                        logger.error('CONFIG: cannot parse commandline configuration file at {}', params.config_file)
                        raise InvalidConfigFile

        # Overwrite items with commandline_items
        if commandline_items != None:
            config_items = overwrite_add(config_items, commandline_items)

        # 5. Load items from passed yaml
        passed_yaml_items = None
        if from_yaml != None:
            try:
                passed_yaml_items = yaml.safe_load(from_yaml)
            except:
                logger.error('CONFIG: failed parsing passed yaml with input {}', from_yaml)
                raise InvalidConfigFile

        #  Overwrite items with passed yaml.
        if passed_yaml_items != None:
            config_items = overwrite_add(config_items, passed_yaml_items)

        # 6. Munchify
        items = munch.munchify(config_items)

        # 7. Post process.
        try:
            # 7.1 Optain remote ip.
            Config.obtain_ip_address(items)

            # 7.2 Fix paths.
            Config.fix_paths(items)

        except PostProcessingError:
            logger.debug("CONFIG: post processing error.")
            raise InvalidConfigError

        # 8. Validate config items.
        Config.validate_config(items)

        return items

    def validate_config(items):
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

    def validate_ip(key, value):
        if not validators.ipv4(value):
            logger.error("CONFIG: Validation error: {} for option {} is not a valid ip address", value, key)
            raise ValidationError

    def validate_int_range(key, value, min, max):
        """
        Validates if a specifed integer falls in the specified range
        """
        if not validators.between(value, min=min, max=max):
            logger.error(
                "CONFIG: Validation error: {} should be between {} and {}.", key, min, max)
            raise ValidationError

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

    def validate_neuron_file(key, value):
        full_path = os.getcwd() + value + '/neuron.py'
        if not os.path.isfile(full_path):
            logger.error("CONFIG: Neuron path does not point to valid neuron at {} and derived path {}", value, full_path)
            raise ValidationError

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
        items['session_settings']['remote_ip'] = value

    def fix_paths(items):
        if items.neuron.datapath and items.neuron.datapath[-1] != '/':
            items.neuron.datapath = items.neuron.datapath + '/'

        if items.session_settings.logdir and items.session_settings.logdir[-1] != '/':
            items.session_settings.logdir = items.session_settings.logdir + '/'

    # def validate_hostname(key, required=False):
    #     if self.has_valid_empty_value(config_key, required):
    #         return

    #     value = self.config[config_key]

    #     if not validators.ipv4(value) and not validators.domain(value):
    #         logger.error(
    #             "CONFIG: Validation error: %s for option %s is not a valid ip address or hostname" %
    #             (value, config_key))

    #         raise ValidationError

    # def validate_socket(self, config_key, required=False):
    #     if self.has_valid_empty_value(config_key, required):
    #         return

    #     value = self.config[config_key]

    #     try:
    #         host, port =  value.split(":")
    #     except ValueError:
    #         logger.error(
    #             "CONFIG: Validation error: %s for option %s is incorrectly formatted. Should be ip:port" %
    #             (value, config_key))
    #         raise ValidationError

    #     if not validators.ipv4(host) and host != "localhost" and not validators.domain(host):
    #         logger.error(
    #             "CONFIG: Validation error: %s for option %s does not contain a valid ip address" %
    #             (value, config_key))

    #         raise ValidationError

    #     try:
    #         port = int(port)
    #     except ValueError:
    #         logger.error(
    #             "CONFIG: Validation error: %s for option %s does contain not a valid port nr" %
    #             (value, config_key))

    #         raise ValidationError

    #     if not validators.between(port, min=1024, max=65535):
    #         logger.error(
    #             "CONFIG: Validation error: %s for option %s port must be between 1024 and 65535" %
    #             (value, config_key))

    #         raise ValidationError