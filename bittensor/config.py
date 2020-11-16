from loguru import logger
import argparse
import munch
import os
import requests
import time
import validators
import yaml

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

    def load (config_path: str = None):
        # 1. Load defaults from defaults.yaml.
        if not os.path.isfile('defaults.yaml'):
                raise FileNotFoundError('Cannot find default configuration at defaults.yaml')
        with open('defaults.yaml', 'r') as f:
            try:
                config_items = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

        # 2. Load items from optional passed config_path
        passed_items = None
        if config_path != None:
            if not os.path.isfile(config_path):
                raise FileNotFoundError('Cannot find a configuration file at', config_path)
            with open(config_path, 'r') as f:
                try:
                    passed_items = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)

        # Overwrite defaults with passed items.
        if passed_items != None:
            config_items = overwrite_add(config_items, passed_items)

        # 3. Load items from optional passed config_path
        neuron_items = None
        if config_items['neuron']['neuron_path'] != None:
            neuron_config_path = os.getcwd() + str(config_items['neuron']['neuron_path']) + '/config.yaml'
            if not os.path.isfile(neuron_config_path):
                raise FileNotFoundError('Cannot find a neuron configuration file at', neuron_config_path)
            with open(neuron_config_path, 'r') as f:
                try:
                    neuron_items = yaml.safe_load(f)
                except yaml.YAMLError as exc:
                    print(exc)

        # Overwrite items with neuron items.
        if neuron_items != None:
            config_items = overwrite_add(config_items, neuron_items)

        # 4. Load items from optional passed --config_path
        commandline_items = None
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file', default=None, help='path to a configuration file')
        params = parser.parse_args()
        if params.config_file != None:
            if not os.path.isfile(params.config_file):
                raise FileNotFoundError('Cannot find a configuration file at', params.config_file)
            else:
                with open(params.config_file, 'r') as f:
                    commandline_items = yaml.safe_load(f)

        # Overwrite items with commandline_items
        if commandline_items != None:
            config_items = overwrite_add(config_items, commandline_items)
                    
        # 5. Validate config items.
        #validate_config(config_items)

        # 6. Munchify
        items = munch.munchify(config_items)
        return items

    def __validate_config(self):
        # Now validate all settings

        # Todo: Implement chain endpoint validation
        try:
            # Chain endpoint is not implemented yet, no validation

            validate_path(Config.NEURON_PATH, required=True)
            self.validate_path(Config.DATAPATH, required=True)
            self.validate_path(Config.LOGDIR, required=True)
            self.validate_ip(Config.REMOTE_IP, required=False)

            self.validate_int_range(Config.AXON_PORT, min=1024, max=65535, required=True)

            self.validate_int_range(Config.METAGRAPH_PORT, min=1024, max=65535, required=True)
            self.validate_int_range(Config.METAGRAPH_SIZE, min=5, max=20000, required=True)

            self.validate_socket(Config.BOOTSTRAP, required=False)

        except ValidationError:
            logger.debug("CONFIG: Validation error")
            raise InvalidConfigError

    def __do_post_processing(self):
        try:
            self.__obtain_ip_address()
            self.__fix_paths()
        except PostProcessingError:
            logger.debug("CONFIG: post processing error.")
            raise InvalidConfigError

    def __obtain_ip_address(self):
        if self.config[Config.REMOTE_IP]:
            return
        try:
            value = requests.get('https://api.ipify.org').text
        except:
            logger.error("CONFIG: Could not retrieve public facing IP from IP API.")
            raise PostProcessingError

        if not validators.ipv4(value):
            logger.error("CONFIG: Response from IP API is not a valid IP.")
            raise PostProcessingError

        self.config[Config.REMOTE_IP] = value

    def __fix_paths(self):
        if self.config[Config.DATAPATH] and self.config[Config.DATAPATH][-1] != '/':
            self.config[Config.DATAPATH] += '/'

        if self.config[Config.LOGDIR] and self.config[Config.LOGDIR][-1] != '/':
            self.config[Config.LOGDIR] += '/'

    def load_config_option(self, section, option):
        """

        This function loads a config option from the config file.
        Note: Do not use this directly, instead use the wrapper functions
        * load_str
        * load_int

        As they cast the value to the right type

        Args:
            key:   Key of the config dict
            section:  section of the config file
            option:  the option of the config file

        Returns:
            * None when the config option does not exist
            * The string encoded value of the option (if there is not value, the string will be empty)

        Throws:
            InvalidConfigFile in case of a section not being present
        """
        try:
            return self.parser.get(section, option)
        except configparser.NoSectionError:
            logger.error("Section %s not found in config.ini" % section)
            raise InvalidConfigFile
        except configparser.NoOptionError:
            return None

    def load_str(self, key, section, option):
        """
        Loads values that should be parsed as string
        """
        value = self.load_config_option(section, option)
        if value is None:
            return

        self.config[key] = value

    def load_int(self, key, section, option):
        """
        Loads values that should be parsed as an int
        """
        try:
            # Return None if option has no value
            value = self.load_config_option(section, option)
            if value is None:
                return

            value = int(value)

            self.config[key] = value

        except ValueError:
            logger.error(
                "CONFIG: An error occured while parsing config.ini. Option '%s' in section '%s' should be an integer" % (
                option, section))
            raise ValueError

    '''
    Validation routines
    '''

    def has_valid_empty_value(self, config_key, required):
        value = self.config[config_key]
        if not value and required:
            logger.error("CONFIG: An error occured while parsing configuration. %s is required" % config_key)
            raise ValidationError
        if not value and not required:
            return True

        return False

    def validate_int_range(self, config_key, min, max, required=False):

        """
        Validates if a specifed integer falls in the specified range
        """
        if self.has_valid_empty_value(config_key, required):
            return

        value = self.config[config_key]

        if not validators.between(value, min=min, max=max):
            logger.error(
                "CONFIG: Validation error: %s should be between %i and %i." % (
                    config_key, min, max))
            raise ValidationError

    def validate_path(self, config_key, required=False):
        if self.has_valid_empty_value(config_key, required):
            return

        path = self.config[config_key]

        try:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error("CONFIG: Validation error: no permission to create path %s for option %s" %
                         (path, config_key))
            raise ValidationError
        except:
            logger.error("CONFIG: Validation error: An undefined error occured while trying to create path %s for option %s" %
                         (path, config_key))
            raise ValidationError

    def validate_ip(self, config_key, required=False):
        if self.has_valid_empty_value(config_key, required):
            return

        value = self.config[config_key]

        if not validators.ipv4(value):
            logger.error(
                "CONFIG: Validation error: %s for option %s is not a valid ip address" %
                (value, config_key))

            raise ValidationError

    def validate_hostname(self, config_key, required=False):
        if self.has_valid_empty_value(config_key, required):
            return

        value = self.config[config_key]

        if not validators.ipv4(value) and not validators.domain(value):
            logger.error(
                "CONFIG: Validation error: %s for option %s is not a valid ip address or hostname" %
                (value, config_key))

            raise ValidationError

    def validate_socket(self, config_key, required=False):
        if self.has_valid_empty_value(config_key, required):
            return

        value = self.config[config_key]

        try:
            host, port =  value.split(":")
        except ValueError:
            logger.error(
                "CONFIG: Validation error: %s for option %s is incorrectly formatted. Should be ip:port" %
                (value, config_key))
            raise ValidationError

        if not validators.ipv4(host) and host != "localhost" and not validators.domain(host):
            logger.error(
                "CONFIG: Validation error: %s for option %s does not contain a valid ip address" %
                (value, config_key))

            raise ValidationError

        try:
            port = int(port)
        except ValueError:
            logger.error(
                "CONFIG: Validation error: %s for option %s does contain not a valid port nr" %
                (value, config_key))

            raise ValidationError

        if not validators.between(port, min=1024, max=65535):
            logger.error(
                "CONFIG: Validation error: %s for option %s port must be between 1024 and 65535" %
                (value, config_key))

            raise ValidationError