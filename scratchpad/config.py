from configparser import ConfigParser
from loguru import logger
import configparser
import argparse
import socket
import sys, os
import re
import uuid
import requests
import validators
from bittensor.crypto import Crypto


class InvalidConfigFile(Exception):
    pass


class ValidationError(Exception):
    pass


class PostProcessingError(Exception):
    pass


class InvalidConfigError(Exception):
    pass


class Config:
    CHAIN_ENDPOINT = "chain_endpoint"
    AXON_PORT = "axon_port"
    METAGRAPH_PORT = "metagraph_port"
    METAGRAPH_SIZE = "metagraph_size"
    BOOTPEER_HOST = "bootpeer_host"
    BOOTPEER_PORT = "bootpeer_port"
    NEURON_PUBKEY = "neuron_key"
    IP = "ip"
    DATAPATH = "datapath"
    LOGPATH = "logpath"

    # Attributes to make parsing of a config file work
    parser = None
    filename = None  # This will be passed as a constructor arg

    # The dict that hold the configuration
    config = dict()

    # Attribute that tells if the configuration is valid
    valid = False

    # Command line arguments, filled during init
    cl_args = None

    def __init__(self, filename, argparser: argparse.ArgumentParser):
        self.filename = filename
        self.argparse = argparser

        self.add_cl_args(argparser)

        """
        This is what happens:
        1) The system defaults for the configuration is set
        2) Then, config.ini is loaded and any value set there is used to overwrite the configuration
        3) Then, the command line is parsed for options and used in the configuratoin
        4) Ultimately some post processing is applied. Specifically, an IP address is obtained from an IP if not configured
        """
        try:
            self.__setup_defaults()
            self.__load_config()
            self.__parse_cl_args()
            self.__validate_config()
            self.__do_post_processing()

            self.valid = True

        except InvalidConfigError:
            self.valid = False

    def add_cl_args(self, parser: argparse.ArgumentParser):
        parser.add_argument('--chain_endpoint', dest=Config.CHAIN_ENDPOINT, type=str, help="bittensor chain endpoint")
        parser.add_argument('--axon_port', dest=Config.AXON_PORT, type=int,
                            help="TCP port that will be used to receive axon connections")
        parser.add_argument('--metagraph_port', dest=Config.METAGRAPH_PORT, type=int,
                            help='TCP port that will be used to receive metagraph connections')
        parser.add_argument('--metagraph_size', dest=Config.METAGRAPH_SIZE, type=int, help='Metagraph cache size')
        parser.add_argument('--bp_host', dest=Config.BOOTPEER_HOST, type=str,
                            help='Hostname or IP of the first peer this neuron should connect to when signing onto the network. <IPv4:port>')
        parser.add_argument('--np_port', dest=Config.BOOTPEER_PORT, type=int,
                            help='TCP Port the bootpeer is listening on')
        parser.add_argument('--neuron_key', dest=Config.NEURON_PUBKEY, type=str, help='Key of the neuron')
        parser.add_argument('--ip', dest=Config.IP, type=str,
                            help='The IP address of this neuron that will be published to the network')
        parser.add_argument('--datapath', dest=Config.DATAPATH, type=str, help='Path to datasets')
        parser.add_argument('--logdir', dest=Config.LOGPATH, type=str, help='Path to logs and saved models')

        self.cl_args = parser.parse_args()

    def isValid(self):
        return self.valid

    def __setup_defaults(self):
        """
        Sets the system default configuration
        """
        self.config = dict({
            self.CHAIN_ENDPOINT: "DEFAULT",
            self.NEURON_PUBKEY: Crypto.public_key_to_string(Crypto.generate_private_ed25519().public_key()),
            self.DATAPATH: None,
            self.LOGPATH: None,
            self.IP: None,
            self.AXON_PORT: 8091,
            self.METAGRAPH_PORT: 8092,
            self.METAGRAPH_SIZE: 10000,
            self.BOOTPEER_HOST: None,
            self.BOOTPEER_PORT: 6000
        })

    def __load_config(self):
        try:
            self.__parse_config_file()
        except FileNotFoundError:
            logger.debug("CONFIG: Warning: %s not found" % self.filename)
        except (InvalidConfigFile, ValueError):
            logger.error(
                "CONFIG: %s is invalid. Try copying and adapting the default configuration file." % self.filename)
            raise InvalidConfigError
        except Exception:
            logger.error("CONFIG: An unspecified error occured.")
            raise InvalidConfigError

    def __parse_config_file(self):
        """
        Loads and parses config.ini

        Throws:
            FileNotFoundError

        """
        self.parser = ConfigParser(allow_no_value=True)

        files_read = self.parser.read(self.filename)
        if self.filename not in files_read:
            raise FileNotFoundError

        # At this point, all sections and options are present. Now load the actual values according to type
        self.load_str(self.CHAIN_ENDPOINT, "general", "chain_endpoint")
        self.load_str(self.NEURON_PUBKEY, "general", "neuron_id")
        self.load_str(self.DATAPATH, "general", "datapath")
        self.load_str(self.LOGPATH, "general", "logdir")

        self.load_str(self.IP, "general", "ip")
        self.load_int(self.AXON_PORT, "axon", "port")

        self.load_int(self.METAGRAPH_PORT, "metagraph", "port")
        self.load_int(self.METAGRAPH_SIZE, "metagraph", "size")

        self.load_str(self.BOOTPEER_HOST, "bootpeer", "host")
        self.load_int(self.BOOTPEER_PORT, "bootpeer", "port")

    def __parse_cl_args(self):
        """

        This will loop over each command line argument. If it has a value,
        it is used to overwrite the already existing configuration

        Args:
            args: The output of the argparse parser's .parse_args() function

        Returns:

        """
        args_dict = vars(self.cl_args)
        for key in args_dict:
            if args_dict[key]:
                self.config[key] = args_dict[key]

    def __validate_config(self):
        # Now validate all settings

        # Todo: Implement chain endpoint validation
        try:
            # Chain endpoint is not implemented yet, no validation

            # self.validate_key(self.NEURON_PUBKEY, required=True)

            self.validate_path(self.DATAPATH, required=True)
            self.validate_path(self.LOGPATH, required=True)
            self.validate_ip(self.IP, required=False)

            self.validate_int_range(self.AXON_PORT, min=1024, max=65535, required=True)

            self.validate_int_range(self.METAGRAPH_PORT, min=1024, max=65535, required=True)
            self.validate_int_range(self.METAGRAPH_SIZE, min=5, max=20000, required=True)

            self.validate_hostname(self.BOOTPEER_HOST, required=True)
            self.validate_int_range(self.BOOTPEER_PORT, min=1024, max=65535, required=True)

        except ValidationError:
            logger.debug("CONFIG: Validation error")
            raise InvalidConfigError

    def __do_post_processing(self):
        try:
            self.__obtain_ip_address()
        except PostProcessingError:
            logger.debug("CONFIG: post processing error.")
            raise InvalidConfigError

    def __obtain_ip_address(self):
        if self.config[self.IP]:
            return
        try:
            value = requests.get('https://api.ipify.org').text
        except:
            logger.error("CONFIG: Could not retrieve public facing IP from IP API.")
            raise PostProcessingError

        if not self.__is_valid_ip(value):
            logger.error("CONFIG: Response from IP API is not a valid IP.")
            raise PostProcessingError

        self.config[self.IP] = value

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

    # Validation routines

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

        if not os.path.exists(path):
            logger.error("CONFIG: Validation error: %s for option %s is does not exist" %
                         (path, config_key))
            raise ValidationError

        # Check if the path is writable

        if not os.access(path, os.W_OK):
            logger.error(
                "CONFIG: Validation error: %s for option %s is not a writable" %
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

    def log_config(self):
        for key in self.config:
            logger.info("CONFIG: %s: %s" % (key, self.config[key]))

    def __getattr__(self, item):
        if item in self.config:
            return self.config[item]











