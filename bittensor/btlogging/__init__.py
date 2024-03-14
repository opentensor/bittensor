""" 
Standardized logging for Bittensor.
"""

# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import os
import argparse

import bittensor.config
from bittensor.btlogging.loggingmachine import LoggingMachine


def add_args(parser: argparse.ArgumentParser, prefix: str = None):
    """Accept specific arguments fro parser"""
    prefix_str = "" if prefix == None else prefix + "."
    try:
        default_logging_debug = os.getenv("BT_LOGGING_DEBUG") or False
        default_logging_trace = os.getenv("BT_LOGGING_TRACE") or False
        default_logging_record_log = os.getenv("BT_LOGGING_RECORD_LOG") or False
        default_logging_logging_dir = (
            os.getenv("BT_LOGGING_LOGGING_DIR") or "~/.bittensor/miners"
        )
        parser.add_argument(
            "--" + prefix_str + "logging.debug",
            action="store_true",
            help="""Turn on bittensor debugging information""",
            default=default_logging_debug,
        )
        parser.add_argument(
            "--" + prefix_str + "logging.trace",
            action="store_true",
            help="""Turn on bittensor trace level information""",
            default=default_logging_trace,
        )
        parser.add_argument(
            "--" + prefix_str + "logging.record_log",
            action="store_true",
            help="""Turns on logging to file.""",
            default=default_logging_record_log,
        )
        parser.add_argument(
            "--" + prefix_str + "logging.logging_dir",
            type=str,
            help="Logging default root directory.",
            default=default_logging_logging_dir,
        )
    except argparse.ArgumentError:
        # re-parsing arguments.
        pass

def config():
    """Get config from the argument parser.

    Return:
        bittensor.config object
    """
    parser = argparse.ArgumentParser()
    add_args(parser)
    return bittensor.config(parser, args=[])


logging = LoggingMachine(config())