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
import rollbar
import sys
import bittensor
import torch

from loguru import logger
logger = logger.opt(colors=True)

# Handler which sends messages to a rollbar server.
class RollbarHandler:
    def write(self, message):
        record = message.record
        if record['level'].name == "WARNING":
            rollbar.report_message(message, 'warning')
        elif record['level'].name == "ERROR":
            rollbar.report_message(message, 'error')
        else:
            pass

class logging:
    __debug_on__:bool = False
    __trace_on__:bool = False
    __sink__:int = None

    @classmethod
    def set_debug(cls, on: bool = True ):
        cls.__debug_on__ = on

    @classmethod
    def set_trace(cls, on: bool = True):
        cls._trace_on__ = on

    @classmethod
    def log_filter(cls, record ):
        if cls.__debug_on__ or cls.__trace_on__:
            return True
        elif record["level"].no >= logger.level('INFO').no:
            return True
        else:
            return False

    @classmethod
    def log_formatter(cls, record):
        extra = record['extra']
        if 'rpc' in extra:
            log_format = "<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue> | " + extra['code_str'] + " | {extra[prefix]} | {extra[direction]} | {extra[arrow]} | {extra[inputs]} | {extra[key_str]} | {extra[rpc_message]} \n"
            return log_format
        if 'receptor' in extra:
            log_format = "<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue> | " + extra['action'] + " | uid:{extra[uid]} | {extra[ip_str]} | hotkey:{extra[hotkey]} | coldkey:{extra[coldkey]} \n"
            return log_format
        else:
            return "<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue> | <level>{level: ^16}</level> | {message}\n"

    @classmethod
    def init(cls):
        # Remove all logger sinks.
        logger.remove( 0 )
        if cls.__sink__ != None:
            logger.remove( cls.__sink__ )

        # Add filtered sys.stdout.
        cls.__sink__ = logger.add ( 
            sys.stdout, 
            filter = cls.log_filter, 
            colorize = True, 
            enqueue = True, 
            backtrace = True, 
            diagnose = True, 
            format = cls.log_formatter
        )

        # Add filtered rollbar handler.
        rollbar_token = os.environ.get("ROLLBAR_TOKEN", False)
        rollbar_env = "production"
        rollbar_handler = RollbarHandler()
        if rollbar_token:
            # Rollbar is enabled.
            logger.info("Error reporting enabled using {}:{}", rollbar_token, rollbar_env)
            rollbar.init(rollbar_token, rollbar_env)
            logger.add (
                sink = rollbar_handler,
                level = 'WARNING',
                colorize = True, 
                enqueue = True, 
                backtrace = True, 
                diagnose = True, 
            )
        
        # Return internal logger
        return logger.bind( internal=True )

    @classmethod
    def rpc_log( cls, axon: bool, forward: bool, is_response: bool, code:int, pubkey: str, inputs:list = [], outputs:list = [], message:str = ''):
        if axon:
            prefix = "Axon"
        else:
            prefix = "Dendrite"
        prefix = prefix.center(len('Dendrite'))

        if forward:
            direction = "Forward"
        else:
            direction = "Backward"
        direction = direction.center(len('Backward'))

        if is_response:
            arrow = "<---"
        else:
            arrow = "--->"
        key_str = "{}".format( pubkey )

        code_color = bittensor.utils.codes.code_to_loguru_color( code )
        code_string = bittensor.utils.codes.code_to_string( code )
        code_string = code_string.center(16)
        code_str = "<" + code_color + ">" + code_string + "</" + code_color + ">"

        if is_response:
            inputs = str(list(outputs)) if outputs != None else '[x]'
        else:
            inputs = str(list(inputs)) if inputs != None else '[x]'
        inputs = inputs.center(15)

        rpc_message = message if message != None else 'None'

        logger.debug( 'rpc', rpc=True, prefix=prefix, direction=direction, arrow=arrow, key_str=key_str, code_str=code_str, inputs = inputs, rpc_message = rpc_message)


    @classmethod
    def create_receptor_log( cls, endpoint: 'bittensor.Endpoint' ):
        logger.debug( 'endpoint', receptor=True, action = '<green>' + 'Connect'.center(16) + '</green>', uid=str(endpoint.uid).center(4), hotkey=endpoint.hotkey, coldkey=endpoint.coldkey, ip_str=endpoint.ip_str().center(27) )

    @classmethod
    def update_receptor_log( cls, endpoint: 'bittensor.Endpoint' ):
        logger.debug( 'endpoint', receptor=True, action = '<blue>' + 'Update'.center(16) + '</blue>', uid=str(endpoint.uid).center(4), hotkey=endpoint.hotkey,  coldkey=endpoint.coldkey, ip_str=endpoint.ip_str().center(27) )

    @classmethod
    def destroy_receptor_log( cls, endpoint: 'bittensor.Endpoint' ):
        logger.debug( 'endpoint', receptor=True, action = '<red>' + 'Destroy'.center(16) + '</red>', uid=str(endpoint.uid).center(4), hotkey=endpoint.hotkey,  coldkey=endpoint.coldkey, ip_str=endpoint.ip_str().center(27) )


logging.init()

