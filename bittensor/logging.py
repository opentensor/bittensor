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

from loguru import logger

# Filter bittensor internal messages, only from internal files.
def bittensor_formatter(record):
    if bittensor.__debug_on__ == True:
        return "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>\n"
    else:
        return "<level>{message}</level>\n"

def bittensor_log_filter( record ):
    if bittensor.__debug_on__ == True:
        return True
    elif record["level"].no >= logger.level('INFO').no:
        return True
    else:
        print(bittensor.__debug_on__, record["level"].no)
        return False

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

def init_logger():
    # Remove all logger sinks.
    logger.remove()

    # Add custom levels.
    logger.level("USER-SUCCESS", no=33, icon="s")
    logger.level("USER-CRITICAL", no=34, icon="c")
    logger.level("USER-ACTION", no=35, icon="a") 
    logger.level("USER-INFO", no=36, icon="i") 

    # Add filtered sys.stdout.
    logger.add( 
        sys.stdout, 
        filter = bittensor_log_filter, 
        colorize = True, 
        enqueue = True, 
        backtrace = True, 
        diagnose = True, 
        format = bittensor_formatter
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
