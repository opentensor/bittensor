import time
import logging
from colorama import (
    init,
    Fore,
    Back,
    Style
)

from bittensor.btlogging.helpers import get_max_logger_name_length

init(autoreset=True)

TRACE_LEVELV_NUM = 5
SUCCESS_LEVELV_NUM = 21

def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVELV_NUM):
        self._log(TRACE_LEVELV_NUM, message, args, **kws) 

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVELV_NUM):
        self._log(SUCCESS_LEVELV_NUM, message, args, **kws) 


logging.SUCCESS = SUCCESS_LEVELV_NUM
logging.addLevelName(SUCCESS_LEVELV_NUM, "SUCCESS")
logging.Logger.success = success

logging.TRACE = TRACE_LEVELV_NUM
logging.addLevelName(TRACE_LEVELV_NUM, "TRACE")
logging.Logger.trace = trace


log_level_color_prefix = {
    logging.NOTSET: Fore.RESET,
    logging.TRACE: Fore.MAGENTA,
    logging.DEBUG: Fore.BLUE,
    logging.INFO: Fore.WHITE,
    logging.SUCCESS: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Back.RED,
}


LOG_FORMATS = {
    level: f"{Fore.BLUE}%(asctime)s{Fore.RESET} | {Style.BRIGHT}{color}%(levelname)s\033[0m | %(message)s" 
    for level, color in log_level_color_prefix.items()
}

LOG_TRACE_FORMATS = {
    level: f"{Fore.BLUE}%(asctime)s{Fore.RESET}"\
        f" | {Style.BRIGHT}{color}%(levelname)s{Fore.RESET}{Back.RESET}{Style.RESET_ALL}"\
        f" | %(name)s:%(filename)s:%(lineno)s"\
        f" | %(message)s" 
    for level, color in log_level_color_prefix.items()
}


class BtStreamFormatter(logging.Formatter):
    trace = False

    def formatTime(self, record, datefmt=None):
        """
        Override formatTime to add milliseconds.
        """
        created = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, created)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", created)
        s += ".{:03d}".format(int(record.msecs))
        return s
    
    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt
        spacing = get_max_logger_name_length() + 2
        record.levelname = f"{record.levelname:^16}"
        if self.trace is True:
            self._style._fmt = LOG_TRACE_FORMATS[record.levelno]
        else:
            self._style._fmt = LOG_FORMATS[record.levelno]
        result = super().format(record)
        self._style._fmt = format_orig

        return result
    
    def set_trace(self, state: bool = True):

        self.trace = state