import time
import logging
from colorama import init, Fore, Back, Style


init(autoreset=True)

TRACE_LEVEL_NUM = 5
SUCCESS_LEVEL_NUM = 21


def trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)


logging.SUCCESS = SUCCESS_LEVEL_NUM
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
logging.Logger.success = success

logging.TRACE = TRACE_LEVEL_NUM
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.Logger.trace = trace

emoji_map = {
    ":white_heavy_check_mark:": "✅",
    ":cross_mark:": "❌",
    ":satellite:": "🛰️",
}


color_map = {
    "<red>": Fore.RED,
    "</red>": Style.RESET_ALL,
    "<blue>": Fore.BLUE,
    "</blue>": Style.RESET_ALL,
    "<green>": Fore.GREEN,
    "</green>": Style.RESET_ALL,
}


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
    level: f"{Fore.BLUE}%(asctime)s{Fore.RESET}"
    f" | {Style.BRIGHT}{color}%(levelname)s{Fore.RESET}{Back.RESET}{Style.RESET_ALL}"
    f" | %(name)s:%(filename)s:%(lineno)s"
    f" | %(message)s"
    for level, color in log_level_color_prefix.items()
}


class BtStreamFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = False

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
        record.levelname = f"{record.levelname:^16}"

        if self.trace is True:
            self._style._fmt = LOG_TRACE_FORMATS[record.levelno]
        else:
            self._style._fmt = LOG_FORMATS[record.levelno]

        for text, emoji in emoji_map.items():
            record.msg = record.msg.replace(text, emoji)
        # Apply color specifiers
        for text, color in color_map.items():
            record.msg = record.msg.replace(text, color)

        result = super().format(record)
        self._style._fmt = format_orig

        return result

    def set_trace(self, state: bool = True):
        self.trace = state


class BtFileFormatter(logging.Formatter):
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
        record.levelname = f"{record.levelname:^16}"
        return super().format(record)
