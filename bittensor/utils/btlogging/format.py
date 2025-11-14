"""
btlogging.format module

This module defines custom logging formatters for the Bittensor project.
"""

import logging
import time
from typing import Optional
from colorama import init, Fore, Back, Style

init(wrap=False)

TRACE_LEVEL_NUM: int = 5
SUCCESS_LEVEL_NUM: int = 21


def _trace(self, message: str, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


def _success(self, message: str, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)


logging.SUCCESS = SUCCESS_LEVEL_NUM
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")
logging.Logger.success = _success

logging.TRACE = TRACE_LEVEL_NUM
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.Logger.trace = _trace

emoji_map: dict[str, str] = {
    ":white_heavy_check_mark:": "✅",
    ":cross_mark:": "❌",
    ":satellite:": "🛰️",
    ":warning:": "⚠️",
    ":arrow_right:": "➡️",
    ":hourglass:": "⏳",
}


color_map: dict[str, str] = {
    "[red]": Fore.RED,
    "[/red]": Style.RESET_ALL,
    "[blue]": Fore.BLUE,
    "[/blue]": Style.RESET_ALL,
    "[green]": Fore.GREEN,
    "[/green]": Style.RESET_ALL,
    "[magenta]": Fore.MAGENTA,
    "[/magenta]": Style.RESET_ALL,
    "[yellow]": Fore.YELLOW,
    "[/yellow]": Style.RESET_ALL,
    "[orange]": Fore.YELLOW,
    "[/orange]": Style.RESET_ALL,
}


log_level_color_prefix: dict[int, str] = {
    logging.NOTSET: Fore.RESET,
    logging.TRACE: Fore.MAGENTA,
    logging.DEBUG: Fore.BLUE,
    logging.INFO: Fore.WHITE,
    logging.SUCCESS: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Back.RED,
}


LOG_FORMATS: dict[int, str] = {
    level: f"{Fore.BLUE}%(asctime)s{Fore.RESET} | {Style.BRIGHT}{color}%(levelname)s\033[0m | %(message)s"
    for level, color in log_level_color_prefix.items()
}

LOG_TRACE_FORMATS: dict[int, str] = {
    level: f"{Fore.BLUE}%(asctime)s{Fore.RESET}"
    f" | {Style.BRIGHT}{color}%(levelname)s{Fore.RESET}{Back.RESET}{Style.RESET_ALL}"
    f" | %(name)s:%(filename)s:%(lineno)s"
    f" | %(message)s"
    for level, color in log_level_color_prefix.items()
}

DEFAULT_LOG_FORMAT: str = (
    f"{Fore.BLUE}%(asctime)s{Fore.RESET} | "
    f"{Style.BRIGHT}{Fore.WHITE}%(levelname)s{Style.RESET_ALL} | "
    f"%(name)s:%(filename)s:%(lineno)s | %(message)s"
)

DEFAULT_TRACE_FORMAT: str = (
    f"{Fore.BLUE}%(asctime)s{Fore.RESET} | "
    f"{Style.BRIGHT}{Fore.WHITE}%(levelname)s{Style.RESET_ALL} | "
    f"%(name)s:%(filename)s:%(lineno)s | %(message)s"
)


class BtStreamFormatter(logging.Formatter):
    """
    A custom logging formatter for the Bittensor project that overrides the time formatting to include milliseconds,
    centers the level name, and applies custom log formats, emojis, and colors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = False

    def formatTime(self, record, datefmt: Optional[str] = None) -> str:
        """
        Override formatTime to add milliseconds.

        Parameters:
            record: The log record.
            datefmt: The date format string.

        Returns:
            The formatted time string with milliseconds.
        """

        created = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, created)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", created)
        s += f".{int(record.msecs):03d}"
        return s

    def format(self, record: "logging.LogRecord") -> str:
        """
        Override format to apply custom formatting including emojis and colors.

        This method saves the original format, applies custom formatting based on the log level and trace flag, replaces
        text with emojis and colors, and then returns the formatted log record.

        Parameters:
            record: The log record.

        Returns:
            The formatted log record.
        """

        format_orig = self._style._fmt
        record.levelname = f"{record.levelname:^8}"

        if record.levelno not in LOG_FORMATS:
            self._style._fmt = (
                DEFAULT_TRACE_FORMAT if self.trace else DEFAULT_LOG_FORMAT
            )
        else:
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
        """Change formatter state."""
        self.trace = state


class BtFileFormatter(logging.Formatter):
    """
    BtFileFormatter

    A custom logging formatter for the Bittensor project that overrides the time formatting to include milliseconds and
    centers the level name.
    """

    def formatTime(
        self, record: "logging.LogRecord", datefmt: Optional[str] = None
    ) -> str:
        """
        Override formatTime to add milliseconds.

        Parameters:
            record: The log record.
            datefmt: The date format string.

        Returns:
            The formatted time string with milliseconds.
        """

        created = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, created)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", created)
        s += f".{int(record.msecs):03d}"
        return s

    def format(self, record: "logging.LogRecord") -> str:
        """
        Override format to center the level name.

        Parameters:
            record: The log record.

        Returns:
            The formatted log record.
        """
        record.levelname = f"{record.levelname:^10}"
        return super().format(record)
