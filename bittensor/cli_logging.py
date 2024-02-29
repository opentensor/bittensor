import sys
import time
import copy

from io import StringIO
from threading import Thread, Event
from loguru import logger
from rich.console import Console as RichConsole

logger.remove()
consolelogger = copy.deepcopy(logger)


class ConsoleLogger:
    """CLI logging using loguru"""

    # todo: highlighting (tensors, json), table printing, mock console
    logger: object = None
    file: object = None
    live_status: object = None
    off: bool = False
    width: int = None

    def __init__(self, file: object = None, width: int = None):
        self.logger = consolelogger
        self.logger = self.logger.opt(colors=True)
        self.file = file
        self.width = width
        # Remove default sink.
        try:
            self.logger.remove(0)
        except Exception:
            pass

        # Remove other sinks.
        for handler_id in list(self.logger._core.handlers):
            self.logger.remove(handler_id)

        if not self.file:
            self.file = sys.stdout

        # simple handler that prints to stdout, no debug, or trace
        self.logger.add(
            self.file,
            level=0,
            filter=lambda r: True if 20 <= r["level"].no else False,
            colorize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format=lambda r: "{message}",
        )

    def turn_off(self):
        """Turn off all logging output."""
        self.off = True

        for handler_id in list(self.logger._core.handlers):
            self.logger.remove(handler_id)

    def turn_on(self):
        """Turn on logging output by re-adding the sinks."""
        if self.off:
            self.off = False

            if not self.file:
                self.file = sys.stdout

            # simple handler that prints to stdout, no debug, or trace
            self.logger.add(
                self.file,
                level=0,
                filter=lambda r: True if 20 <= r["level"].no else False,
                colorize=True,
                enqueue=True,
                backtrace=False,
                diagnose=False,
                format=lambda r: "{message}",
            )

    def success(self, message: object, sufix: object = None):
        """Print sucess message to console"""
        if sufix is not None:
            self.print(f"\u2714 <green>{message}</green>{sufix}\n")
        else:
            self.print(f"\u2714 <green>{message}</green>\n")

    def warning(self, message: object):
        """Print warning message to console"""
        self.print(f"\u26A0 <yellow>{message}</yellow>\n")

    def error(self, message: object, error: object = None):
        """Print fail message to console"""
        if error:
            # todo: add highlighting to json errors
            self.print(f"\u274c <red>{message}</red>: error: {error}\n")
        else:
            self.print(f"\u274c <red>{message}</red>\n")

    def info(self, message: object):
        """Print warning message to console"""
        self.print(f"{message}\n")

    def print(self, *args):
        """Print raw messages to console"""
        if self.live_status is not None:
            self.live_status.stop()
            # clear status before printing output
            self.logger.info("\033[2K\r")
            self.live_status = None

        # todo: add highlighting for objects (e. weights)
        self.logger.info(*args)

    def status(self, message: object, icon: str = None):
        """Status logging"""
        self.live_status = StatusLogger(self.logger, message, icon)
        return self.live_status

    def clear(self):
        self.print("\033[2J\n")

    def rich_print(self, text, **kwargs):
        # todo: implement width for tables, other kargs ?
        self.print(self.generate_printable(text, **kwargs))

    def generate_printable(self, *args, **kwargs):
        c = RichConsole(file=StringIO(), force_terminal=True, width=self.width)
        c.print(*args, **kwargs)
        return c.file.getvalue()


class StatusLogger:
    """Replicates rich status command, and spinner animation"""

    ANIMATION_FRAMES: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, console, message: object, icon: str = None):
        self.logger = console
        self.message = message
        self.frame_index = 0
        self.thread = None
        self.done = Event()
        self.icon = "\U0001F4E1"

        if icon is not None:
            self.icon = icon

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        # make cursor invisible
        self.logger.info("\033[?25l")
        self.logger.info(f"{self.ANIMATION_FRAMES[0]} {self.icon} {self.message}")

        # Start animation thread
        self.thread = Thread(target=self._update_animation, args=(self.done,))
        self.thread.start()

    def stop(self):
        # make cursor visible
        self.logger.info("\033[?25h")
        self.done.set()

    def _update_animation(self, event):
        """Updates the frames, deletes the current line, stopped by event"""
        while not self.done.is_set():
            self.frame_index = (self.frame_index + 1) % len(self.ANIMATION_FRAMES)
            frame = self.ANIMATION_FRAMES[self.frame_index]
            self.logger.info(f"\033[2K\r{frame} {self.icon} {self.message}")
            time.sleep(0.1)
