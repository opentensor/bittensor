import asyncio

from bittensor.v2.core.cli import (
console, ALIAS_TO_COMMAND, COMMANDS, CLIErrorParser, Cli as CLI_CLASS
)

console = console
ALIAS_TO_COMMAND = ALIAS_TO_COMMAND
COMMANDS = COMMANDS
CLIErrorParser = CLIErrorParser


class Cli:
    def __init__(self, *args, **kwargs):
        self._async_instance = CLI_CLASS(*args, **kwargs)

    def __getattr__(self, item):
        attr = getattr(self._async_instance, item)
        if asyncio.iscoroutinefunction(attr):
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(attr(*args, **kwargs))

            return sync_wrapper
        return attr
