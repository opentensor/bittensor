from rich.console import Console
from rich.table import Table
import requests
import bittensor
import argparse
import subprocess
import re


class CheckEndpointCommand:
    """
    CheckEndpointCommand checks the network connectivity status of predefined Bittensor network endpoints.

    This command is used to diagnose connection issues and ensure that the specified endpoints are reachable
    over the network. It uses the `nc` (netcat) command-line utility to perform a simple test on each endpoint.

    The command aggregates and displays the results in a table format using the `rich` library, providing
    a clear and color-coded status for each endpoint. The table includes the network name, endpoint URL,
    connectivity status, and any messages returned by the `nc` command.

    Usage:
        This class is intended to be used as a command within the Bittensor CLI. It can be invoked by
        running the CLI with the `check` command:

        ```
        btcli subtensor check
        ```
        ┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ Network      ┃ Endpoint                                  ┃ Status  ┃ Message                                                              ┃
        ┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ local        │ ws://127.0.0.1:9944                       │ Failed  │ nc: connectx to 127.0.0.1 port 9944 (tcp) failed: Connection refused │
        │ finney       │ wss://entrypoint-finney.opentensor.ai:443 │ Success │ Connection to entrypoint-finney.opentensor.ai port 443  succeeded!   │
        │ test         │ wss://test.finney.opentensor.ai:443/      │ Success │ Connection to test.finney.opentensor.ai port 443  succeeded!         │
        │ archive      │ wss://archive.chain.opentensor.ai:443/    │ Success │ Connection to archive.chain.opentensor.ai port 443  succeeded!       │
        └──────────────┴───────────────────────────────────────────┴─────────┴──────────────────────────────────────────────────────────────────────┘

    Note:
        This command assumes that the `nc` command is available in the system's environment where the CLI
        is executed. It also assumes that the endpoints defined in the `endpoints` dictionary are in the
        correct format and the `rich` library is available for displaying the output table.
    """

    @staticmethod
    def run(cli):
        console = bittensor.__console__

        endpoints = {
            "local": bittensor.__local_entrypoint__,
            "finney": bittensor.__finney_entrypoint__,
            "test": bittensor.__finney_test_entrypoint__,
            "archive": bittensor.__archive_entrypoint__,
        }

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Network", style="dim", width=12)
        table.add_column("Endpoint")
        table.add_column("Status")
        table.add_column("Message")

        # Regular expression to extract hostname and port from URL
        url_pattern = re.compile(r"(?:wss?://)?([^:/]+)(?::(\d+))?")

        # Iterate over the endpoints and check connectivity
        for network, url in endpoints.items():
            match = url_pattern.match(url)
            if match:
                host, port = match.groups()
                port = port or (
                    "443" if url.startswith("wss://") else "80"
                )  # Default ports for wss and ws respectively
                result = subprocess.run(
                    ["nc", "-vz", host, port],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                if result.returncode == 0:
                    status = "[green]Success[/green]"
                else:
                    status = f"[red]Failed[/red]"
            else:
                status = "[red]Error: Invalid URL[/red]"
            table.add_row(network, url, status, result.stdout.strip())

        console.print(table)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        check_parser = parser.add_parser(
            "check", help="Check connectivity with Bittensor network endpoints."
        )
        # No additional arguments needed for this command.

    @staticmethod
    def check_config(config: argparse.Namespace):
        pass
