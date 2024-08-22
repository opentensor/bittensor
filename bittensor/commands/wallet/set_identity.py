import argparse
from rich.table import Table
from rich.prompt import Prompt
from sys import getsizeof

import bittensor


class SetIdentityCommand:
    """
    Executes the :func:`set_identity` command within the Bittensor network, which allows for the creation or update of a delegate's on-chain identity.

    This identity includes various
    attributes such as display name, legal name, web URL, PGP fingerprint, and contact
    information, among others.

    Optional Arguments:
        - ``display``: The display name for the identity.
        - ``legal``: The legal name for the identity.
        - ``web``: The web URL for the identity.
        - ``riot``: The riot handle for the identity.
        - ``email``: The email address for the identity.
        - ``pgp_fingerprint``: The PGP fingerprint for the identity.
        - ``image``: The image URL for the identity.
        - ``info``: The info for the identity.
        - ``twitter``: The X (twitter) URL for the identity.

    The command prompts the user for the different identity attributes and validates the
    input size for each attribute. It provides an option to update an existing validator
    hotkey identity. If the user consents to the transaction cost, the identity is updated
    on the blockchain.

    Each field has a maximum size of 64 bytes. The PGP fingerprint field is an exception
    and has a maximum size of 20 bytes. The user is prompted to enter the PGP fingerprint
    as a hex string, which is then converted to bytes. The user is also prompted to enter
    the coldkey or hotkey ``ss58`` address for the identity to be updated. If the user does
    not have a hotkey, the coldkey address is used by default.

    If setting a validator identity, the hotkey will be used by default. If the user is
    setting an identity for a subnet, the coldkey will be used by default.

    Usage:
        The user should call this command from the command line and follow the interactive
        prompts to enter or update the identity information. The command will display the
        updated identity details in a table format upon successful execution.

    Example usage::

        btcli wallet set_identity

    Note:
        This command should only be used if the user is willing to incur the 1 TAO transaction
        fee associated with setting an identity on the blockchain. It is a high-level command
        that makes changes to the blockchain state and should not be used programmatically as
        part of other scripts or applications.
    """

    def run(cli: "bittensor.cli"):
        r"""Create a new or update existing identity on-chain."""
        try:
            subtensor: "bittensor.subtensor" = bittensor.subtensor(
                config=cli.config, log_verbose=False
            )
            SetIdentityCommand._run(cli, subtensor)
        finally:
            if "subtensor" in locals():
                subtensor.close()
                bittensor.logging.debug("closing subtensor connection")

    def _run(cli: "bittensor.cli", subtensor: "bittensor.subtensor"):
        r"""Create a new or update existing identity on-chain."""
        console = bittensor.__console__

        wallet = bittensor.wallet(config=cli.config)

        id_dict = {
            "display": cli.config.display,
            "legal": cli.config.legal,
            "web": cli.config.web,
            "pgp_fingerprint": cli.config.pgp_fingerprint,
            "riot": cli.config.riot,
            "email": cli.config.email,
            "image": cli.config.image,
            "twitter": cli.config.twitter,
            "info": cli.config.info,
        }

        for field, string in id_dict.items():
            if getsizeof(string) > 113:  # 64 + 49 overhead bytes for string
                raise ValueError(f"Identity value `{field}` must be <= 64 raw bytes")

        identified = (
            wallet.hotkey.ss58_address
            if str(
                Prompt.ask(
                    "Are you updating a validator hotkey identity?",
                    default="y",
                    choices=["y", "n"],
                )
            ).lower()
            == "y"
            else None
        )

        if (
            str(
                Prompt.ask(
                    "Cost to register an Identity is [bold white italic]0.1 Tao[/bold white italic], are you sure you wish to continue?",
                    default="n",
                    choices=["y", "n"],
                )
            ).lower()
            == "n"
        ):
            console.print(":cross_mark: Aborted!")
            exit(0)

        wallet.coldkey  # unlock coldkey
        with console.status(":satellite: [bold green]Updating identity on-chain..."):
            try:
                subtensor.update_identity(
                    identified=identified,
                    wallet=wallet,
                    params=id_dict,
                )
            except Exception as e:
                console.print(f"[red]:cross_mark: Failed![/red] {e}")
                exit(1)

            console.print(":white_heavy_check_mark: Success!")

        identity = subtensor.query_identity(identified or wallet.coldkey.ss58_address)

        table = Table(title="[bold white italic]Updated On-Chain Identity")
        table.add_column("Key", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Address", identified or wallet.coldkey.ss58_address)
        for key, value in identity.items():
            table.add_row(key, str(value) if value is not None else "None")

        console.print(table)

    @staticmethod
    def check_config(config: "bittensor.config"):
        if not config.is_set("wallet.name") and not config.no_prompt:
            config.wallet.name = Prompt.ask(
                "Enter [bold dark_green]coldkey[/bold dark_green] name", default=bittensor.defaults.wallet.name
            )
        if not config.is_set("wallet.hotkey") and not config.no_prompt:
            config.wallet.hotkey = Prompt.ask(
                "Enter wallet hotkey", default=bittensor.defaults.wallet.hotkey
            )
        if not config.is_set("subtensor.network") and not config.no_prompt:
            config.subtensor.network = Prompt.ask(
                "Enter subtensor network",
                default=bittensor.defaults.subtensor.network,
                choices=bittensor.__networks__,
            )
            (
                _,
                config.subtensor.chain_endpoint,
            ) = bittensor.subtensor.determine_chain_endpoint_and_network(
                config.subtensor.network
            )
        if not config.is_set("display") and not config.no_prompt:
            config.display = Prompt.ask("Enter display name", default="")
        if not config.is_set("legal") and not config.no_prompt:
            config.legal = Prompt.ask("Enter legal string", default="")
        if not config.is_set("web") and not config.no_prompt:
            config.web = Prompt.ask("Enter web url", default="")
        if not config.is_set("pgp_fingerprint") and not config.no_prompt:
            config.pgp_fingerprint = Prompt.ask(
                "Enter pgp fingerprint (must be 20 bytes)", default=None
            )
        if not config.is_set("riot") and not config.no_prompt:
            config.riot = Prompt.ask("Enter riot", default="")
        if not config.is_set("email") and not config.no_prompt:
            config.email = Prompt.ask("Enter email address", default="")
        if not config.is_set("image") and not config.no_prompt:
            config.image = Prompt.ask("Enter image url", default="")
        if not config.is_set("twitter") and not config.no_prompt:
            config.twitter = Prompt.ask("Enter twitter url", default="")
        if not config.is_set("info") and not config.no_prompt:
            config.info = Prompt.ask("Enter info", default="")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        new_coldkey_parser = parser.add_parser(
            "set_identity",
            help="""Create or update identity on-chain for a given cold wallet. Must be a subnet owner.""",
        )
        new_coldkey_parser.add_argument(
            "--display",
            type=str,
            help="""The display name for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--legal",
            type=str,
            help="""The legal name for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--web",
            type=str,
            help="""The web url for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--riot",
            type=str,
            help="""The riot handle for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--email",
            type=str,
            help="""The email address for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--pgp_fingerprint",
            type=str,
            help="""The pgp fingerprint for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--image",
            type=str,
            help="""The image url for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--info",
            type=str,
            help="""The info for the identity.""",
        )
        new_coldkey_parser.add_argument(
            "--twitter",
            type=str,
            help="""The twitter url for the identity.""",
        )
        bittensor.wallet.add_args(new_coldkey_parser)
        bittensor.subtensor.add_args(new_coldkey_parser)
