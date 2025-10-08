from dataclasses import dataclass


@dataclass
class RegistrationParams:
    @classmethod
    def burned_register(
        cls,
        netuid: int,
        hotkey_ss58: str,
    ) -> dict:
        """Returns the parameters for the `burned_register`."""
        return {
            "netuid": netuid,
            "hotkey": hotkey_ss58,
        }

    @classmethod
    def register_network(
        cls,
        hotkey_ss58: str,
    ) -> dict:
        """Returns the parameters for the `register_network`."""
        return {"hotkey": hotkey_ss58}

    @classmethod
    def register(
        cls,
        netuid: int,
        coldkey_ss58: str,
        hotkey_ss58: str,
        block_number: int,
        nonce: int,
        work: list[int],
    ) -> dict:
        """Returns the parameters for the `register`."""
        return {
            "coldkey": coldkey_ss58,
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "block_number": block_number,
            "nonce": nonce,
            "work": work,
        }

    @classmethod
    def set_subnet_identity(
        cls,
        netuid: int,
        hotkey_ss58: str,
        subnet_name: str,
        github_repo: str,
        subnet_contact: str,
        subnet_url: str,
        logo_url: str,
        discord: str,
        description: str,
        additional: str,
    ) -> dict:
        """Returns the parameters for the `set_subnet_identity`."""
        return {
            "hotkey": hotkey_ss58,
            "netuid": netuid,
            "subnet_name": subnet_name,
            "github_repo": github_repo,
            "subnet_contact": subnet_contact,
            "subnet_url": subnet_url,
            "logo_url": logo_url,
            "discord": discord,
            "description": description,
            "additional": additional,
        }
