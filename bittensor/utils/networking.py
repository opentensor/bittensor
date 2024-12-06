"""Utils for handling local network with ip and ports."""

import asyncio
import json
import os
import urllib
from functools import wraps
from typing import Optional

import netaddr
import requests
from retry import retry
from websockets.exceptions import ConnectionClosed

from bittensor.utils.btlogging import logging


def int_to_ip(int_val: int) -> str:
    """Maps an integer to a unique ip-string
    Args:
        int_val  (int):
            The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

    Returns:
        str_val (str):
            The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Raises:
        netaddr.core.AddrFormatError (Exception): Raised when the passed int_vals is not a valid ip int value.
    """
    return str(netaddr.IPAddress(int_val))


def ip_to_int(str_val: str) -> int:
    """Maps an ip-string to a unique integer.
    arg:
        str_val (:tyep:`str`, `required):
            The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Returns:
        int_val  (:type:`int128`, `required`):
            The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

    Raises:
        netaddr.core.AddrFormatError (Exception):
            Raised when the passed str_val is not a valid ip string value.
    """
    return int(netaddr.IPAddress(str_val))


def ip_version(str_val: str) -> int:
    """Returns the ip version (IPV4 or IPV6).
    arg:
        str_val (:tyep:`str`, `required):
            The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Returns:
        int_val  (:type:`int128`, `required`):
            The ip version (Either 4 or 6 for IPv4/IPv6)

    Raises:
        netaddr.core.AddrFormatError (Exception):
            Raised when the passed str_val is not a valid ip string value.
    """
    return int(netaddr.IPAddress(str_val).version)


def ip__str__(ip_type: int, ip_str: str, port: int):
    """Return a formatted ip string"""
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)


class ExternalIPNotFound(Exception):
    """Raised if we cannot attain your external ip from CURL/URLLIB/IPIFY/AWS"""


def get_external_ip() -> str:
    """Checks CURL/URLLIB/IPIFY/AWS for your external ip.
    Returns:
        external_ip  (:obj:`str` `required`):
            Your routers external facing ip as a string.

    Raises:
        ExternalIPNotFound (Exception):
            Raised if all external ip attempts fail.
    """
    # --- Try AWS
    try:
        external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try ipconfig.
    try:
        process = os.popen("curl -s ifconfig.me")
        external_ip = process.readline()
        process.close()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try ipinfo.
    try:
        process = os.popen("curl -s https://ipinfo.io")
        external_ip = json.loads(process.read())["ip"]
        process.close()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try myip.dnsomatic
    try:
        process = os.popen("curl -s myip.dnsomatic.com")
        external_ip = process.readline()
        process.close()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try urllib ipv6
    try:
        external_ip = urllib.request.urlopen("https://ident.me").read().decode("utf8")
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    # --- Try Wikipedia
    try:
        external_ip = requests.get("https://www.wikipedia.org").headers["X-Client-IP"]
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except Exception:
        pass

    raise ExternalIPNotFound


def get_formatted_ws_endpoint_url(endpoint_url: Optional[str]) -> Optional[str]:
    """
    Returns a formatted websocket endpoint url.
    Note: The port (or lack thereof) is left unchanged
    Args:
        endpoint_url (Optional[str]):
            The endpoint url to format.
    Returns:
        formatted_endpoint_url (Optional[str]): The formatted endpoint url. In the form of ws://<endpoint_url> or wss://<endpoint_url>
    """
    if endpoint_url is None:
        return None

    if endpoint_url[0:6] != "wss://" and endpoint_url[0:5] != "ws://":
        endpoint_url = f"ws://{endpoint_url}"

    return endpoint_url


def ensure_connected(func):
    """Decorator ensuring the function executes with an active substrate connection."""

    # TODO we need to rethink the logic in this

    def is_connected(substrate) -> bool:
        """Check if the substrate connection is active."""
        try:
            asyncio.get_event_loop().run_until_complete(
                asyncio.wait_for(substrate.ws.ws.ping(), timeout=7.0)
            )
            return True
        except (TimeoutError, ConnectionClosed, AttributeError):
            return False

    @retry(
        exceptions=ConnectionRefusedError,
        tries=5,
        delay=5,
        backoff=1,
        logger=logging,
    )
    def reconnect_with_retries(self):
        """Attempt to reconnect with retries using retry library."""
        logging.console.info("Attempting to reconnect to substrate...")
        self._get_substrate()
        logging.console.success("Connection successfully restored!")

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper function where `self` is expected to be a Subtensor instance."""
        if not is_connected(self.substrate):
            logging.debug("Substrate connection inactive. Attempting to reconnect...")
            self._get_substrate()

        try:
            return func(self, *args, **kwargs)
        except ConnectionClosed:
            logging.console.warning(
                "WebSocket connection closed. Attempting to reconnect 5 times..."
            )
            try:
                reconnect_with_retries(self)
                return func(self, *args, **kwargs)
            except ConnectionRefusedError:
                logging.critical("Unable to restore connection. Raising exception.")
                raise ConnectionRefusedError("Failed to reconnect to substrate.")

    return wrapper
