"""Utils for handling local network with ip and ports."""

import os
import subprocess
from typing import Optional
from urllib import request as urllib_request

import netaddr
import requests
from async_substrate_interface.utils import json


class ExternalIPNotFound(Exception):
    """Raised if we cannot attain your external ip from CURL/URLLIB/IPIFY/AWS"""


def int_to_ip(int_val: int) -> str:
    """Maps an integer to a unique ip-string

    Parameters:
        int_val: The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

    Returns:
        str_val: The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6
    """
    return str(netaddr.IPAddress(int_val))


def ip_to_int(str_val: str) -> int:
    """Maps an ip-string to a unique integer.

    Parameters:
        str_val: The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Returns:
        int_val: The integer representation of an ip. Must be in the range (0, 3.4028237e+38).
    """
    return int(netaddr.IPAddress(str_val))


def ip_version(str_val: str) -> int:
    """Returns the ip version (IPV4 or IPV6).

    Parameters:
        str_val: The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Returns:
        int_val: The ip version (Either 4 or 6 for IPv4/IPv6)
    """
    return int(netaddr.IPAddress(str_val).version)


def ip__str__(ip_type: int, ip_str: str, port: int):
    """Return a formatted ip string"""
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)


def get_external_ip() -> str:
    """Checks CURL/URLLIB/IPIFY/AWS for your external ip.

    Returns:
        external_ip (str): Your routers external facing ip as a string.

    Raises:
        ExternalIPNotFound(Exception): Raised if all external ip attempts fail.
    """
    # --- Try AWS
    try:
        external_ip = requests.get("https://checkip.amazonaws.com").text.strip()
        ip_to_int(external_ip)
        return str(external_ip)
    except Exception:
        pass

    # --- Try ifconfig.me
    try:
        result = subprocess.run(
            ["curl", "-s", "ifconfig.me"], capture_output=True, text=True, timeout=10
        )
        external_ip = result.stdout.strip()
        ip_to_int(external_ip)
        return str(external_ip)
    except Exception:
        pass

    # --- Try ipinfo.
    try:
        result = subprocess.run(
            ["curl", "-s", "https://ipinfo.io"], capture_output=True, text=True, timeout=10
        )
        external_ip = json.loads(result.stdout)["ip"]
        ip_to_int(external_ip)
        return str(external_ip)
    except Exception:
        pass

    # --- Try myip.dnsomatic
    try:
        result = subprocess.run(
            ["curl", "-s", "myip.dnsomatic.com"], capture_output=True, text=True, timeout=10
        )
        external_ip = result.stdout.strip()
        ip_to_int(external_ip)
        return str(external_ip)
    except Exception:
        pass

    # --- Try urllib ipv6
    try:
        external_ip = urllib_request.urlopen("https://ident.me").read().decode("utf8")
        ip_to_int(external_ip)
        return str(external_ip)
    except Exception:
        pass

    # --- Try Wikipedia
    try:
        external_ip = requests.get("https://www.wikipedia.org").headers["X-Client-IP"]
        ip_to_int(external_ip)
        return str(external_ip)
    except Exception:
        pass

    raise ExternalIPNotFound


def get_formatted_ws_endpoint_url(endpoint_url: Optional[str]) -> Optional[str]:
    """
    Returns a formatted websocket endpoint url.

    Parameters:
        endpoint_url: The endpoint url to format.

    Returns:
        formatted_endpoint_url: The formatted endpoint url. In the form of ws://<endpoint_url> or wss://<endpoint_url>

    Note: The port (or lack thereof) is left unchanged.
    """
    if endpoint_url is None:
        return None

    if endpoint_url[0:6] != "wss://" and endpoint_url[0:5] != "ws://":
        endpoint_url = f"ws://{endpoint_url}"

    return endpoint_url
