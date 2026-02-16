"""Utils for handling local network with ip and ports."""

import os
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

    Raises:
        netaddr.AddrFormatError: If the integer value is out of valid IP range.
    """
    return str(netaddr.IPAddress(int_val))


def ip_to_int(str_val: str) -> int:
    """Maps an ip-string to a unique integer.

    Parameters:
        str_val: The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Returns:
        int_val: The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

    Raises:
        netaddr.AddrFormatError: If the string is not a valid IP address.
    """
    return int(netaddr.IPAddress(str_val))


def ip_version(str_val: str) -> int:
    """Returns the ip version (IPV4 or IPV6).

    Parameters:
        str_val: The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Returns:
        int_val: The ip version (Either 4 or 6 for IPv4/IPv6)

    Raises:
        netaddr.AddrFormatError: If the string is not a valid IP address.
    """
    return int(netaddr.IPAddress(str_val).version)


def ip__str__(ip_type: int, ip_str: str, port: int) -> str:
    """Return a formatted ip string

    Parameters:
        ip_type: The IP version (4 or 6).
        ip_str: The IP address as a string.
        port: The port number.

    Returns:
        A formatted string of form /ipv<version>/<ip>:<port>
    """
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)


def _validate_ip_response(ip_str: str) -> str:
    """Validate and normalize an IP address string obtained from an external service.

    Parameters:
        ip_str: The IP address string to validate.

    Returns:
        The validated IP address as a string.

    Raises:
        ValueError: If the string is not a valid IP address.
    """
    ip_str = ip_str.strip()
    ip_to_int(ip_str)  # raises netaddr.AddrFormatError if invalid
    return ip_str


def get_external_ip() -> str:
    """Checks CURL/URLLIB/IPIFY/AWS for your external ip.

    Tries multiple external services in sequence. If one fails (due to network
    errors, invalid responses, timeouts, etc.), it falls through to the next.

    Returns:
        external_ip (str): Your routers external facing ip as a string.

    Raises:
        ExternalIPNotFound: Raised if all external ip attempts fail.
    """
    # --- Try AWS
    try:
        external_ip = requests.get(
            "https://checkip.amazonaws.com", timeout=5
        ).text.strip()
        return _validate_ip_response(external_ip)
    except Exception:
        pass

    # --- Try ipconfig.
    try:
        process = os.popen("curl -s ifconfig.me")
        external_ip = process.readline()
        process.close()
        return _validate_ip_response(external_ip)
    except Exception:
        pass

    # --- Try ipinfo.
    try:
        process = os.popen("curl -s https://ipinfo.io")
        external_ip = json.loads(process.read())["ip"]
        process.close()
        return _validate_ip_response(external_ip)
    except Exception:
        pass

    # --- Try myip.dnsomatic
    try:
        process = os.popen("curl -s myip.dnsomatic.com")
        external_ip = process.readline()
        process.close()
        return _validate_ip_response(external_ip)
    except Exception:
        pass

    # --- Try urllib ipv6
    try:
        external_ip = urllib_request.urlopen(
            "https://ident.me", timeout=5
        ).read().decode("utf8")
        return _validate_ip_response(external_ip)
    except Exception:
        pass

    # --- Try Wikipedia
    try:
        external_ip = requests.get(
            "https://www.wikipedia.org", timeout=5
        ).headers["X-Client-IP"]
        return _validate_ip_response(external_ip)
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
