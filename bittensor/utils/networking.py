""" Utils for handling local network with ip and ports.
"""
# The MIT License (MIT)
# Copyright © 2021-2022 Yuma Rao
# Copyright © 2022-2023 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import urllib
import json
import netaddr
import requests

from loguru import logger


def int_to_ip(int_val: int) -> str:
    r"""Maps an integer to a unique ip-string
    Args:
        int_val  (:type:`int128`, `required`):
            The integer representation of an ip. Must be in the range (0, 3.4028237e+38).

    Returns:
        str_val (:tyep:`str`, `required):
            The string representation of an ip. Of form *.*.*.* for ipv4 or *::*:*:*:* for ipv6

    Raises:
        netaddr.core.AddrFormatError (Exception):
            Raised when the passed int_vals is not a valid ip int value.
    """
    return str(netaddr.IPAddress(int_val))


def ip_to_int(str_val: str) -> int:
    r"""Maps an ip-string to a unique integer.
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
    r"""Returns the ip version (IPV4 or IPV6).
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
    r"""Checks CURL/URLLIB/IPIFY/AWS for your external ip.
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


def get_formatted_ws_endpoint_url(endpoint_url: str) -> str:
    """
    Returns a formatted websocket endpoint url.
    Note: The port (or lack thereof) is left unchanged
    Args:
        endpoint_url (str, `required`):
            The endpoint url to format.
    Returns:
        formatted_endpoint_url (str, `required`):
            The formatted endpoint url. In the form of ws://<endpoint_url> or wss://<endpoint_url>
    """
    if endpoint_url[0:6] != "wss://" and endpoint_url[0:5] != "ws://":
        endpoint_url = "ws://{}".format(endpoint_url)

    return endpoint_url
