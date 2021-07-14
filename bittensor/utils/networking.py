# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import miniupnpc
import netaddr
import os
import requests
import urllib

from loguru import logger

def int_to_ip(int_val: int) -> str:
    r""" Maps an integer to a unique ip-string 
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
    r""" Maps an ip-string to a unique integer.
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

def ip__str__(ip_type:int, ip_str:str, port:int):
    return "/ipv%i/%s:%i" % (ip_type, ip_str, port)

class ExternalIPNotFound(Exception):
    """ Raised if we cannot attain your external ip from CURL/URLLIB/IPIFY/AWS """

def get_external_ip() -> str:
    r""" Checks CURL/URLLIB/IPIFY/AWS for your external ip.
        Returns:
            external_ip  (:obj:`str` `required`):
                Your routers external facing ip as a string.

        Raises:
            ExternalIPNotFound (Exception):
                Raised if all external ip attempts fail.
    """
    # --- Try curl.
    try:
        external_ip  = os.popen('curl -s ifconfig.me').readline()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except:
        pass

    # --- Try ipify
    try:
        external_ip = requests.get('https://api.ipify.org').text
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except:
        pass

    # --- Try AWS
    try:
        external_ip = requests.get('https://checkip.amazonaws.com').text.strip()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except:
        pass

    # --- Try myip.dnsomatic 
    try:
        external_ip  = os.popen('curl -s myip.dnsomatic.com').readline()
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except:
        pass    

    # --- Try urllib ipv6 (Doesn't work yet)
    try:
        external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except:
        pass

    # --- Try Wikipedia ipv6 (Doesn't work yet)
    try:
        external_ip = requests.get('https://www.wikipedia.org').headers['X-Client-IP']
        assert isinstance(ip_to_int(external_ip), int)
        return str(external_ip)
    except:
        pass

    raise ExternalIPNotFound


class UPNPCException(Exception):
    """ Raised when trying to perform a port mapping on your router. """


def upnpc_delete_port_map(external_port: int):
    r""" Deletes a UPNPC created port mapping from your router.

        Args: 
            external_port (int, `required`):
                The port to un-map from your router.

        Raises:
            UPNPCException (Exception):
                Raised if UPNPC port map delete fails.
    """
    try:
        logger.info('UPNPC: Deleting port map {}', external_port)
        u = miniupnpc.UPnP()
        u.discoverdelay = 200
        u.discover()
        u.selectigd()
        u.deleteportmapping(external_port, 'TCP')
        logger.info('UPNPC: Delete Success')

    except Exception as e:
        raise UPNPCException(e)

def upnpc_create_port_map(port: int):
    r""" Creates a upnpc port map on your router from passed external_port to local port.

        Args: 
            port (int, `required`):
                The local machine port to map from your external port.

        Return:
            external_port (int, `required`):
                The external port mapped to the local port on your machine.

        Raises:
            UPNPCException (Exception):
                Raised if UPNPC port mapping fails, for instance, if upnpc is not enabled on your router.
    """
    try:
        u = miniupnpc.UPnP()
        u.discoverdelay = 200
        logger.debug('UPNPC: Using UPnP to open a port on your router ...')
        logger.debug('UPNPC: Discovering... delay={}ms', u.discoverdelay)
        ndevices = u.discover()
        u.selectigd()
        logger.debug('UPNPC: ' + str(ndevices) + ' device(s) detected')

        ip = u.lanaddr
        external_ip = u.externalipaddress()

        logger.debug('UPNPC: your local ip address: ' + str(ip))
        logger.debug('UPNPC: your external ip address: ' + str(external_ip))
        logger.debug('UPNPC: status = ' + str(u.statusinfo()) + " connection type = " + str(u.connectiontype()))

        # find a free port for the redirection
        external_port = port
        rc = u.getspecificportmapping(external_port, 'TCP')
        while rc != None and external_port < 65536:
            external_port += 1
            rc = u.getspecificportmapping(external_port, 'TCP')
        if rc != None:
            raise UPNPCException("UPNPC: No available external ports for port mapping.")

        logger.info('UPNPC: trying to redirect remote: {}:{} => local: {}:{} over TCP', external_ip, external_port, ip, port)
        u.addportmapping(external_port, 'TCP', ip, port, 'Bittensor: %u' % external_port, '')
        logger.info('UPNPC: Create Success')

        return external_port

    except Exception as e:
        raise UPNPCException(e)