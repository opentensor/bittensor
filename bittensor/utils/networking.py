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

def ip__str__(ip_type, ip_str):
    return "/ipv%i/%s" % (ip_type, ip_str)

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
        return str(external_ip)
    except:
        pass

    # --- Try urllib
    try:
        external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
        return str(external_ip)
    except:
        pass

    # --- Try ipify
    try:
        external_ip = requests.get('https://api.ipify.org').text
        return str(external_ip)
    except:
        pass

    # --- Try AWS
    try:
        external_ip = requests.get('https://checkip.amazonaws.com').text.strip()
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

def upnpc_create_port_map(local_port: int):
    r""" Creates a upnpc port map on your router from passed external_port to local port.

        Args: 
            local_port (int, `required`):
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

        local_ip = u.lanaddr
        external_ip = u.externalipaddress()

        logger.debug('UPNPC: your local ip address: ' + str(local_ip))
        logger.debug('UPNPC: your external ip address: ' + str(external_ip))
        logger.debug('UPNPC: status = ' + str(u.statusinfo()) + " connection type = " + str(u.connectiontype()))

        # find a free port for the redirection
        external_port = local_port
        rc = u.getspecificportmapping(external_port, 'TCP')
        while rc != None and external_port < 65536:
            external_port += 1
            rc = u.getspecificportmapping(external_port, 'TCP')
        if rc != None:
            raise UPNPCException("UPNPC: No available external ports for port mapping.")

        logger.info('UPNPC: trying to redirect remote: {}:{} => local: {}:{} over TCP', external_ip, external_port, local_ip, local_port)
        u.addportmapping(external_port, 'TCP', local_ip, local_port, 'Bittensor: %u' % external_port, '')
        logger.info('UPNPC: Create Success')

        return external_port

    except Exception as e:
        raise UPNPCException(e)