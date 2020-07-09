# A Tool for punching a hole in UPNPC enabled routers.

import argparse
import miniupnpc
import random
from loguru import logger


class Nat:
    @staticmethod
    def delete_port_map(port: int):
        """ Deletes the port map on the desired port """
        try:
            logger.info('Using UPnP for port mapping...')
            u = miniupnpc.UPnP()
            u.discoverdelay = 200
            logger.info('Discovering... delay=%ums' % u.discoverdelay)
            ndevices = u.discover()
            logger.info(str(ndevices) + ' device(s) detected')
            u.selectigd()
            logger.info('Deleting mapped port=%u' % port)
            b = u.deleteportmapping(port, 'TCP')
        except Exception as e:
            logger.error('Exception in UPnP :', e)
            pass

    @staticmethod
    def create_port_map(port: int = random.randint(7000, 60000)):
        try:
            u = miniupnpc.UPnP()
            u.discoverdelay = 200
            logger.info('Using UPnP for port mapping...')
            logger.info('Discovering... delay=%ums' % u.discoverdelay)
            ndevices = u.discover()
            logger.info(str(ndevices) + ' device(s) detected')

            u.selectigd()
            local_ip = u.lanaddr
            external_ip = u.externalipaddress()
            local_port = int(port)
            external_port = local_port

            logger.info('local ip address :' + str(local_ip))
            logger.info('external ip address :' + str(external_ip))
            logger.info(str(u.statusinfo()) + " " + str(u.connectiontype()))

            # find a free port for the redirection
            rc = u.getspecificportmapping(external_port, 'TCP')
            while rc != None and external_port < 65536:
                external_port += 1
                rc = u.getspecificportmapping(external_port, 'TCP')
            if rc != None:
                logger.error('Exception in UPnP : ' + str(rc))

            logger.info('trying to redirect %s port %u TCP => %s port %u TCP' %
                        (external_ip, external_port, local_ip, local_port))
            b = u.addportmapping(external_port, 'TCP', local_ip, local_port,
                                 'UPnP IGD Tester port %u' % external_port, '')

        except Exception as e:
            logger.error('Exception in UPnP :', e)
            return external_ip, -1

        print('success' + ':' + str(external_ip) + ':' + str(external_port))
        return external_ip, external_port
