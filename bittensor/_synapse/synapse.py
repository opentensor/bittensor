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

import os
import time
import bittensor
import argparse

class Synapse( bittensor.grpc.BittensorServicer ):

    def __init__( 
            self, 
            config: 'bittensor.Config' =  None, 
            metagraph: 'bittensor.metagraph.Metagraph' = None
        ):
        """ Initializes a new Synapse.
            Args:
                config (:obj:`bittensor.Config`, `optional`, defaults to bittensor.config()):
                    bittensor config object.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`, defaults to bittensor.metagraph.Metagraph()):
                    bittensor metagraph object.
        """
        if config == None: config = bittensor.config()
        Synapse.check_config( config )
        self.priority_threadpool = bittensor.prioritythreadpool()
        self.metagraph = metagraph

    @classmethod
    def config(cls) -> 'bittensor.Config':
        """ Returns the config for this synapse."""
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'synapse.blacklist.stake', type=float, help='The amount of stake (tao) required to make a call.', default=10)
            parser.add_argument('--' + prefix_str + 'synapse.blacklist.allow_non_registered', action='store_true', help='''If true, allow non-registered peers''', default=True)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod   
    def help(cls):
        """ Print help to stdout """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()

    @classmethod   
    def add_defaults(cls, defaults):
        """ Add default values to defaults object"""
        defaults.synapse = bittensor.Config()
        defaults.synapse.blacklist.stake = os.getenv('BT_SYNAPSE_BLACKLIST_STAKE') if os.getenv('BT_SYNAPSE_BLACKLIST_STAKE') != None else 10
        defaults.synapse.blacklist.allow_non_registered = os.getenv('BT_SYNAPSE_BLACKLIST_ALLOW_NON_REGISTERED') if os.getenv('BT_SYNAPSE_BLACKLIST_ALLOW_NON_REGISTERED') != None else True

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def __str__(self):
        return "synapse"
    
    def _attach( self, axon: 'bittensor.axon.Axon' ):
        """ _attach: Attaches the synapse to the axon."""
        bittensor.grpc.add_BittensorServicer_to_server( self, axon.server )

    # Instance priority called by subclass priority which is called by super priority.
    def priority( self, forward_call: bittensor.BittensorCall ) -> float:
        raise NotImplementedError('Must implement priority() in subclass.')

    def _priority( self, forward_call: bittensor.BittensorCall ) -> float:
        return self.priority()
    
    def __priority( self, forward_call: bittensor.BittensorCall ) -> bool:
        """ __priority: Returns the priority of the forward call.
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    forward_call to check.
            Returns:
                float: priority of the forward call.
        """
        # Call subclass priority, if not implemented use the 
        # metagraph priority based on stake.
        try:
            return float( self._priority( forward_call ) )
        except:
            if self.metagraph != None:
                uid = self.metagraph.hotkeys.index( forward_call.hotkey )
                return float( self.metagraph.S[uid].item() )
            else:
                return 0.0 

    # Instance blacklist called by subclass blacklist which is called by super blacklist.
    def blacklist( self, forward_call: bittensor.BittensorCall ) -> bool:
        raise NotImplementedError('Must implement subclass_blacklist() in subclass.')

    def _blacklist( self, forward_call: bittensor.BittensorCall ) -> bool:
        return self._blacklist( forward_call )
    
    def __blacklist( self, forward_call: bittensor.BittensorCall ) -> bool:
        """ ___blacklist: Checks if the forward call is blacklisted.
            Args:
                forward_call (:obj:`bittensor.BittensorCall`, `required`):
                    forward_call to check.
            Returns:
                bool: True if blacklisted, False otherwise.
        """
        # Call subclass blacklist and optionaly return if metagraph is None.
        try:
            return self._blacklist( forward_call )
        except:
            pass
        if self.metagraph == None: return False

        # Check for registration
        def registration_check():
            is_registered = forward_call.hotkey in self.metagraph.hotkeys
            if not is_registered:
                if self.config.synapse.blacklist.allow_non_registered:
                    return False
                raise Exception('Registration blacklist')

        # Blacklist based on stake.
        def stake_check() -> bool:
            uid = self.metagraph.hotkeys.index( forward_call.hotkey )
            if self.metagraph.S[uid].item() < self.config.synapse.blacklist.stake:
                raise Exception('Stake blacklist')
            return False

        # Optionally blacklist based on checks.
        try:
            registration_check()
            stake_check()            
            return False
        except Exception as e:
            return True

    def forward(self, forward_call: bittensor.BittensorCall ) -> bittensor.BittensorCall:
        raise NotImplementedError('Must implement forward() in subclass.')
    
    def pre_process_request_proto_to_forward_call( 
            self, 
            request_proto: 'bittensor.ForwardRequest' 
        ) -> 'bittensor.BittensorCall':
        """ pre_process_request_proto_to_forward_call
            ------------------------------------------
            Args:
                request_proto (bittensor.ForwardRequest):
                    request_proto to process in to a forward call.
            Returns:
                bittensor.BittensorCall (:obj:`bittensor.BittensorCall`, `required`):
                    forward call processed from the request proto.
            """
        raise NotImplementedError('Must implement pre_process_request_proto_to_forward_call() in subclass.')
    
    def post_process_forward_call_to_response_proto( 
            self, 
            forward_call: 'bittensor.BittensorCall' 
        ) -> 'bittensor.ForwardResponse':
        """ post_process_forward_call_to_response_proto
            --------------------------------------------
            Args:
                forward_call (bittensor.BittensorCall):
                    forward_call to process in to a response proto.
            Returns:    
                response (bittensor.ForwardResponse):
                    response proto processed from the forward call.
        """
        raise NotImplementedError('Must implement post_process_forward_call_to_response_proto() in subclass.')
        
    def _Forward( self, request_proto: 'bittensor.ForwardRequest' ) -> 'bittensor.BittensorCall':
        forward_call = self.pre_process_request_proto_to_forward_call( request_proto = request_proto )
        try:
            # Check blacklist.
            if self.__blacklist( forward_call ): raise Exception('Blacklisted')
            # Get priority.
            priority = self.__priority( forward_call )
            # Queue the forward call.
            future = self.priority_threadpool.submit(
                self.forward,
                forward_call = forward_call,
                priority = priority,
            )
        except Exception as e:
            forward_call.request_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.request_message = str(e)
        finally:
            # Log request.
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = True, 
                is_response = False, 
                code = forward_call.request_code, 
                call_time = time.time() - forward_call.start_time, 
                pubkey = forward_call.hotkey, 
                uid = None, 
                inputs = forward_call.get_inputs_shape() if forward_call.request_code == bittensor.proto.ReturnCode.Success else None,
                outputs = None,
                message = forward_call.request_message,
                synapse = self.__str__()
            )
            if forward_call.request_code != bittensor.proto.ReturnCode.Success:
                return self.post_process_forward_call_to_response_proto( forward_call )

        # Do forward.
        try:
            # Get the result.
            future.result( timeout = forward_call.timeout )

        except Exception as e:
            forward_call.response_code = bittensor.proto.ReturnCode.UnknownException
            forward_call.response_message = str(e)
        finally:
            # Log response
            bittensor.logging.rpc_log ( 
                axon = True, 
                forward = True, 
                is_response = True, 
                code = forward_call.response_code, 
                call_time = time.time() - forward_call.start_time, 
                pubkey = forward_call.hotkey, 
                uid = None, 
                inputs = list( forward_call.get_inputs_shape() ) if forward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                outputs = list( forward_call.get_outputs_shape() ) if forward_call.response_code == bittensor.proto.ReturnCode.Success else None,
                message = forward_call.response_message,
                synapse = self.__str__()
            )
            return self.post_process_forward_call_to_response_proto( forward_call )
