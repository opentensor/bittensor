""" Manages a pool of grpc connections as receptors
"""
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

import math
from typing import Tuple, List, Union
from threading import Lock

import torch
from loguru import logger
import concurrent
import bittensor
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor

logger = logger.opt(colors=True)

class ReceptorPool ( torch.nn.Module ):
    """ Manages a pool of grpc connections as receptors
    """
    def __init__(
        self, 
        wallet: 'bittensor.Wallet',
        thread_pool: 'ThreadPoolExecutor',
        max_worker_threads: int,
        max_active_receptors: int,
        compression: str,
    ):
        super().__init__()
        self.wallet = wallet
        self.thread_pool = thread_pool
        self.max_worker_threads = max_worker_threads
        self.max_active_receptors = max_active_receptors
        self.receptors = {}
        self.cull_mutex = Lock()
        self.max_processes = 10
        self.compression = compression
        self.total_requests = 0


        
        try:
            self.external_ip = str(net.get_external_ip())
        except Exception:
            self.external_ip = None

    def __str__(self):
        return "ReceptorPool({},{})".format(len(self.receptors), self.max_active_receptors)

    def __repr__(self):
        return self.__str__()
    
    def __exit__(self):
        for receptor in self.receptors:
            receptor.__del__()

    def get_total_requests(self):
        return self.total_requests
    def get_receptors_state(self):
        r""" Return the state of each receptor.
            Returns:
                states (:obj:`List[grpc.channel.state]`)
                    The state of receptor.
        """
        return {hotkey: v.state() for hotkey, v in self.receptors.items()}



    def get_receptor_hotkeys(self):
        """
        Returns the receptor hotkeys.
        Returns:
            receptor_hotkeys: list of strings 

        """
        receptor_hotkeys = list(self.receptor.keys())
        return receptor_hotkeys

    def rm_receptor(self, key):
        '''
        Remove a receptor by the key.
        Args:
            key (str): The hotkey of the receptor to remove.
        Returns:
            key (str): The hotkey of the remove receptor
        

        '''

        self.receptors[ key ].close()
        del self.receptors[ key ]
        return key
    delete_receptor = del_receptor = rm_receptor

    def rm_all_receptors(self):
        for key in  deepcopy(list(self.receptors.keys())):
            self.rm_receptor(key=key)
    delete_all_receptors = del_all_receptors = remove_all_receptors = rm_all_receptors

    def forward(
            self, 
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.Synapse' ],
            inputs: Union[List [ torch.Tensor ], torch.Tensor],
            timeout: int,
            min_success = None, 
            return_success_only=False, 
            return_type = tuple,
            max_workers=None,
            graph=None,
            graph_features=['stake', 'ranks', 'trust', 'consensus', 'incentive', 'emission', 'dividends'],
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Forward tensor inputs to endpoints.
            Args:
                endpoints (:obj:`List[ bittensor.Endpoint ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from x are sent forward to these endpoints.
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    TODO(const): Allow multiple tensors.
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.
                timeout (int):
                    Request timeout.

                min_success (int):
                    Minimum Number of Successes before Returning the Function. This can be useful
                    if you want to call many endpoints but only need to receive the first N to come back successful.

                return_success_only (bool):
                    Return successful returns (code==1) only if True.
                return_type (str/type):
                    The return type of the tensor. The options are as follows
                        tuple: return a tuple of (tensors, codes, times, *graph_features)
                        dict: return a dictionary of {tensors: tensors, codes: codes, times: times, **graph_features_dictionary}
                    Note: If you include metagraph features this will be added in the order of the {graph_features} list
                max_workers (int),
                    Maximum number of workers. If None, defaults to the number of endpoints.
                graph (bittensor.graph):
                    if the graph is passed, the state of the graph will be added to the uids for additional info.
                
                graph_features (:obj:`List[torch.Tensor]`)
                    The list of additional graph features per uid you want to include in the response.
                    defaults: ['stake', 'ranks', 'trust', 'consensus', 'incentive', 'emission', 'dividends'],
            Returns:

                if return_type in [tuple]:
                    (forward_outputs (:obj:`List[ List[ torch.FloatTensor ]]` of shape :obj:`(num_endpoints * (num_synapses * (shape)))`, `required`):
                        Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.
                    forward_codes (:obj:`List[ List[bittensor.proto.ReturnCodes] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                        dendrite backward call return ops.
                    forward_times (:obj:`List[ List [float] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                        dendrite backward call times
                    **graph_features)
                elif return_type in [dict]:
                    dictionary of keys {outputs, codes, times, **graph_features}
        """


        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))

        receptors = [ self._get_or_create_receptor_for_endpoint( endpoint ) for endpoint in endpoints ]

        # Init argument iterables.
        call_args = []
        for idx, receptor in enumerate( receptors ):
            call_args.append({ 
                'receptor': receptor, 
                'inputs': inputs [ idx ] ,
                'synapses': synapses, 
                'timeout': timeout
            }) 

        # Init function.
        def call_forward( args ):
            return args['receptor'].forward( args['synapses'], args['inputs'], args['timeout'] )
        
        # Unpack responses
        results_dict = dict(
            outputs=[],
            codes=[],
            times=[],
            uids=[]
        )


        # resolve the minumum number of successes
        if min_success == None:
            min_success = len(endpoints)
        if min_success < 1:
            min_success = int(min_success*len(endpoints))
        elif min_success >= 1:
            min_success = int(min(min_success,len(endpoints)))
        elif min_success <= 0:
            raise Exception(' REQUIRED: 0<min_success<len(endpoints)')


        # ensure max_workers
        if not isinstance(max_workers, int):
            max_workers = len(endpoints)


        # Submit calls to receptors.
        with concurrent.futures.ThreadPoolExecutor( max_workers = max_workers ) as executor:
            

            # submit the calls asynchronously
            # map the futures to the call_arguments in case you want to the the inputs in the final result (not currently implemented)

            future_map = {}
            for idx, call_arg in enumerate(call_args):
                future = executor.submit( call_forward, call_arg)
                future_map[future] = call_arg

            success_response_cnt = 0 # success count 
            for i,future in enumerate(concurrent.futures.as_completed(future_map)):
                
                # get future_call_argds
                future_call_args = future_map.pop(future)

                # get uid (for metagraph retreival) make 1 uid per synapse (same uid)
                endpoint_uid = [future_call_args['receptor'].endpoint.uid ]*len(synapses)
                
                # get respone
                response = future.result()

                if response[1][0] == 1:
                    # this indicates a successful response 
                    success_response_cnt += 1
                    results_dict['outputs'].append( response[0] )
                    results_dict['codes'].append( response[1] )
                    results_dict['times'].append( response[2] )
                    results_dict['uids'].append(endpoint_uid)
                else:
                    # when return_success_only, ignore the responses that arent successful        
                    if not return_success_only:
                        results_dict['outputs'].append( response[0] )
                        results_dict['codes'].append( response[1] )
                        results_dict['times'].append( response[2] )
                        results_dict['uids'].append(endpoint_uid)

                future_call_args['receptor'].semaphore.release()  

                if graph != None:
                    graph_state_dict = graph.state_dict()
                    for k in graph_features:
                        results_dict[k] = [[v]*len(synapses) for v in graph_state_dict[k][[i[0] for i in results_dict['uids']]].tolist()]
        
                # when the success_response_cnt > min_success
                if success_response_cnt >= min_success:
                    break

        # cancel all the running futures and delete the future map
        for future, future_call_args in future_map.items():
            future.cancel()
            future_call_args['receptor'].semaphore.release()
        del future_map
 
        self._destroy_receptors_over_max_allowed()



        # resolve output type as either a tuple or a dictionary

        if return_type in ['tuple', tuple]:
            return_result =  [results_dict['outputs'], 
                    results_dict['codes'], 
                    results_dict['times'],
                    results_dict['uids']]

            # add graph features if they exists
            for k in graph_features:
                if k in results_dict:
                    return_result.append(results_dict[k])
            return tuple(return_result)
        elif return_type in ['dict', dict]:
            return results_dict



    def backward(
                self, 
                endpoints: List [ 'bittensor.Endpoint' ],
                synapses: List[ 'bittensor.Synapse' ],
                inputs: List [ torch.Tensor ],
                grads: List [ List[ torch.FloatTensor ] ],
                timeout: int
            ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Backward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List['bittensor.Endpoint']` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of x. Tensors from x are sent backward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    synapse.

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of list of grad tensors where each grad corresponds to a synapse call on an endpoint.
                
                timeout (int):
                    request timeout.

            Returns:
                backward_outputs (:obj:`List[ List[ torch.FloatTensor] ]` of shape :obj:`num_endpoints * (batch_size, sequence_len, -1)]`, `required`):
                    Gradients returned from the backward call one per endpoint.

                backward_codes (:obj:`List[ List[ bittensor.proto.ReturnCodes ] ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call return ops, one per endpoint and synapse.

                backward_times (:obj:`List[float]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call times one per endpoint and synapse.
        """
        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))
        if len(endpoints) != len(grads):
            raise ValueError('Endpoints must have the same length as passed grads_dy. Got {} and {}'.format(len(endpoints), len(grads)))
        for grads_per_synapse in grads:
            if len(grads_per_synapse) != len(synapses):
                raise ValueError('Gradients must have the same length as passed synapses. Got {} and {}'.format(len(grads_per_synapse), len(synapses)))

        # Init receptors.
        receptors = [ self._get_or_create_receptor_for_endpoint( endpoint ) for endpoint in endpoints ]

        # Init argument iterables.
        call_args = []
        for idx, receptor in enumerate( receptors ):
            call_args.append({ 
                'receptor': receptor, 
                'synapses': synapses, 
                'inputs': inputs [ idx ] ,
                'grads': grads [ idx ] ,
                'timeout': timeout
            }) 

        # Init function.
        def call_backward( args ):
            return args['receptor'].backward ( 
                synapses = args['synapses'], 
                inputs = args['inputs'], 
                grads = args['grads'], 
                timeout = args['timeout'] 
            )

        # Submit calls to receptors.
        with concurrent.futures.ThreadPoolExecutor( max_workers = len(endpoints) ) as executor:
            responses = executor.map ( call_backward, call_args, timeout=10*timeout )

        # Release semephore.
        for receptor in receptors:
            receptor.semaphore.release()
            
        # Unpack responses
        backward_outputs = []
        backward_codes = []
        backward_times = []
        for response in responses:
            backward_outputs.append( response[0] )
            backward_codes.append( response[1] )
            backward_times.append( response[2] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        # ---- Return ----
        return backward_outputs, backward_codes, backward_times

    def _destroy_receptors_over_max_allowed( self ):
        r""" Destroys receptors based on QPS until there are no more than max_active_receptors.
        """
        with self.cull_mutex:
            # ---- Finally: Kill receptors over max allowed ----
            while len(self.receptors) > self.max_active_receptors:
                min_receptor_qps = math.inf
                receptor_to_remove = None
                for next_receptor in self.receptors.values():
                    next_qps = next_receptor.stats.forward_qps.value
                    sema_value = next_receptor.semaphore._value
                    if (min_receptor_qps > next_qps) and (sema_value == self.max_processes):
                        receptor_to_remove = next_receptor
                        min_receptor_qps = next_receptor.stats.forward_qps.value
                        
                if receptor_to_remove != None:
                    try:
                        bittensor.logging.destroy_receptor_log(receptor_to_remove.endpoint)
                        self.receptors[ receptor_to_remove.endpoint.hotkey ].close()
                        del self.receptors[ receptor_to_remove.endpoint.hotkey ]
                    except KeyError:
                        pass
                elif receptor_to_remove == None:
                    break

    def _get_or_create_receptor_for_endpoint( self, endpoint: 'bittensor.Endpoint' ) -> 'bittensor.Receptor':
        r""" Finds or creates a receptor TCP connection associated with the passed Neuron Endpoint
            Returns
                receptor: (`bittensor.Receptor`):
                    receptor with tcp connection endpoint at endpoint.ip:endpoint.port
        """
        # ---- Find the active receptor for this endpoint ----
        if endpoint.hotkey in self.receptors:
            receptor = self.receptors[ endpoint.hotkey ]

            # Change receptor address.
            if receptor.endpoint.ip != endpoint.ip or receptor.endpoint.port != endpoint.port:
                #receptor.close()
                bittensor.logging.update_receptor_log( endpoint )
                receptor = bittensor.receptor (
                    endpoint = endpoint, 
                    wallet = self.wallet,
                    external_ip = self.external_ip,
                    max_processes = self.max_processes
                )            
                self.receptors[ receptor.endpoint.hotkey ] = receptor

        # ---- Or: Create a new receptor ----
        else:
            bittensor.logging.create_receptor_log( endpoint )
            receptor = bittensor.receptor (
                    endpoint = endpoint, 
                    wallet = self.wallet,
                    external_ip = self.external_ip,
                    max_processes = self.max_processes,
                    compression = self.compression
            )
            self.receptors[ receptor.endpoint.hotkey ] = receptor
            
        receptor.semaphore.acquire()
        return receptor