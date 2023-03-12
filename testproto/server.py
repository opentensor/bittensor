
import grpc
import asyncio
import messages_pb2
import messages_pb2_grpc
from concurrent import futures
thread_pool = futures.ThreadPoolExecutor( max_workers = 2 )
server = grpc.server( 
    thread_pool,
    maximum_concurrent_rpcs = 2,
    options = [('grpc.keepalive_time_ms', 100000),
                ('grpc.keepalive_timeout_ms', 500000)]
)
server.add_insecure_port( 'localhost:9091' )

class Serivcer( messages_pb2_grpc.ServerThingServicer ):

    def Forward( self, request, context ):
        print (request)
        return messages_pb2.FMessage( thing = request.thing + 1 )
    
servicer = Serivcer()
messages_pb2_grpc.add_ServerThingServicer_to_server( servicer, server )
server.start()


channel = grpc.aio.insecure_channel( 'localhost:9091' )
stub = messages_pb2_grpc.ServerThingStub( channel )

request = messages_pb2.FMessage( thing = 1 )

asyncio_future = stub.Forward( request = request )
loop = asyncio.get_event_loop()
response = loop.run_until_complete( asyncio_future) 
print (response)
    
    # Wait for response.
    # return await asyncio.wait_for( asyncio_future, timeout = timeout )
#response = stub.Forward( request )
#print (response)
