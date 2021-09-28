import torch
import grpc
import bittensor
from datetime import datetime
import logging

wallet =  bittensor.wallet (
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
def sign(wallet):
    nounce = datetime.now().strftime(format= '%m%d%Y%H%M%S%f')
    message  = nounce+str(wallet.hotkey.ss58_address) 
    spliter = 'bitxx'
    signature = spliter.join([nounce,str(wallet.hotkey.ss58_address),wallet.hotkey.sign(message)])
    return signature


wallet.create_new_coldkey( use_password=False, overwrite = True)
wallet.create_new_hotkey( use_password=False, overwrite = True)

axon = bittensor.axon(wallet = wallet)

def test_forward_not_implemented():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_tensor_success():
    def forward( pubkey:str, inputs_x: torch.FloatTensor):
        return torch.zeros( [inputs_x.shape[0], inputs_x.shape[1], bittensor.__network_dim__])
    axon.attach_forward_callback( forward, modality=2)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_empty_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[]
    )
    response, code, call_time, message = axon._forward( request )
    assert code ==  bittensor.proto.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ x ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_forward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, call_time, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_deserialization():
    def forward( pubkey:str, inputs_x: torch.Tensor):
        return None
    axon.attach_forward_callback( forward, modality = bittensor.proto.Modality.TENSOR)
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.EmptyResponse

def test_backward_invalid_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.InvalidRequest


def test_backward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    g = dict()
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ x, g]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException


def test_backward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_grad_inputs_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(2, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_response_deserialization_error():
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ):
        return None
    axon.attach_backward_callback( backward,modality=bittensor.proto.Modality.TENSOR)
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.EmptyResponse

def test_backward_response_success():
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon.attach_backward_callback( backward,modality = bittensor.proto.Modality.TENSOR )
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version_as_int__,
        hotkey = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, call_time, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_grpc_forward_works():
    def forward( pubkey:str, inputs_x:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon = bittensor.axon (
        port = 8080,
        ip = '127.0.0.1',
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:8080',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized]
    )
    response = stub.Forward(request,
                            metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',sign(axon.wallet)),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ))

    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()

def test_grpc_backward_works():
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])

    axon = bittensor.axon (
        port = 8080,
        ip = '127.0.0.1',
    )
    axon.attach_backward_callback( backward , modality = bittensor.proto.Modality.TENSOR)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:8080',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized, grads_serialized]
    )
    response = stub.Backward(request,
                             metadata = (
                                    ('rpc-auth-header','Bittensor'),
                                    ('bittensor-signature',sign(axon.wallet)),
                                    ('bittensor-version',str(bittensor.__version_as_int__)),
                                    ))
    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()

def test_grpc_forward_fails():
    def forward( pubkey:str, inputs_x:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])
    axon = bittensor.axon (
        port = 8080,
        ip = '127.0.0.1',
    )
    axon.attach_forward_callback( forward,  modality = bittensor.proto.Modality.TENSOR )
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:8080',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized]
    )
    try:
        response = stub.Forward(request)
    except grpc.RpcError as rpc_error_call:
        grpc_code = rpc_error_call.code()
        assert grpc_code == grpc.StatusCode.UNAUTHENTICATED

    axon.stop()

def test_grpc_backward_fails():
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor):
        return torch.zeros( [1, 1, 1])

    axon = bittensor.axon (
        port = 8080,
        ip = '127.0.0.1',
    )
    axon.attach_backward_callback( backward , modality = bittensor.proto.Modality.TENSOR)
    axon.start()

    channel = grpc.insecure_channel(
            '127.0.0.1:8080',
            options=[('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1)])
    stub = bittensor.grpc.BittensorStub( channel )

    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    grads_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version_as_int__,
        hotkey = '1092310312914',
        tensors = [inputs_serialized, grads_serialized]
    )
    
    try:
        response = stub.Backward(request)
    except grpc.RpcError as rpc_error_call:
        grpc_code = rpc_error_call.code()
        assert grpc_code == grpc.StatusCode.UNAUTHENTICATED

    axon.stop()

def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        val = s.connect_ex(('localhost', port))
        if val == 0:
            return True
        else:
            return False

def test_axon_is_destroyed():
    logging.basicConfig()
    port = 8081
    assert is_port_in_use( port ) == False
    axon = bittensor.axon ( port = port )
    assert is_port_in_use( port ) == True
    axon.start()
    assert is_port_in_use( port ) == True
    axon.stop()
    assert is_port_in_use( port ) == False
    axon.__del__()
    assert is_port_in_use( port ) == False

    port = 8082
    assert is_port_in_use( port ) == False
    axon2 = bittensor.axon ( port = port )
    assert is_port_in_use( port ) == True
    axon2.start()
    assert is_port_in_use( port ) == True
    axon2.__del__()
    assert is_port_in_use( port ) == False

    port_3 = 8086
    assert is_port_in_use( port_3 ) == False
    axonA = bittensor.axon ( port = port_3 )
    assert is_port_in_use( port_3 ) == True
    axonB = bittensor.axon ( port = port_3 )
    assert axonA.server != axonB.server
    assert is_port_in_use( port_3 ) == True
    axonA.start()
    assert is_port_in_use( port_3 ) == True
    axonB.start()
    assert is_port_in_use( port_3 ) == True
    axonA.__del__()
    assert is_port_in_use( port ) == False
    axonB.__del__()
    assert is_port_in_use( port ) == False


if __name__ == "__main__":
    test_axon_is_destroyed()