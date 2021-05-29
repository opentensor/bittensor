import torch
import grpc
import bittensor

wallet =  bittensor.wallet (
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
) 
wallet.create_new_coldkey( use_password=False, overwrite = True)
wallet.create_new_hotkey( use_password=False, overwrite = True)

axon = bittensor.axon(wallet = wallet)

def test_forward_success():
    def forward( pubkey:str, inputs: torch.FloatTensor, modality:int ):
        return torch.zeros( [inputs.shape[0], inputs.shape[1], bittensor.__network_dim__])
    axon.attach_forward_function( forward )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_forward_not_implemented():
    axon.attach_forward_function( None )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.NotImplemented

def test_forward_empty_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[]
    )
    response, code, message = axon._forward( request )
    assert code ==  bittensor.proto.ReturnCode.EmptyRequest

def test_forward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ x ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException

def test_forward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized ]
    )
    response, code, message  = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_forward_deserialization():
    def forward( pubkey:str, inputs: torch.FloatTensor, modality:int ):
        return None
    axon.attach_forward_function( forward )
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
  
    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._forward( request )
    assert code == bittensor.proto.ReturnCode.ResponseDeserializationException

def test_backward_invalid_request():
    inputs_raw = torch.rand(3, 3, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)

    request = bittensor.proto.TensorMessage(
        version = bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[inputs_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.InvalidRequest


def test_backward_deserialization_error():
    x = dict()  # Not tensors that can be deserialized.
    g = dict()
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ x, g]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestDeserializationException


def test_backward_text_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TEXT, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_image_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.IMAGE, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_tensor_shape_error():
    inputs_raw = torch.rand(1, 1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message  = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException


def test_backward_grad_inputs_shape_error():
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(2, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.RequestShapeException

def test_backward_response_deserialization_error():
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ):
        return None
    axon.attach_backward_function( backward )
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.ResponseSerializationException

def test_backward_response_success():
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ):
        return torch.zeros( [1, 1, 1])
    axon.attach_backward_function( backward )
    inputs_raw = torch.rand(1, 1, 1)
    grads_raw = torch.rand(1, 1, bittensor.__network_dim__)
    serializer = bittensor.serializer( serialzer_type = bittensor.proto.Serializer.MSGPACK )
    inputs_serialized = serializer.serialize(inputs_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    grads_serialized = serializer.serialize(grads_raw, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH)
    request = bittensor.proto.TensorMessage(
        version=bittensor.__version__,
        public_key = axon.wallet.hotkey.public_key,
        tensors=[ inputs_serialized, grads_serialized]
    )
    response, code, message = axon._backward( request )
    assert code == bittensor.proto.ReturnCode.Success

def test_grpc_forward_works():
    axon = bittensor.axon(
        local_port = 8080,
        local_ip = '127.0.0.1'
    )
    def forward( pubkey:str, inputs_x:torch.FloatTensor, modality:int ):
        return torch.zeros( [1, 1, 1])
    axon.attach_forward_function( forward )
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
        version = bittensor.__version__,
        public_key = '1092310312914',
        tensors = [inputs_serialized]
    )
    response = stub.Forward(request)
    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()

def test_grpc_backward_works():
    axon = bittensor.axon(
        local_port = 8080,
        local_ip = '127.0.0.1'
    )
    def backward( pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ):
        return torch.zeros( [1, 1, 1])
    axon.attach_backward_function( backward )
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
        version = bittensor.__version__,
        public_key = '1092310312914',
        tensors = [inputs_serialized, grads_serialized]
    )
    response = stub.Backward(request)
    outputs = serializer.deserialize(response.tensors[0], to_type=bittensor.proto.TensorType.TORCH)
    assert outputs.tolist() == [[[0]]]
    axon.stop()


if __name__ == "__main__":
    test_forward_success()
    test_forward_not_implemented()
    test_forward_empty_request()
    test_forward_deserialization_error()
    test_forward_text_shape_error()
    test_forward_image_shape_error()
    test_forward_tensor_shape_error()
    test_forward_deserialization()
    test_backward_invalid_request()
    test_backward_deserialization_error()
    test_backward_text_shape_error()
    test_backward_image_shape_error()
    test_backward_tensor_shape_error()
    test_backward_grad_inputs_shape_error()
    test_backward_response_deserialization_error()
    test_backward_response_success()
    test_grpc_forward_works()
    test_grpc_backward_works()
    