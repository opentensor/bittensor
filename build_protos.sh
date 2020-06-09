# Build protos.
	python3 -m grpc.tools.protoc bittensor.proto  -I. --python_out=. --grpc_python_out=.
