./scripts/build_protos.sh
docker build -t bittensor/bittensor:latest -f Dockerfile .
docker push bittensor/bittensor:subtensor_tests