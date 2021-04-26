./scripts/build_protos.sh
docker build -t bittensor/bittensor:subtensor_tests -f Dockerfile .
docker push bittensor/bittensor:subtensor_tests