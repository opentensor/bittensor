./scripts/build_protos.sh
docker build -t bittensor/bittensor:latest -f Dockerfile.base .
docker push bittensor/bittensor:latest