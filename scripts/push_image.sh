./scripts/build_protos.sh
docker build -t unconst/bittensor:latest -f Dockerfile.base .
docker push unconst/bittensor:latest