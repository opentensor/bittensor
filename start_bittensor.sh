#!/usr/bin/env bash
set -o errexit

# Change to script's directory
cd "$(dirname "$0")"

# Load constants
source ./scripts/constant.sh

# Check script check_requirements
source scripts/check_requirements.sh

function print_help () {
  echo "Script for starting Bittensor instances."
  echo "Usage ./bittensor.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help       Print this help message and exit"
  echo " -n, --neuron     bittensor neuron name e.g. boltzmann"
  echo " -l, --logdir     Logging directory."
  echo " -p, --port       Bind side port for accepting requests."
  echo " -r, --remote     Run instance locally."
  echo " -t, --token      Digital ocean API token."
}


identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
# Bind the grpc server to this address with port
bind_address="0.0.0.0"
# Advertise this address on the EOS chain.
machine=$(whichmachine)
echo "Detected host: $machine"
if [[ "$machine" == "Darwin" ||  "$machine" == "Mac" ]]; then
    serve_address="host.docker.internal"
else
    serve_address="172.17.0.1"
fi

# hardcoded port for now
# TODO(shibshib) should bind and advertise port instead.
port=8120

# Directory for sinking logs and model updates.
# TODO(const) Should be root dir.
logdir="data/$identity/logs"
# Is this service running on digital ocean.
remote="false"
# Digital ocean API token for creating remote instances.
token="none"
# Neuron: The client adhering to the Bittensor protocol.
neuron="mnist"

# Read command line args
while test 9 -gt 0; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -p|--port)
      port=`echo $2`
      tbport=$((port+1))
      shift
      shift
      ;;
    -l|--logdir)
      logdir=`echo $2`
      shift
      shift
      ;;
    -r|--remote)
      remote="false"
      shift
      ;;
    -t|--token)
      token=`echo $2`
      shift
      shift
      ;;
    -n|--neuron)
      neuron=`echo $2`
      shift
      shift
      ;;
    *)
      break
      ;;
  esac
done

function start_local_service() {
    log "=== Running Locally ==="

    # Init image if non-existent
    log "=== Building bittensor image ==="

    if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
        log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
        docker build --tag $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./examples/$neuron/Dockerfile .
    else
         # Build anyway
    docker build --tag $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./examples/$neuron/Dockerfile .
    fi

    # Stop the container if it is already running.
    if [[ "$(docker ps -a | grep bittensor-$identity)" ]]; then
        log "=== stopping bittensor-$identity ==="
        docker stop bittensor-$identity || true
        docker rm bittensor-$identity || true
    fi

    # Trap control C (for clean docker container tear down.)
    function teardown() { 
        log "=== stop bittensor_container ==="
        docker stop bittensor-$identity

        exit 0
    }

    trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

    # Build start command
    script="./scripts/bittensor.sh"
    COMMAND="$script $identity $serve_address $port $logdir $neuron"
    log "Run command: $COMMAND"

    # Run docker service
    log "=== run docker container locally ==="
    log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
    docker run --rm --name bittensor-$identity -d -t \
    -p $port:$port \
    --mount type=bind,source="$(pwd)"/scripts,target=/bittensor/scripts \
    --mount type=bind,source="$(pwd)"/data,target=/bittensor/data \
    --mount type=bind,source="$(pwd)"/examples,target=/bittensor/examples \
    $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

    log "=== follow logs ==="
    docker logs bittensor-$identity --follow
}

# Main function
function main() {

log "██████╗░██╗████████╗████████╗███████╗███╗░░██╗░██████╗░█████╗░██████╗░"
log "██╔══██╗██║╚══██╔══╝╚══██╔══╝██╔════╝████╗░██║██╔════╝██╔══██╗██╔══██╗"
log "██████╦╝██║░░░██║░░░░░░██║░░░█████╗░░██╔██╗██║╚█████╗░██║░░██║██████╔╝"
log "██╔══██╗██║░░░██║░░░░░░██║░░░██╔══╝░░██║╚████║░╚═══██╗██║░░██║██╔══██╗"
log "██████╦╝██║░░░██║░░░░░░██║░░░███████╗██║░╚███║██████╔╝╚█████╔╝██║░░██║"
log "╚═════╝░╚═╝░░░╚═╝░░░░░░╚═╝░░░╚══════╝╚═╝░░╚══╝╚═════╝░░╚════╝░╚═╝░░╚═╝"

log "identity: $identity"
log "remote: $remote"
log "port: $port"
log "server address: $serve_address"
log "logdir: $logdir"
log "neuron: $neuron"

start_local_service
}

main

