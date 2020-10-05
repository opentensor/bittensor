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
  echo " -h, --help           Print this help message and exit"
  echo " -n, --neuron         Bittensor neuron name e.g. boltzmann"
  echo " -l, --logdir         Logging directory."
  echo " -p, --port           Bind side port for accepting requests."
  echo " -c, --chain_endpoint Bittensor chain endpoint."
  echo " -a, --axon_port      Axon terminal bind port."
  echo " -m, --metagraph_port Metagraph bind port."
  echo " -s, --metagraph_size Metagraph cache size."
  echo " -b, --bootstrap      Metagraph boot peer."
  echo " -k, --neuron_key     Neuron Public Key."
  echo " -r, --remote_ip      Remote IP of container."
  echo " -mp, --model_path    Path to a saved version of the model to resume training."
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

# Neuron: The client adhering to the Bittensor protocol.

# Defaults

# Directory for sinking logs and model updates.# TODO(const) Should be root dir.
logdir="data/$identity/logs"
# Default neuron
neuron='mnist'
# Chain endpoint
chain_endpoint='none'
# Ports to bind to this container
port=$(( ( RANDOM % 60000 ) + 5000 ))
# axon port
axon_port=$(( ( RANDOM % 60000 ) + 5000 ))
default_bootstrap_port=$(( ( RANDOM % 60000 ) + 5000 ))
default_axon_port=$(( ( RANDOM % 60000 ) + 5000 ))
default_metagraph_port=$(( ( RANDOM % 60000 ) + 5000 ))
# metagraph port
metagraph_port='none'
# bootstrap port
bootstrap_port='none'
# Metagraph size
metagraph_size='0'
# Is this service running on digital ocean. Default is local docker host.
remote_ip="host.docker.internal"
# Bootstrap port
bootstrap_port=$default_bootstrap_port
bootstrap='none'
# Neuron key
neuron_key='none'
# Model path
model_path='none'


# Read command line args
while test 12 -gt 0; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -n|--neuron)
      neuron=${2:-'mnist'}
      shift
      shift
      ;;
    -l|--logdir)
      logdir=${2:-$logdir}
      shift
      shift
      ;;
    -p|--port)
      port=${2:-$default_port}
      shift
      shift
      ;;
    -c|--chain_endpoint)
      chain_endpoint=${2:-'none'}
      shift
      ;;
    -a|--axon_port)
      axon_port=`${2:-default_axon_port}`
      shift
      shift
      ;;
    -m|--metagraph_port)
      metagraph_port=${2:-$default_metagraph_port}
      shift
      shift
      ;;
    -s|--metagraph_size)
      metagraph_size=${2:-'100000'}
      shift
      shift
      ;;
    -b|--bootstrap)
      bootstrap_port=${2:-$default_bootstrap_port}
      bootstrap="${serve_address}:${bootstrap_port}"
      shift
      shift
      ;;
    -k|--neuron_key)
      neuron_key=${2:-'none'}
      shift
      shift
      ;;
    -r|--remote_ip)
      remote=${2:-'host.docker.internal'}
      shift
      ;;
    -mp|--model_path)
      model_path=${2:-'none'}
      shift
      ;;
    *)
      break
      ;;
  esac
done

# Look into axon port definition
# Extend to open up axon port to allow comms here.

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
    COMMAND="$script $identity $serve_address $port $logdir $neuron $chain_endpoint $axon_port $metagraph_port $metagraph_size $bootstrap $neuron_key $remote_ip $model_path"
    log "Run command: $COMMAND"

    # Run docker service
    dest_port=$metagraph_port
    if [ $dest_port == 'none' ]; then
      dest_port=$bootstrap_port
    fi

    log "=== run docker container locally ==="
    log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
    if [ $remote_ip == 'host.docker.internal' ]; then
      docker run --rm --name bittensor-$identity -d -t \
        --network=host \
        --mount type=bind,source="$(pwd)"/scripts,target=/bittensor/scripts \
        --mount type=bind,source="$(pwd)"/data,target=/bittensor/data \
        --mount type=bind,source="$(pwd)"/examples,target=/bittensor/examples \
        $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
    else
      docker run --rm --name bittensor-$identity -d -t \
        -p $port:$dest_port \
        -p $axon_port:$axon_port \
        --network=host \
        --mount type=bind,source="$(pwd)"/scripts,target=/bittensor/scripts \
        --mount type=bind,source="$(pwd)"/data,target=/bittensor/data \
        --mount type=bind,source="$(pwd)"/examples,target=/bittensor/examples \
        $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
    fi
    

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
log "remote_ip: $remote_ip"
log "port: $port"
log "server address: $serve_address"
log "logdir: $logdir"
log "neuron: $neuron"
log "bootstrap port: $bootstrap_port"

start_local_service
}

main

