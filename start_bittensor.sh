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
  echo " -ax, --axon_port     The bootstrapping peer's axon port."
  echo " -m, --metagraph_port Metagraph bind port."
  echo " -s, --metagraph_size Metagraph cache size."
  echo " -b, --bootstrap      Metagraph boot peer."
  echo " -k, --neuron_key     Neuron Public Key."
  echo " -r, --remote_ip      Remote IP of container."
  echo " -mp, --model_path    Path to a saved version of the model to resume training."
}

identity=$(LC_CTYPE=C tr -dc 'a-z' < /dev/urandom | head -c 7 | xargs)
# Bind the grpc server to this address with port
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
# DO token
token='none'
# follow logs?
logging='false'


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
      shift
      ;;
    -ax|--axon_port)
      axon_port=${2:-$default_axon_port}
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
      bootstrap=${2:-'none'}
      bootstrap_port=${bootstrap##*:}
      shift
      shift
      ;;
    -k|--neuron_key)
      neuron_key=${2:-'none'}
      shift
      shift
      ;;
    -r|--remote_ip)
      remote_ip=${2:-'host.docker.internal'}
      shift
      shift
      ;;
    -mp|--model_path)
      model_path=${2:-'none'}
      shift
      shift
      ;;
    -t|--token)
      token=${2:-'none'}
      shift
      shift
      ;;
    -l|--logging)
      logging=${2:-'false'}
      shift
      shift
      ;;
    -i|--identity)
      identity=${2:-$identity}
      shift
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

    # Stop the container if it is already running.
    if [[ "$(docker ps -a | grep bittensor-$identity)" ]]; then
        log "=== stopping bittensor-$identity ==="
        docker stop bittensor-$identity || true
        docker rm bittensor-$identity || true
    fi

    # Init image if non-existent
    log "=== Building bittensor image ==="

    if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
        log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
        docker build --tag $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./examples/$neuron/Dockerfile .
    else
         # Build anyway
    docker build --tag $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./examples/$neuron/Dockerfile .
    fi

    # Trap control C (for clean docker container tear down.)
    function teardown() { 
        log "=== stop bittensor_container ==="
        docker stop bittensor-$identity

        exit 0
    }

    trap teardown INT SIGHUP SIGINT SIGTERM ERR

    # Build start command
    bittensor_script="./scripts/bittensor.sh"

    COMMAND="$bittensor_script $identity $serve_address $port $logdir $neuron $chain_endpoint $axon_port $metagraph_port $metagraph_size $bootstrap $neuron_key $remote_ip $model_path"
    log "Run command: $COMMAND"

    # Run docker service
    dest_port=$metagraph_port
    if [ $dest_port == 'none' ]; then
      dest_port=$bootstrap_port
    fi

    log "=== run docker container locally ==="
    log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
    
    docker run --rm --name bittensor-$identity -d -t \
      --network=host \
      $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

    if [ $logging == 'true' ]; then
      log "=== follow logs ==="
      docker logs bittensor-$identity --follow
      trap teardown EXIT
    fi
}


function start_remote_service() {
  log "=== run instance remotely ==="

  # Build trap "ctrl+c" (for clean docker container teardown from DO)
  function teardown() {
    log "=== tearing down remote instance ==="
    eval $(docker-machine env -u)
    echo "To tear down host, run: "
    echo "       docker-machine stop bittensor-$identity & docker-machine rm bittensor-$identity --force"
    exit 0
  }

  trap teardown INT SIGHUP SIGTERM ERR EXIT

  # initialize host
  log "=== initializing remote host ==="
  if [[ "$(docker-machine ls | grep bittensor-$identity)" ]]; then
    # Host already exists
    log "bittensor-$identity droplet already exists"
  else
    log "Creating droplet: bittensor-$identity"
    droplet_create_cmd="docker-machine create --driver digitalocean --digitalocean-size s-1vcpu-1gb --digitalocean-access-token ${token} bittensor-$identity"
    log "Create command: $droplet_create_cmd"
    eval $droplet_create_cmd
  fi

  # Set docker context to droplet
  log "=== switching droplet context ==="
  eval $(docker-machine env bittensor-$identity)

  # Stop container if it is already running
  if [[ "$(docker ps -a | grep bittensor-$identity)" ]]; then
    log "=== stopping bittensor-$identity ==="
    docker stop bittensor-$identity || true
    docker rm bittensor-$identity || true
  fi

  # Build image
  if [[ "$(docker images -q $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
    log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./examples/$neuron/Dockerfile .
  else
    # Build anyway
    docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./examples/$neuron/Dockerfile .
  fi

  # Find external IP address of this droplet.
  droplet_ip_address=$(eval docker-machine ip bittensor-$identity)
  log "Droplet IP address: $droplet_ip_address:$port"

  # Build start command
  bittensor_script="./scripts/bittensor.sh"
  remote_ip=$droplet_ip_address

  COMMAND="$bittensor_script $identity $serve_address $port $logdir $neuron $chain_endpoint $axon_port $metagraph_port $metagraph_size $bootstrap $neuron_key $remote_ip $model_path"
  log "Run command: $COMMAND"
  
  # Run docker service
  dest_port=$metagraph_port
  if [ $dest_port == 'none' ]; then
    dest_port=$bootstrap_port
  fi

  # Run docker service.
  log "=== Run docker container on remote host. ==="
  log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
  docker run --rm --name bittensor-$identity -d -t \
    -p $dest_port:$dest_port \
    -p $axon_port:$axon_port \
    $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"
  
  if [ $logging == 'true' ]; then
    log "=== follow logs ==="
    docker logs bittensor-$identity --follow
    trap teardown EXIT
  fi

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
log "metagraph_port: $metagraph_port"
log "logdir: $logdir"
log "neuron: $neuron"
log "bootstrap: $bootstrap_port"
log "Axon port: $axon_port"
log "Follow logging: $logging"

if [ "$token" == "none" ]; then
  start_local_service
else
  start_remote_service
fi

}

main

