#!/usr/bin/env bash
set -o errexit

# Change to script's directory
cd "$(dirname "$0")"

# Load constants
source constant.sh

# Check script check_requirements
source check_requirements.sh

function print_help () {
  echo "Script for starting Bittensor instances."
  echo "Usage ./bittensor.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo " -h, --help           Print this help message and exit"
  echo " -t, --token          DigitalOcean token"
  echo " -l, --logging        Follow Docker container logs"
  echo " -m, --metagraph_port Metagraph port"
  echo " -a, --axon_port      Axon port"
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

token="None"
logging=False
metagraph_port="8121"
axon_port="8122"

# Read command line args
while test 4 -gt 0; do
  case "$1" in
    -h|--help)
      print_help
      exit 0
      ;;
    -t|--token)
      token=${2:-'None'}
      shift
      shift
      ;;
    -l|--logging)
      logging=${2:-False}
      shift
      shift
      ;;
    *)
      break
      ;;
    esac
done

function start_local_metagraph() {
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
        docker build --tag $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./Dockerfile .
    else
         # Build anyway
    docker build --tag $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./Dockerfile .
    fi

    # Trap control C (for clean docker container tear down.)
    function teardown() { 
        log "=== stop bittensor_container ==="
        docker stop bittensor-$identity

        exit 0
    }

    trap teardown INT SIGHUP SIGINT SIGTERM ERR

    metagraph_script="scripts/launch_metagraph.py --axon_port $axon_port --metagraph_port $metagraph_port"
    COMMAND="python3 $metagraph_script"
    log "Run command: $COMMAND"
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

function start_remote_metagraph() {
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
  log "Building $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
  docker build -t $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG -f ./Dockerfile .
 

  # Find external IP address of this droplet.
  droplet_ip_address=$(eval docker-machine ip bittensor-$identity)
  log "Droplet IP address: $droplet_ip_address:$port"

  metagraph_script="scripts/launch_metagraph.py --axon_port $axon_port --metagraph_port $metagraph_port"
  COMMAND="python3 $metagraph_script"
  log "Run command: $COMMAND"

  log "=== Run docker container on remote host. ==="
  log "=== container image: $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG ==="
  docker run --rm --name bittensor-$identity -d -t \
    -p $metagraph_port:$metagraph_port \
    -p $axon_port:$axon_port \
    $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG /bin/bash -c "$COMMAND"

  if [ $logging == 'true' ]; then
    log "=== follow logs ==="
    docker logs bittensor-$identity --follow
    trap teardown EXIT
  fi

}

function main() {
    log "██████╗░██╗████████╗████████╗███████╗███╗░░██╗░██████╗░█████╗░██████╗░"
    log "██╔══██╗██║╚══██╔══╝╚══██╔══╝██╔════╝████╗░██║██╔════╝██╔══██╗██╔══██╗"
    log "██████╦╝██║░░░██║░░░░░░██║░░░█████╗░░██╔██╗██║╚█████╗░██║░░██║██████╔╝"
    log "██╔══██╗██║░░░██║░░░░░░██║░░░██╔══╝░░██║╚████║░╚═══██╗██║░░██║██╔══██╗"
    log "██████╦╝██║░░░██║░░░░░░██║░░░███████╗██║░╚███║██████╔╝╚█████╔╝██║░░██║"
    log "╚═════╝░╚═╝░░░╚═╝░░░░░░╚═╝░░░╚══════╝╚═╝░░╚══╝╚═════╝░░╚════╝░╚═╝░░╚═╝"

    log "Digital Ocean Token: $token"
    log "Follow logging: $logging"
    log "Metagraph port: $metagraph_port"
    log "Axon port: $axon_port"

    if [ "$token" == "None" ]; then
        start_local_metagraph
    else
        start_remote_metagraph
    fi
}

main