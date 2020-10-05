#!/usr/bin/env bash
source ./scripts/constant.sh

# script arguments

# Network identity
IDENTITY=$1
# Address to post to the EOS chain. (our API endpoint)
SERVE_ADDRESS=$2
# Port to bind endpoint on.
PORT=$3
# Directory to save checkpoints and logs.
LOGDIR=$4
# Python client to run.
NEURON=$5
# Chain Endpoint
CHAIN_ENDPOINT=$6
# Axon port
AXON_PORT=$7
# Metagraph Port
METAGRAPH_PORT=$8
# Metagraph size
METAGRAPH_SIZE=$9
# Bootstrap peer
BOOTSTRAP_PEER=${10}
# Neuron key
NEURON_KEY=${11}
# Remote or local run
REMOTE_IP=${12}
# MODEL PATH
MODEL_PATH=${13}

function start_neuron() {
    log ""
    log "=== start Neuron ==="

    #COMMAND="python3 ./examples/$NEURON/main.py --metagraph_port=$METAGRAPH_PORT --axon_port=$AXON_PORT --remote_ip=$REMOTE_IP"

    #if [ $METAGRAPH_PORT == 'none' ]; then
    #  COMMAND="python3 ./examples/$NEURON/main.py --bootstrap='$SERVE_ADDRESS:$BOOTSTRAP_PORT' --axon_port=$AXON_PORT --remote_ip=$REMOTE_IP"
    #fi

    COMMAND="python3 ./examples/$NEURON/main.py"

    if [ $CHAIN_ENDPOINT != 'none' ]; then
      COMMAND="${COMMAND} --chain_endpoint=${CHAIN_ENDPOINT}"
    fi

    if [ $AXON_PORT != 'none' ]; then
      COMMAND="${COMMAND} --axon_port=${AXON_PORT}"
    fi

    if [ $METAGRAPH_PORT != 'none' ]; then
      COMMAND="${COMMAND} --metagraph_port=$METAGRAPH_PORT"
    fi

    if [ $METAGRAPH_SIZE != '0' ]; then
      COMMAND="${COMMAND} --metagraph_size=${METAGRAPH_SIZE}"
    fi

    if [ $BOOTSTRAP_PEER != 'none' ]; then
      COMMAND="${COMMAND} --bootstrap=${BOOTSTRAP_PEER}"
    fi

    if [ $NEURON_KEY != 'none' ]; then
      COMMAND="${COMMAND} --neuron_key=${NEURON_KEY}"
    fi    

    if [ $REMOTE_IP != 'none' ]; then
      COMMAND="${COMMAND} --remote_ip=${REMOTE_IP}"
    fi   

    if [ $MODEL_PATH != 'none' ]; then
      COMMAND="${COMMAND} --remote_ip=${MODEL_PATH}"
    fi   

    log "$COMMAND"
    eval $COMMAND &
    NeuronPID=$!
}

function main() {
    mkdir -p data/$IDENTITY
    touch data/$IDENTITY/bittensor_logs.out

    # Intro logs.
    log "=== BitTensor ==="
    log "Args {"
    log "   IDENTITY: $IDENTITY"
    log "   SERVE_ADDRESS: $SERVE_ADDRESS"
    log "   PORT: $PORT"
    log "   LOGDIR: $LOGDIR"
    log "   NEURON: neurons/$NEURON/main.py"
    log "}"
    log ""

    # Build protos
    ./scripts/build_protos.sh

    # Start the Neuron object.
    start_neuron

    # Trap control C (for clean docker container tear down.)
  function teardown() {

    kill -9 $NeuronPID
    log "=== stopped Nucleus ==="

    exit 0
  }

    # NOTE(const) SIGKILL cannot be caught because it goes directly to the kernal.
    trap teardown INT SIGHUP SIGINT SIGTERM

    # idle waiting for abort from user
    read -r -d '' _ </dev/tty
}

# Run 
main