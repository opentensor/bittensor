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

function start_neuron() {
    log ""
    log "=== start Neuron ==="
    COMMAND="python3 ./examples/$NEURON/main.py"
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