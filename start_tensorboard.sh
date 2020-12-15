#!/usr/bin/env bash
set -o errexit

# Change to script's directory
cd "$(dirname "$0")"

function log() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.debug(\"$1\")"
}

neuron='mnist'

# Read command line args
while test 2 -gt 0; do
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
    *)
      break
      ;;
  esac
done

function start_tensorboard() {
    log "=== start Tensorboard ==="

    tensorboard --logdir=data --reload_multifile True 
    TensorboardPID=$!
}

function main() {

    # Trap control C (for clean docker container tear down.)
    function teardown() { 
        exit 0
    }

    trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

    start_tensorboard
}

main


