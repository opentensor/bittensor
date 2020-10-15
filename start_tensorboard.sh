#!/usr/bin/env bash
set -o errexit

# Change to script's directory
cd "$(dirname "$0")"

# Load constants
source ./scripts/constant.sh

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

function update_machine_logs() {
    # Retrieve all docker machines
    docker_machines=($(docker-machine ls --format '{{ .Name }}'))

    for machine_name in "${docker_machines[@]}"
    do
        # Switch context to the current docker machine we are servicing
        eval $(docker-machine env $machine_name)

        # Docker container for this machine should be mounted on local machine now. Let's verify. 
        container_name=$(docker container ls --format '{{ .Names }}' | grep "$machine_name")
        if [[ $machine_name == $container_name ]]; then
            log "Machine container "$machine_name" successfully mounted locally. Extracting new data."
            model_data_dir=$(docker exec $machine_name ls /bittensor/data | grep "$neuron")
            
            # Make a directory locally for this machine's data
            if [ ! -d "data/$model_data_dir" ]; then
                mkdir data/$model_data_dir
            fi

            # retrieve this machine's data and place it in the local host
            rsync --inplace -avPe 'docker exec -i' -r $machine_name:/bittensor/data/$model_data_dir/ data/$model_data_dir
        fi
    done
}

function start_tensorboard() {
    log "=== start Tensorboard ==="

    tensorboard --logdir=data --reload_multifile True &
    TensorboardPID=$!
}

function main() {

    # Trap control C (for clean docker container tear down.)
    function teardown() { 
        exit 0
    }

    trap teardown INT SIGHUP SIGINT SIGTERM ERR EXIT

    start_tensorboard

    while true; do
        update_machine_logs 
        sleep 5
    done &

}

main


