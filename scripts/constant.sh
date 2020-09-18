#!/usr/bin/env bash

DOCKER_IMAGE_NAME="bittensor"
DOCKER_IMAGE_TAG="latest"

function trace() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.debug(\"$1\")" > /dev/null 2>&1
}

function log() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.debug(\"$1\")"
}

function success() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.success(\"$1\")"
}

function failure() {
    python3 -c "from loguru import logger; logger.add(\"data/$IDENTITY/bittensor_logs.out\"); logger.error(\"$1\")"
}

function whichmachine() {
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)     machine=Linux;;
        Darwin*)    machine=Mac;;
        CYGWIN*)    machine=Cygwin;;
        MINGW*)     machine=MinGw;;
        *)          machine="UNKNOWN:${unameOut}"
    esac
    echo ${machine}
}


# Parse config YAML.
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
