#!/bin/bash

####
# Utils
####
source ${BASH_SOURCE%/*}/utils.sh
###

# 1. Get options

## Defaults
APPLY="false"

while [[ $# -gt 0 ]]; do
  case $1 in
    -A|--apply)
      APPLY="true"
      shift # past argument
      ;;
    -V|--version)
      VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [[ -z $VERSION ]]; then
    echo_error "Version to release required (-V, --version)"
    exit 1
fi

DOCKER_IMAGE_NAME="opentensorfdn/bittensor:$VERSION"

# 2. Login
if [[ $APPLY == "true" ]]; then
  echo_info "Docker registry login"
  sudo docker login
else
  echo_warning "Dry run execution. Not login into docker registry"
fi

# 3. Creating docker image
if [[ $APPLY == "true" ]]; then
  echo_info "Building docker image"
  sudo docker build -t $DOCKER_IMAGE_NAME .
else
  echo_warning "Dry run execution. Not building docker image '$DOCKER_IMAGE_NAME'"
fi


# 4. Uploading docker image
if [[ $APPLY == "true" ]]; then
  echo_info "Pushing docker image"
sudo docker push $DOCKER_IMAGE_NAME
else
  echo_warning "Dry run execution. Not pushing docker image '$DOCKER_IMAGE_NAME'"
fi
