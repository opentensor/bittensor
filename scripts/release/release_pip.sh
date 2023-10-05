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

# 2. Cleaning up
echo_info "Removing dirs: dist/ and build/"
rm -rf dist/
rm -rf build/

# 3. Creating python wheel
echo_info "Generating python wheel"
python3 setup.py sdist bdist_wheel

# 3. Upload wheel to pypi
if [[ $APPLY == "true" ]]; then
    echo_info "Uploading python wheel"
    python3 -m twine upload --repository bittensor dist/*
else
    echo_warning "Dry run execution. Not uploading python wheel"
fi
