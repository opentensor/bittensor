#!/bin/bash

#
# In this script you are going to find the process of releasing bittensor.
#
# This script needs:
#   - An existing VERSION file
#   - Version in VERSION file is not a git tag already
#
# This process will generate:
#   - Tag in Github repo: https://github.com/opentensor/bittensor/tags
#   - Release in Github: https://github.com/opentensor/bittensor/releases
#   - New entry in CHANGELOG.md file
#   - Python wheel in pypi: https://pypi.org/project/bittensor/
#   - Docker image in dockerhub: https://hub.docker.com/r/opentensorfdn/bittensor/tags (TODO)
#

###
# Utils
###

source ${BASH_SOURCE%/*}/utils.sh

function help(){
    echo Usage:
    echo \ \  $0
    echo
    echo This script release a bittensor version.
    echo
    echo This script needs:
    echo \ \ - An existing VERSION file
    echo \ \ - Version in VERSION file is not a git tag already
    echo
}
###

###
# Start of release process
###

# 0. Check requirements
# Expected state for the execution environment
#  - VERSION file exists
#  - __version__ exists inside file 'bittensor/__init__.py'
#  - Both versions have the expected format and are the same

VERSION_FILENAME='VERSION'
CODE_WITH_VERSION='bittensor/__init__.py'

if [[ ! -f $VERSION_FILENAME ]]; then
  echo_error "Requirement failure: $VERSION_FILENAME does not exist"
  help
  exit 1
fi

CODE_VERSION=`grep '__version__\ \=\ ' $CODE_WITH_VERSION | awk '{print $3}' | sed "s/'//g"`
VERSION=$(cat $VERSION_FILENAME)

if ! [[ "$CODE_VERSION" =~ ^[0-9]+.[0-9]+.[0-9]+$ ]];then
  echo_error "Requirement failure: Version in code '$CODE_VERSION' with wrong format"
  exit 1
fi

if ! [[ "$VERSION" =~ ^[0-9]+.[0-9]+.[0-9]+$ ]];then
  echo_error "Requirement failure: Version in file '$VERSION' with wrong format"
  exit 1
fi

if [[ $CODE_VERSION != $VERSION ]]; then
  echo_error "Requirement failure: version in code ($CODE_VERSION) and version in file ($VERSION) are not the same. You should fix that before release code."
  help
  exit 1
fi

# 1. Get options

## Defaults
APPLY="false"
APPLY_ACTION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      exit 0
      ;;
    -A|--apply)
      APPLY="true"
      APPLY_ACTION="--apply"
      shift # past argument
      ;;
    -T|--github-token)
      GITHUB_TOKEN="$2"
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

if [[ $APPLY == "true" ]]; then
  echo_warning "Not a Dry run exection"
else
  echo_warning "Dry run execution"
fi

if [[ -z $GITHUB_TOKEN ]]; then
    echo_error "Github token required (-T, --github-token)"
    exit 1
fi

# 2. Checking version

CURRENT_VERSION_EXISTS=$(git tag | grep $VERSION)
if [[ ! -z $CURRENT_VERSION_EXISTS ]]; then
    echo_error "Current version '$VERSION' already exists"
    help
    exit 1
fi

VERSION=$(cat $VERSION_FILENAME)
PREV_VERSION_TAG=`get_git_tag_higher_version`

TAG_NAME=v$VERSION

## 2.1. Current VERSION is not already a tag

echo_info "Detected new version tag: $VERSION"
echo_info "Previous version tag: $PREV_VERSION_TAG"
echo_info "Tag generated: $TAG_NAME"

# 3. Create Github resources
${BASH_SOURCE%/*}/release_github.sh $APPLY_ACTION --github-token $GITHUB_TOKEN -P $PREV_VERSION_TAG -V $VERSION

# 4. Generate python wheel and upload it to Pypi
if [[ $APPLY == "true" ]]; then
  echo_warning "Releasing pip package"
  ${BASH_SOURCE%/*}/release_pip.sh $APPLY_ACTION
else
  echo_warning "Dry run execution. Not releasing pip package"
fi

# 5. Creating docker image and upload
if [[ $APPLY == "true" ]]; then
  echo_warning "Releasing docker image"
  ${BASH_SOURCE%/*}/release_docker.sh $APPLY_ACTION --version $VERSION
else
  echo_warning "Dry run execution. Not releasing docker image"
fi
