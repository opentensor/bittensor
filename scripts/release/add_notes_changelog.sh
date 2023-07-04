#!/bin/bash

####
# Utils
####
source ${BASH_SOURCE%/*}/utils.sh
source ${BASH_SOURCE%/*}/github_utils.sh
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
    -P|--previous-version-tag)
      PREV_TAG_VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -V|--version)
      VERSION="$2"
      shift # past argument
      shift # past value
      ;;
    -T|--github-token)
      GITHUB_TOKEN="$2"
      shift # past argument
      shift # past value
      ;;
    -B|--release-branch)
      RELEASE_BRANCH="$2"
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

if [[ -z $GITHUB_TOKEN && $APPLY == "true" ]]; then
  echo_error "Github token required (-T, --github-token)"
  exit 1
fi

if [[ -z $PREV_TAG_VERSION ]]; then
  echo_error "Previous version tag required (-P, --previous-version-tag)"
  exit 1
fi

if [[ -z $VERSION ]]; then
  echo_error "Version to release required (-V, --version)"
  exit 1
fi

if [[ -z $RELEASE_BRANCH ]]; then
  echo_warning "Release branch not specified with (-B, --release-branch) assuming: release/$VERSION"
  RELEASE_BRANCH=release/$VERSION
fi

DATE=$(date +"%Y-%m-%d")
RELEASE_NAME="$VERSION / $DATE"
TAG_NAME=v$VERSION
PREV_TAG_NAME=v$PREV_TAG_VERSION

# 2.2. Generate release notes
if [[ $APPLY == "true" ]]; then
  echo_info "Generating Github release notes"
  RESPONSE=$(generate_github_release_notes_for_changelog $GITHUB_TOKEN)
  DESCRIPTION=$(echo $RESPONSE | jq '.body' | tail -1 | sed "s/\"//g")

  if [ $(echo $RESPONSE | jq '.body' | wc -l) -eq 1 ]; then
    if [ $(echo $RESPONSE | jq '.' | grep 'documentation_url' | wc -l) -gt 0 ]; then
      echo_error "Something went wrong generating Github release notes"
      echo $RESPONSE | jq --slurp '.[0]'
      exit 1
    fi
  
    if [ $(echo $RESPONSE | jq '.type' | grep 'error' | wc -l) -gt 0 ]; then
      echo_error "Something went wrong generating Github release notes"
      echo $RESPONSE | jq --slurp '.[1]'
      exit 1
    fi
  fi
else
  echo_warning "Dry run execution. Not generating Github release notes"
fi

if [[ $APPLY == "true" ]]; then
  echo_info "Adding release notes to CHANGELOG.md"
  sed -i "2 i\\\n## $RELEASE_NAME" CHANGELOG.md
  sed -i "4 i\\\n$DESCRIPTION\n" CHANGELOG.md
else
  echo_warning "Dry run execution. Not adding release notes to CHANGELOG.md"
fi