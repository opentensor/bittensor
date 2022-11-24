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
    -U|--update)
      VERSION_TYPE="$2"
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

if [[ $VERSION_TYPE != "major" && $VERSION_TYPE != "minor" && $VERSION_TYPE != "patch" && $VERSION_TYPE != "rc" ]]; then
  echo_error "Incorrect version type (-V|--version). Version types accepted: {major, minor, patch}"
  exit 1
fi

VERSION=$(cat VERSION)
CODE_WITH_VERSION='bittensor/__init__.py'

MAJOR=$(awk -F. '{print $1}' <<< $VERSION)
MINOR=$(awk -F. '{print $2}' <<< $VERSION)
PATCH=$(awk -F. '{print $3}' <<< $VERSION)

# RC version
RC=$(awk -F- '{print $NF}' <<< $version)
if [ -z $RC ]; then
  CURRENT_VERSION="$MAJOR.$MINOR.$PATCH"
else
  CURRENT_VERSION="$MAJOR.$MINOR.$PATCH-$RC"
fi

case $VERSION_TYPE in
    "major")
        echo_info "Applying a $VERSION_TYPE update"
        NEW_VERSION="$((MAJOR + 1)).0.0"
        ;;
    "minor")
        echo_info "Applying a $VERSION_TYPE update"
        NEW_VERSION="$MAJOR.$((MINOR + 1)).0"
        ;;
    "patch")
        echo_info "Applying a $VERSION_TYPE update"
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        ;;
    "rc")
        SUFFIX=$2
        if [ -z $SUFFIX ]; then
            echo_error "Suffix is needed when updating version to a RC"
            exit 1
        fi
        NEW_VERSION="$MAJOR.$MINOR.$PATCH-$SUFFIX"
        ;;
    *)
    echo_error "This operation is not allowed. Try one of the following: {major, minor, patch, rc}"
    exit 1
    ;;
esac


echo_info "Current version: $CURRENT_VERSION"
echo_info "New version: $NEW_VERSION"

if [[ $APPLY == "true" ]]; then
    echo_info "Updating version in code: sed -i "18,30s/$VERSION/$NEW_VERSION/g" $CODE_WITH_VERSION"
    sed -i "18,30s/$VERSION/$NEW_VERSION/g" $CODE_WITH_VERSION
    echo_info "Updating version in file: echo -n $NEW_VERSION > VERSION"
    echo -n $NEW_VERSION > VERSION
else
    echo_warning "Dry run execution. Version update not applied"
    echo_info "Use -A or --apply to apply changes"
fi