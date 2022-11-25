#!/bin/bash

####
# Utils
####
source ${BASH_SOURCE%/*}/utils.sh

function tag_repository()
{
  VERSION=$1
  git tag -a $VERSION -m "Release $VERSION"
  git push origin --tags
}

function remove_tag()
{
  VERSION=$1
  git tag -d $VERSION
  git push --delete origin $VERSION
}

function generate_github_release_notes_post_data()
{
  cat <<EOF
{
  "tag_name":"$TAG_NAME",
  "previous_tag_name":"$PREV_TAG_NAME",
  "name":"$RELEASE_NAME",
  "draft":false,
  "prerelease":false,
  "generate_release_notes":false
}
EOF
}

function generate_github_release_post_data()
{
  cat <<EOF
{
  "tag_name":"$TAG_NAME",
  "name":"$RELEASE_NAME",
  "body":"$DESCRIPTION",
  "draft":false,
  "prerelease":false,
  "generate_release_notes":false
}
EOF
}

function generate_github_release_notes()
{
  SECRET=$1
  curl --silent \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Authorization: Bearer $SECRET" \
    https://api.github.com/repos/opentensor/bittensor/releases/generate-notes \
    --data "$(generate_github_release_notes_post_data)"
}

function create_github_release()
{
  SECRET=$1
  curl --silent \
      -X POST \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer $SECRET" \
      https://api.github.com/repos/opentensor/bittensor/releases \
      --data "$(generate_github_release_post_data)" > /dev/null
}
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

if [[ -z $GITHUB_TOKEN ]]; then
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

# 2. Github
DATE=$(date +"%Y-%m-%d")
RELEASE_NAME="$VERSION / $DATE"
PREV_TAG_NAME=$PREV_TAG_VERSION
TAG_NAME=v$VERSION

# 2.1 Create Git tag for the repository
echo_info "Tagging repository"
tag_repository $TAG_NAME

# 2.2. Generate release notes
echo_info "Generating Github release notes"
DESCRIPTION=$(generate_github_release_notes $GITHUB_TOKEN | jq '.body' | tail -1 | sed "s/\"//g")

# 2.3 Create Github release
if [[ $APPLY == "true" ]]; then
    echo_info "Generating Github release"
    create_github_release $GITHUB_TOKEN
else
    echo_warning "Dry run execution. Not creating Github release"
fi

if [[ $APPLY == "true" ]]; then
    echo_info "Adding release notes to CHANGELOG.md"
    sed -i "2 i\\\n## $RELEASE_NAME" CHANGELOG.md
    sed -i "4 i\\\n$DESCRIPTION\n" CHANGELOG.md
else
    echo_warning "Dry run execution. Not adding release notes to CHANGELOG.md"
fi

if [[ $APPLY == "false" ]]; then
    echo_warning "Dryn run. We need to remove generated tag"
    remove_tag $TAG_NAME
fi