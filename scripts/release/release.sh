#!/bin/bash

#
# In this script you are going to find the process of releasing bittensor.
#
# This script needs:
#   - That the current VERSION does not already exist
#   - An existing pass secret with the key github/api_bash_access_token
#     - Check pass if you don't know it: https://www.passwordstore.org/
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

help(){
    echo Usage:
    echo \ \  $0
    echo
    echo This script release a bittensor version.
    echo
    echo This script needs:
    echo \ \ - That the current VERSION does not already exist
    echo \ \ - An existing pass secret with the key github/api_bash_access_token
    echo \ \ \ \ - Check pass if you do not know it: https://www.passwordstore.org/
    echo
    echo The release process will generate:
    echo \ \ - Tag in Github repo: https://github.com/opentensor/bittensor/tags
    echo \ \ - Release in Github: https://github.com/opentensor/bittensor/releases
    echo \ \ - New entry in CHANGELOG.md file
    echo \ \ - Python wheel in pypi: https://pypi.org/project/bittensor/
    echo \ \ - Docker image in dockerhub: https://hub.docker.com/r/opentensorfdn/bittensor/tags
    echo
}

generate_github_release_notes_post_data()
{
  cat <<EOF
{
  "tag_name":"$TAG_NAME",
  "name":"$RELEASE_NAME",
  "draft":false,
  "prerelease":false,
  "generate_release_notes":false
}
EOF
}

generate_github_release_post_data()
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

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

function echo_error {
    echo -e "${RED}[ERROR]${NC} $1"
}

function echo_info {
    echo -e "${GREEN}[INFO]${NC} $1"
}

###
# Start of release process
###

# if the user requests help, the usage is shown.
if [ "$1" == "--help" -o "$1" == "-h" ];then
    help
    exit 0
fi

VERSION=$(cat VERSION)
TAG_NAME=v$VERSION

# 0. Check requirements

## 0.1. Current VERSION is not already a tag

CURRENT_VERSION_EXISTS=$(git tag | grep $VERSION)

if [[ ! -z $CURRENT_VERSION_EXISTS ]]; then
    echo_error "Current version '$VERSION' already exists"
    help
    exit 1
fi

# 1. Update version if applied
## Using script: ./scripts/update_version.sh

echo_info "Detected version: $VERSION"
echo_info "Tag generated: $TAG_NAME"

# 2. Tag the repository with version
function tag_repository()
{
    git tag -a v$VERSION -m "Release $VERSION"
    git push origin --tags
}

echo_info "Tagging repository"
tag_repository

# 3. Create Github release
function generate_github_release_notes()
{
    curl --silent \
        -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $(pass github/api_bash_access_token)" \
        https://api.github.com/repos/opentensor/bittensor/releases/generate-notes \
        --data "$(generate_github_release_notes_post_data)"
}

function create_github_release()
{
    curl --silent \
        -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $(pass github/api_bash_access_token)" \
        https://api.github.com/repos/opentensor/bittensor/releases \
        --data "$(generate_github_release_post_data)" > /dev/null
}

DATE=$(date +"%Y-%m-%d")
RELEASE_NAME="$VERSION / $DATE"

echo_info "Generating Github release notes"
DESCRIPTION=$(generate_github_release_notes | jq '.body' | tail -1 | sed "s/\"//g")

echo_info "Generating Github release"
create_github_release

echo_info "Adding release notes to CHANGELOG.md"
sed -i "2 i\\\n## $RELEASE_NAME" CHANGELOG.md
sed -i "4 i\\\n$DESCRIPTION\n" CHANGELOG.md

# 4. Generate python wheel
echo_info "Removing dirs: dist/ and build/"
rm -rf dist/
rm -rf build/

echo_info "Generating python wheel"
python3 setup.py sdist bdist_wheel

# 5. Upload pypi wheel
python3 -m twine upload dist/*

# 6. Creating docker image

# 7. Uploading docker image