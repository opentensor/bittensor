#!/bin/bash

####
# Utils
####
source ${BASH_SOURCE%/*}/utils.sh

#
# Params:
#   - First positional argument: version of the tag
#
function tag_repository()
{
    VERSION=$1

    if [[ -z $VERSION ]]; then
        echo_error "tag_repository needs VERSION"
        exit 1
    fi

    git tag -a $VERSION -m "Release $VERSION"
    git push origin --tags
}

#
# Params:
#   - First positional argument: version of the tag
#
function remove_tag()
{
    VERSION=$1

    if [[ -z $VERSION ]]; then
        echo_error "remove_tag needs VERSION"
        exit 1
    fi

    git tag -d $VERSION
    git push --delete origin $VERSION
}

#
# Needs:
#   - TAG_NAME
#   - PREV_TAG_NAME
#   - RELEASE_NAME
#
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

#
# Needs:
#   - TAG_NAME
#   - RELEASE_BRANCH
#   - RELEASE_NAME
#
function generate_github_release_notes_for_changelog_post_data()
{
  cat <<EOF
{
  "tag_name":"$TAG_NAME",
  "target_commitish":"$RELEASE_BRANCH",
  "name":"$RELEASE_NAME",
  "draft":false,
  "prerelease":false,
  "generate_release_notes":false
}
EOF
}

#
# Needs:
#   - TAG_NAME
#   - PREV_TAG_NAME
#   - RELEASE_NAME
#
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

#
# Params:
#   - First positional argument: github access token
#
function generate_github_release_notes()
{
    SECRET=$1

    if [[ -z $SECRET ]]; then
        echo_error "generate_github_release_notes needs SECRET"
        exit 1
    fi

    curl --silent \
        -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $SECRET" \
        https://api.github.com/repos/opentensor/bittensor/releases/generate-notes \
        --data "$(generate_github_release_notes_post_data)"
}

#
# Params:
#   - First positional argument: github access token
#
function generate_github_release_notes_for_changelog()
{
    SECRET=$1

    if [[ -z $SECRET ]]; then
        echo_error "generate_github_release_notes_for_changelog needs SECRET"
        exit 1
    fi

    curl --silent \
        -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $SECRET" \
        https://api.github.com/repos/opentensor/bittensor/releases/generate-notes \
        --data "$(generate_github_release_notes_for_changelog_post_data)"
}

#
# Params:
#   - github access token
#
# Needs:
#   - function 'generate_github_release_post_data' to provide that request data
#   - TAG_NAME which is created within the main github_release script
#   - RELEASE_NAME which is created within the main github_release script
#   - DESCRIPTION which is generated with a previous call of 'generate_github_release_notes'
#
function create_github_release()
{
    SECRET=$1

    if [[ -z $SECRET ]]; then
        echo_error "create_github_release needs SECRET"
        exit 1
    fi

    curl --silent \
        -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer $SECRET" \
        https://api.github.com/repos/opentensor/bittensor/releases \
        --data "$(generate_github_release_post_data)" > /dev/null
}