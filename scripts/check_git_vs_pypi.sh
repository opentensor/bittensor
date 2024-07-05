#!/bin/bash

# requirements
sudo apt install -y jq curl tar

# create variable for workspace
export GHPP_WORKSPACE=$HOME/ghppc_tmp


# Function to create a temp workspace for diff checking
make_clean_diff_check_workspace() {
  local exit=$1
  
  rm -rf $GHPP_WORKSPACE
  if [ $exit -gt -1 ]; then
    exit $exit
  fi
  mkdir -p $GHPP_WORKSPACE
}

# Function to fetch the specified or latest release from GitHub
fetch_github_release() {
  local org=$1
  local repo=$2
  local version=$3
  
  local release_url
  if [ "$version" = "latest" ]; then
    release_url=$(curl -s "https://api.github.com/repos/$org/$repo/releases/latest" | grep tarball_url | cut -d '"' -f 4)
  else
    release_url=$(curl -s "https://api.github.com/repos/$org/$repo/releases/tags/$version" | grep tarball_url | cut -d '"' -f 4)
  fi
  
  if [ -z "$release_url" ]; then
    # Try again with v prefix for tagging versions
    release_url=$(curl -s "https://api.github.com/repos/$org/$repo/releases/tags/v$version" | grep tarball_url | cut -d '"' -f 4)
    if [ -z "$release_url" ]; then
      echo "Failed to fetch GitHub release URL ($release_url)"
      make_clean_diff_check_workspace 1
    fi
  fi
  
  local github_tarball="github_${version}.tar.gz"
  cd $GHPP_WORKSPACE/
  curl -sL -o $github_tarball $release_url
  mkdir -p github_source
  tar -xzf $github_tarball -C github_source --strip-components=1 > /dev/null 2>&1
}

# Function to fetch the specified or latest source from PyPI
fetch_pypi_source() {
  local package=$1
  local version=$2
  
  local release_url
  if [ "$version" = "latest" ]; then
    release_url=$(curl -s "https://pypi.org/pypi/$package/json" | jq -r '.urls[] | select(.packagetype=="sdist") | .url')
  else
    release_url=$(curl -s "https://pypi.org/pypi/$package/$version/json" | jq -r '.urls[] | select(.packagetype=="sdist") | .url')
  fi
  
  if [ -z "$release_url" ]; then
    echo "Failed to fetch PyPI release URL."
    make_clean_diff_check_workspace 1
  fi
  
  local pypi_tarball="pypi_${version}.tar.gz"
  cd $GHPP_WORKSPACE/
  curl -sL -o $pypi_tarball $release_url
  mkdir -p pypi_source
  tar -xzf $pypi_tarball -C pypi_source --strip-components=1 > /dev/null 2>&1
}

# Function to perform a diff check on the specified directory
perform_diff_check() {
  local dir=$1
  local continue_on_error=$2
  local repnm=$3
  local vernm=$4

  cd $GHPP_WORKSPACE/
  diff_output=$(diff -r "github_source/$dir" "pypi_source/$dir")

  if [ -z "$diff_output" ]; then
    echo "No differences found between the GitHub and PyPI sources for the '$dir' directory."
  else
    echo "$diff_output"
    
    # Copy before continuing to preserve 
    echo "Repos with diff will be kept in $GHPP_WORKSPACE-$repnm-$vernm-hasdiff"
    rm -rf $GHPP_WORKSPACE-$repnm-$vernm-hasdiff
    mv $GHPP_WORKSPACE $GHPP_WORKSPACE-$repnm-$vernm-hasdiff
    sleep 5

    if [ $continue_on_error = 0 ]; then
      echo "Possibly unsafe to proceeed, exiting!"
      make_clean_diff_check_workspace 1
    fi    
  fi
}

# Function to fetch all GitHub releases
fetch_and_check_all_github_releases() {
  local gorg=$1
  local grepo=$2
  local prepo=$3
  releases=$(curl -s "https://api.github.com/repos/$gorg/$grepo/releases" | jq -r '.[].tag_name')

  for release in $releases; do
    echo "Fetching GitHub release $release"
    fetch_github_release $gorg $grepo $release
    echo "Fetching PyPI source $release"
    fetch_pypi_source $prepo $release
    echo "Performing diff check on the repos"
    perform_diff_check $grepo 1 $grepo $release
    make_clean_diff_check_workspace -1
  done
}


# Main script
main() {
  if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <github_org> <github_repo> <pypi_repo> [version]"
    make_clean_diff_check_workspace 1
  fi
  
  local github_org=$1
  local github_repo=$2
  local pypi_repo=$3
  local version=${4:-latest}

  make_clean_diff_check_workspace -1
  
  if [ "$version" = "all" ]; then
    fetch_and_check_all_github_releases $github_org $github_repo $pypi_repo
  else
    echo "Fetching GitHub release from $github_org/$github_repo (version: $version)"
    fetch_github_release $github_org $github_repo $version
    
    echo "Fetching PyPI source for $pypi_repo (version: $version)"
    fetch_pypi_source $pypi_repo $version
    
    echo "Performing diff check on the repos"
    perform_diff_check $github_repo 0 $github_repo $version
  fi
  make_clean_diff_check_workspace 0
}

main "$@"