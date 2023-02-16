#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

function echo_error {
    echo -e "${RED}[ERROR]${NC} $1"
}

function echo_warning {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function echo_info {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function echo_json {
    echo "{\"type\":\"$1\",\"message\":\"$2\"}"
}

function get_git_tag_higher_version {
    echo `git tag -l --sort -version:refname | head -n 1`
}