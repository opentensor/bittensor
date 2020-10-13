#!/usr/bin/env bash
set -o errexit

# Change to script's repo
source ./scripts/constant.sh

# Check script check_requirements
source scripts/check_requirements.sh

function print_help() {
    echo "Script for starting visualization instance."
    echo "Usage ./start_vis.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo " -h, --help           Print this help message and exit."
    echo " -t, --token          Digital ocean API token."
}

config_path=''

function main() {
    log "████████╗███████╗███╗░░██╗░██████╗░█████╗░██████╗░██╗░░░██╗██╗███████╗"
    log "╚══██╔══╝██╔════╝████╗░██║██╔════╝██╔══██╗██╔══██╗██║░░░██║██║╚════██║"
    log "░░░██║░░░█████╗░░██╔██╗██║╚█████╗░██║░░██║██████╔╝╚██╗░██╔╝██║░░███╔═╝"
    log "░░░██║░░░██╔══╝░░██║╚████║░╚═══██╗██║░░██║██╔══██╗░╚████╔╝░██║██╔══╝░░"
    log "░░░██║░░░███████╗██║░╚███║██████╔╝╚█████╔╝██║░░██║░░╚██╔╝░░██║███████╗"
    log "░░░╚═╝░░░╚══════╝╚═╝░░╚══╝╚═════╝░░╚════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝╚══════╝"
}