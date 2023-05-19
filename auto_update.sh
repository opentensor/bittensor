#!/bin/bash

# Initialize variables
script=""
proc_name="auto_run_bittensor" 
args=()

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --script) script="$2"; shift ;;
        --name) name="$2"; shift ;;
        --*) args+=("$1=$2"); shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if script argument was provided
if [[ -z "$script" ]]; then
    echo "The --script argument is required."
    exit 1
fi

branch=$(git branch --show-current)            # get current branch.
echo watching branch: $branch
echo pm2 process name: $proc_name

# Get the current file hash
current_hash=$(git log -n 1 --pretty=format:%H -- $script)
echo current_hash: $current_hash

# Check if script is already running with pm2
if pm2 status | grep -q $proc_name; then
    echo "The script is already running with pm2. Stopping and restarting..."
    pm2 delete $proc_name
fi

# Run the Python script with the arguments using pm2
echo "Running $script with the following arguments with pm2:"
echo "${args[@]}"
pm2 start "$script" --name $proc_name --interpreter python3 -- "${args[@]}"

while true; do
    # Fetch the latest changes from the repository
    git fetch origin $branch

    # Get the latest file hash
    latest_hash=$(git log -n 1 --pretty=format:%H -- origin $branch -- $script)
    echo "current script hash:" "$current_hash" 
    echo "latest script hash:" "$latest_hash" 

    # If the file has been updated
    if [ "$current_hash" != "$latest_hash" ]; then
        echo "The file has been updated. Updating the local copy."

        # Pull the latest changes
        git pull

        # Update the current file hash
        current_hash=$latest_hash

        # Check if script is already running with pm2
        if pm2 status | grep -q $proc_name; then
            echo "The script is already running with pm2. Stopping and restarting..."
            pm2 delete $proc_name
        fi

        # Run the Python script with the arguments using pm2
        echo "Running $script with the following arguments with pm2:"
        echo "${args[@]}"
        pm2 start "$script" --name $proc_name --interpreter python3 -- "${args[@]}"

        echo ""
    else
        echo ""
    fi

    # Wait for a while before the next check
    sleep 5
done
