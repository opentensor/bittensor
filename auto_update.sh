#!/bin/bash

# Initialize variables
script=""
proc_name="auto_run" 
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
    git fetch origin/$branch

    # Get the latest file hash
    latest_hash=$(git log -n 1 --pretty=format:%H -- origin/$branch -- $script)
    echo git log -n 1 --pretty=format:%H -- origin/$branch -- $script

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
        
    else
        echo "The file has not been updated."
        echo "current_hash" "$current_hash" "==" "$latest_hash" 
    fi

    # Wait for a while before the next check
    sleep 1
done


# # Check for auto updates
# self_update() {
#     cd "$SCRIPTPATH"
#     git fetch origin $branch

#                                                # in the next line
#                                                # 1. added double-quotes (see below)
#                                                # 2. removed grep expression so
#                                                # git-diff will check only script
#                                                # file
#     [ -n "$(git diff --name-only "origin/$branch" "$script")" ] && {
#         echo "Found a new version of me, updating myself..."
#         git pull --force
#         git checkout "$branch"
#         git pull --force
#         echo "Running the new version..."
#         cd -                                   # return to original working dir


#         # Remove the old process.
#         if pm2 status | grep -q $proc_name; then
#             echo "The script is already running with pm2. Stopping and restarting..."
#             pm2 delete $proc_name
#         fi

#         # Re run the Python script with the arguments using pm2
#         echo "Running $script with the following arguments with pm2:"
#         echo "${args[@]}"
#         pm2 start "$script" --name $proc_name --interpreter python3 -- "${args[@]}"

#         # Now exit this old instance
#         exit 1
#     }
#     echo "Already the latest version."
# }
# self_update

# # Check if script is already running with pm2
# if pm2 status | grep -q $proc_name; then
#     echo "The script is already running with pm2. Stopping and restarting..."
#     pm2 delete $proc_name
# fi

# # Run the Python script with the arguments using pm2
# echo "Running $script with the following arguments with pm2:"
# echo "${args[@]}"
# pm2 start "$script" --name $proc_name --interpreter python3 -- "${args[@]}"






#                                                # Here I remark changes

# SCRIPT="$(readlink -f "$0")"
# echo SCRIPT: $SCRIPT

# SCRIPTFILE="$(SCRIPTFILE "$SCRIPT")"             # get name of the file (not full path)
# echo SCRIPTFILE: $SCRIPTFILE

# SCRIPTPATH="$(dirname "$SCRIPT")"
# echo SCRIPTPATH: $SCRIPTPATH

# SCRIPTNAME="$0"
# echo SCRIPTNAME: $SCRIPTNAME

# ARGS=( "$@" )                                  # fixed to make array of args (see below)
# echo ARGS: $ARGS

# script_name=$ARGS
# echo file: $SCRIPTPATH/$file

# branch=$(git branch --show-current)            # get current branch.
# echo branch: $branch

# self_update() {
#     cd "$SCRIPTPATH"
#     git fetch

#                                                # in the next line
#                                                # 1. added double-quotes (see below)
#                                                # 2. removed grep expression so
#                                                # git-diff will check only script
#                                                # file
#     [ -n "$(git diff --name-only "origin/$BRANCH" "$script_name")" ] && {
#         echo "Found a new version of me, updating myself..."
#         git pull --force
#         git checkout "$BRANCH"
#         git pull --force
#         echo "Running the new version..."
#         cd -                                   # return to original working dir
#         exec "$SCRIPTNAME" "${ARGS[@]}"

#         # Now exit this old instance
#         exit 1
#     }
#     echo "Already the latest version."
# }
# #self_update
# echo “some code”