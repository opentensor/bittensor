#!/bin/bash

# Extract the repository owner
REPO_OWNER=$(echo $CIRCLE_PULL_REQUEST | awk -F'/' '{print $(NF-3)}')

# Extract the repository name
REPO_NAME=$(echo $CIRCLE_PULL_REQUEST | awk -F'/' '{print $(NF-2)}')

# Extract the pull request number
PR_NUMBER=$(echo $CIRCLE_PULL_REQUEST | awk -F'/' '{print $NF}')


PR_DETAILS=$(curl -s \
  "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/pulls/$PR_NUMBER")


IS_DRAFT=$(echo "$PR_DETAILS" | jq -r .draft)
echo $IS_DRAFT

if [ "$IS_DRAFT" == "true" ]; then
  echo "This PR is a draft. Skipping the workflow."
  exit 1
else
  echo "This PR is not a draft. Proceeding with the workflow."
  exit 0
fi
