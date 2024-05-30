#!/bin/bash

PR_NUMBER=$CIRCLE_PULL_REQUEST
REPO_OWNER=$(echo $CIRCLE_PROJECT_USERNAME)
REPO_NAME=$(echo $CIRCLE_PROJECT_REPONAME)

PR_NUMBER=${PR_NUMBER##*/}

PR_DETAILS=$(curl -s \
  "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/pulls/$PR_NUMBER")

IS_DRAFT=$(echo "$PR_DETAILS" | jq -r .draft)

if [ "$IS_DRAFT" == "true" ]; then
  echo "This PR is a draft. Skipping the workflow."
  exit 0
else
  echo "This PR is not a draft. Proceeding with the workflow."
  exit 1
fi
