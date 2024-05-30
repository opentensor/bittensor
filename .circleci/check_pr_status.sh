#!/bin/bash

PR_DETAILS=$(curl -s $CIRCLE_PULL_REQUEST)

IS_DRAFT=$(echo "$PR_DETAILS" | jq -r .draft)

if [ "$IS_DRAFT" == "true" ]; then
  echo "This PR is a draft. Skipping the workflow."
  exit 0
else
  echo "This PR is not a draft. Proceeding with the workflow."
  exit 1
fi
