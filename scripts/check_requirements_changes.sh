#!/bin/bash

# Check if requirements files have changed in the last commit
if git diff --name-only HEAD~1 | grep -E 'requirements/prod.txt|requirements/dev.txt'; then
    echo "Requirements files have changed. Running compatibility checks..."
    exit 0
else
    echo "Requirements files have not changed. Skipping compatibility checks..."
    exit 1
fi
