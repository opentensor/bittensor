#!/bin/bash

# Check if requirements files have changed in the last commit
if git diff --name-only HEAD~1 | grep -E 'requirements/prod.txt|requirements/dev.txt'; then
    echo "Requirements files have changed. Running compatibility checks..."
    echo 'export REQUIREMENTS_CHANGED="true"' >> $BASH_ENV
else
    echo "Requirements files have not changed. Skipping compatibility checks..."
    echo 'export REQUIREMENTS_CHANGED="false"' >> $BASH_ENV
fi
