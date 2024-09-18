#!/bin/bash

# ruff checks formating
echo ">>> Run the pre-submit format check with \`ruff format .\`."
ruff format .

echo ">>> Run the pre-submit format check with \`mypy\`."

# mypy checks python versions compatibility
versions=("3.9" "3.10" "3.11")
for version in "${versions[@]}"; do
    echo "Running mypy for Python $version..."
    mypy --ignore-missing-imports bittensor/ --python-version="$version"
done

# flake8 checks errors count in bittensor folder
error_count=$(flake8 bittensor/ --count)
echo ">>> Flake8 found ${error_count} errors."
