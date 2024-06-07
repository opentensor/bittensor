#!/bin/bash

# black checks formating
echo ">>> Run the pre-submit format check with \`black .\`."
python3 -m black --exclude '(env|venv|.eggs|.git)' .

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
