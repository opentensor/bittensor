#!/bin/bash

# Locate Python 3
PYTHON=$(command -v python3)
if [ -z "$PYTHON" ]; then
    echo "Error: Python 3 is not installed or not found in PATH."
    exit 1
fi

echo "Using Python: $PYTHON"

echo " -- Upgrading the certifi package"
$PYTHON -m pip install --upgrade certifi

echo " -- Fetching the path to the certifi certificate bundle"
CERTIFI_CAFILE=$($PYTHON -c "import certifi; print(certifi.where())")

echo " -- Resolving OpenSSL directory and certificate file path"
OPENSSL_DIR=$($PYTHON -c "import ssl; print(ssl.get_default_verify_paths().openssl_cafile.rsplit('/', 1)[0])")
OPENSSL_CAFILE=$($PYTHON -c "import ssl; print(ssl.get_default_verify_paths().openssl_cafile.rsplit('/', 1)[-1])")

echo " -- Navigating to the OpenSSL directory"
cd "$OPENSSL_DIR" || { echo "Failed to navigate to $OPENSSL_DIR"; exit 1; }

echo " -- Removing any existing certificate file or symlink"
rm -f "$OPENSSL_CAFILE"

echo " -- Creating a symlink to the certifi certificate bundle"
ln -s "$CERTIFI_CAFILE" "$OPENSSL_CAFILE"

echo " -- Setting appropriate file permissions"
chmod 775 "$OPENSSL_CAFILE"

echo " -- Update complete"
