# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import subprocess
import sys

from bittensor import __version__

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "certifi":
        # Resolve the path to certifi.sh
        certifi_script = os.path.join(os.path.dirname(__file__), "utils", "certifi.sh")
        if not os.path.exists(certifi_script):
            print(f"Error: certifi.sh not found at {certifi_script}")
            sys.exit(1)

        # Ensure the script is executable
        os.chmod(certifi_script, 0o755)

        # Run the script
        subprocess.run([certifi_script], check=True)
    else:
        print(f"Bittensor SDK version: {__version__}")
