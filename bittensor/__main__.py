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
