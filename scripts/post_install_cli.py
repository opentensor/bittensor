import os
import subprocess
import sys


def post_install():
    # Determine the shell type (bash, zsh, etc.)
    shell = os.environ.get("SHELL")
    if "bash" in shell:
        shell_config = "~/.bashrc"
    elif "zsh" in shell:
        shell_config = "~/.zshrc"
    else:
        print("Unsupported shell for autocompletion.")
        return

    # Generate the completion script
    completion_script = subprocess.check_output(
        [sys.executable, "-m", "bittensor.cli", "--print-completion", shell]
    ).decode()

    # Append the completion script to the shell configuration file
    with open(os.path.expanduser(shell_config), "a") as file:
        file.write("\n# Bittensor CLI Autocompletion\n")
        file.write(completion_script)


if __name__ == "__main__":
    post_install()
