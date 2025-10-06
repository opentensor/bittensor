"""
This module serves primarily as a reference and auxiliary resource for developers.

Although any command can be constructed directly within a test without relying on the pre-generated call definitions,
the provided command lists (divided into sudo and non-sudo categories), together with the pallet reference,
significantly streamline the creation of accurate, maintainable, and well-structured end-to-end tests.

In practice, these definitions act as convenient blueprints for composing extrinsic calls and understanding the
structure of available Subtensor operations.
"""

import os
from bittensor import Subtensor
from bittensor.extras.dev_framework.calls.sudo_calls import *  # noqa: F401
from bittensor.extras.dev_framework.calls.non_sudo_calls import *  # noqa: F401
from bittensor.extras.dev_framework.calls.pallets import *  # noqa: F401

HEADER = '''"""
This file is auto-generated. Do not edit manually.

For developers:
- Use the function `recreate_calls_subpackage()` to regenerate this file.
- The command lists are built dynamically from the current Subtensor metadata (`Subtensor.substrate.metadata`).
- Each command is represented as a `namedtuple` with fields:
    * System arguments: wallet, pallet (and `sudo` for sudo calls).
    * Additional arguments: taken from the extrinsic definition (with type hints for reference).
- These namedtuples are intended as convenient templates for building commands in tests and end-to-end scenarios.

Note:
    Any manual changes will be overwritten the next time the generator is run.
'''

IMPORT_TEXT = '''
"""

from collections import namedtuple


'''


def recreate_calls_subpackage(network="local"):
    """Fetch the list of pallets and their call and save them to the corresponding modules."""
    sub = Subtensor(network=network)

    spec_version = sub.query_constant("System", "Version").value["spec_version"]
    spec_version_text = f"    Subtensor spec version: {spec_version}"
    non_sudo_calls = []
    sudo_calls = []
    pallets = []
    meta = sub.substrate.metadata
    for pallet in meta.pallets:
        calls = getattr(pallet.calls, "value", [])
        if calls:
            pallets.append(pallet.name)
        for call in calls:
            name = call.get("name")
            fields_lst = call.get("fields", [])
            fields_and_annot = [
                f"{f.get('name')}: {f.get('typeName')}" for f in fields_lst
            ]
            fields = [f'"{f.get("name")}"' for f in fields_lst]

            if name.startswith("sudo_"):
                sudo_calls.append(
                    f'{name.upper()} = namedtuple("{name.upper()}", ["wallet", "pallet", "sudo", {", ".join(fields)}])  '
                    f"# args: [{', '.join(fields_and_annot)}]  | Pallet: {pallet.name}"
                )
            else:
                non_sudo_calls.append(
                    f'{name.upper()} = namedtuple("{name.upper()}", ["wallet", "pallet", {", ".join(fields)}])  '
                    f"# args: [{', '.join(fields_and_annot)}]  | Pallet: {pallet.name}"
                )

    sudo_text = (
        HEADER + spec_version_text + IMPORT_TEXT + "\n".join(sorted(sudo_calls)) + "\n"
    )
    non_sudo_text = (
        HEADER
        + spec_version_text
        + IMPORT_TEXT
        + "\n".join(sorted(non_sudo_calls))
        + "\n"
    )
    pallets_text = f'""""\n{spec_version_text} \n"""\n' + "\n".join(
        [f'{p} = "{p}"' for p in pallets]
    )

    sudo_calls_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sudo_calls.py"
    )
    non_sudo_calls_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "non_sudo_calls.py"
    )
    pallets_file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pallets.py"
    )

    # rewrite sudo_calls.py
    with open(sudo_calls_file_path, "w") as f:
        f.write(sudo_text)

    print(f"Module {sudo_calls_file_path} has been recreated successfully.")

    # rewrite non_sudo_calls.py
    with open(non_sudo_calls_file_path, "w") as f:
        f.write(non_sudo_text)

    print(f"Module {non_sudo_calls_file_path} has been recreated successfully.")

    # rewrite pallets.py
    with open(pallets_file_path, "w") as f:
        f.write(pallets_text)

    print(f"Module {pallets_file_path} has been recreated successfully.")


if __name__ == "__main__":
    recreate_calls_subpackage()
