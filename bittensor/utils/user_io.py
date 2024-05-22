# The MIT License (MIT)
# Copyright © 2021-2022 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor
from rich.prompt import Confirm, Prompt
import sys

INPUT_WIDTH = 60

def formatted_prompt(name, default_value):
    formatted_message = "Enter {}".format(name)
    aligned_message = "{:<{width}}".format(formatted_message, width=INPUT_WIDTH - len(str(default_value)) - 3)
    return Prompt.ask(aligned_message, default=str(default_value))

def user_input_float(name, default_value):
    val = formatted_prompt(name, default_value)
    if val == default_value:
        return default_value
    try:
        result = float(val)
    except ValueError:
        bittensor.__console__.print(
            ":cross_mark: [red]Invalid {}[/red]: [bold white]{}[/bold white]".format(
                name,
                val
            )
        )
        sys.exit()
    return result

def user_input_int(name, default_value):
    val = formatted_prompt(name, default_value)
    if val == default_value:
        return default_value
    try:
        result = int(val)
    except ValueError:
        bittensor.__console__.print(
            ":cross_mark: [red]Invalid {}[/red]: [bold white]{}[/bold white]".format(
                name,
                val
            )
        )
        sys.exit()
    return result

def user_input_str(name, default_value):
    return str(formatted_prompt(name, default_value))

def user_input_confirmation(action):
    formatted_message = "Would you like to {}".format(action)
    aligned_message = "{:<{width}}".format(formatted_message, width=INPUT_WIDTH-6)
    return Confirm.ask(aligned_message)

def print_summary_header(message):
    bittensor.__console__.print("=============================================================")
    bittensor.__console__.print("   {}".format(message))

def print_summary_item(name, value):
    bittensor.__console__.print("      {}: {}".format(name, value))

def print_summary_message(message):
    bittensor.__console__.print("      {}".format(message))

def print_summary_footer():
    bittensor.__console__.print("-----------------------------")
