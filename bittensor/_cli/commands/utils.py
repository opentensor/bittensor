
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import sys
import os
import torch
import bittensor
from typing import List, Dict, Any, Optional
from rich.prompt import Confirm, Prompt, PromptBase
import requests
from dataclasses import dataclass
console = bittensor.__console__

class IntListPrompt(PromptBase):
    """ Prompt for a list of integers. """
    
    def check_choice( self, value: str ) -> bool:
        assert self.choices is not None
        # check if value is a valid choice or all the values in a list of ints are valid choices
        return value == "All" or \
            value in self.choices or \
            all( val.strip() in self.choices for val in value.replace(',', ' ').split( ))


def check_netuid_set( config: 'bittensor.Config', subtensor: 'bittensor.Subtensor', allow_none: bool = False ):
    if subtensor.network =='finney':
        all_netuids = [str(netuid) for netuid in subtensor.get_subnets()]
        if len(all_netuids) == 0:
            console.print(":cross_mark:[red]There are no open networks.[/red]")
            sys.exit()

        # Make sure netuid is set.
        if config.get('netuid', 'notset') == 'notset':
            if not config.no_prompt:
                netuid = IntListPrompt.ask("Enter netuid", choices=all_netuids, default=str(all_netuids[0]))
            else:
                netuid = str(bittensor.defaults.netuid) if not allow_none else 'None'
        else:
            netuid = config.netuid
            
        if isinstance(netuid, str) and netuid.lower() in ['none'] and allow_none:
            config.netuid = None
        else:
            try:
                config.netuid = int(netuid)
            except ValueError:
                raise ValueError('netuid must be an integer or "None" (if applicable)')


def check_for_cuda_reg_config( config: 'bittensor.Config' ) -> None:
    """Checks, when CUDA is available, if the user would like to register with their CUDA device."""
    if torch.cuda.is_available():
        if not config.no_prompt:
            if config.subtensor.register.cuda.get('use_cuda') == None: # flag not set
                # Ask about cuda registration only if a CUDA device is available.
                cuda = Confirm.ask("Detected CUDA device, use CUDA for registration?\n")
                config.subtensor.register.cuda.use_cuda = cuda

            # Only ask about which CUDA device if the user has more than one CUDA device.
            if config.subtensor.register.cuda.use_cuda and config.subtensor.register.cuda.get('dev_id') is None:
                devices: List[str] = [str(x) for x in range(torch.cuda.device_count())]
                device_names: List[str] = [torch.cuda.get_device_name(x) for x in range(torch.cuda.device_count())]
                console.print("Available CUDA devices:")
                choices_str: str = ""
                for i, device in enumerate(devices):
                    choices_str += ("  {}: {}\n".format(device, device_names[i]))
                console.print(choices_str)
                dev_id = IntListPrompt.ask("Which GPU(s) would you like to use? Please list one, or comma-separated", choices=devices, default='All')
                if dev_id.lower() == 'all':
                    dev_id = list(range(torch.cuda.device_count()))
                else:
                    try:
                        # replace the commas with spaces then split over whitespace.,
                        # then strip the whitespace and convert to ints.
                        dev_id = [int(dev_id.strip()) for dev_id in dev_id.replace(',', ' ').split()]
                    except ValueError:
                        console.log(":cross_mark:[red]Invalid GPU device[/red] [bold white]{}[/bold white]\nAvailable CUDA devices:{}".format(dev_id, choices_str))
                        sys.exit(1)
                config.subtensor.register.cuda.dev_id = dev_id
        else:
            # flag was not set, use default value.
            if config.subtensor.register.cuda.get('use_cuda') is None: 
                config.subtensor.register.cuda.use_cuda = bittensor.defaults.subtensor.register.cuda.use_cuda

def get_hotkey_wallets_for_wallet( wallet ) -> List['bittensor.wallet']:
    hotkey_wallets = []
    hotkeys_path = wallet.path + '/' + wallet.name + '/hotkeys'
    try:
        hotkey_files = next(os.walk(os.path.expanduser(hotkeys_path)))[2]
    except StopIteration:
        hotkey_files = []
    for hotkey_file_name in hotkey_files:
        try:
            hotkey_for_name = bittensor.wallet( path = wallet.path, name = wallet.name, hotkey = hotkey_file_name )
            if hotkey_for_name.hotkey_file.exists_on_device() and not hotkey_for_name.hotkey_file.is_encrypted():
                hotkey_wallets.append( hotkey_for_name )
        except Exception:
            pass
    return hotkey_wallets

def get_coldkey_wallets_for_path( path: str ) -> List['bittensor.wallet']:
    try:
        wallet_names = next(os.walk(os.path.expanduser(path)))[1]
        return [ bittensor.wallet( path= path, name=name ) for name in wallet_names ]
    except StopIteration:
        # No wallet files found.
        wallets = []
    return wallets

def get_all_wallets_for_path( path:str ) -> List['bittensor.wallet']:
    all_wallets = []
    cold_wallets = get_coldkey_wallets_for_path(path)
    for cold_wallet in cold_wallets:
        if cold_wallet.coldkeypub_file.exists_on_device() and not cold_wallet.coldkeypub_file.is_encrypted():
            all_wallets.extend( get_hotkey_wallets_for_wallet(cold_wallet) )
    return all_wallets

@dataclass
class DelegatesDetails:
    name: str
    url: str
    description: str
    signature: str

    @classmethod
    def from_json(cls, json: Dict[str, any]) -> 'DelegatesDetails':
        return cls(
            name=json['name'],
            url=json['url'],
            description=json['description'],
            signature=json['signature'],
        )

def _get_delegates_details_from_github(requests_get, url: str) -> Dict[str, DelegatesDetails]:
    response = requests_get(url)
    

    if response.status_code == 200:
        all_delegates: Dict[str, Any] = response.json()
        all_delegates_details = {}
        for delegate_hotkey, delegates_details in all_delegates.items():
            all_delegates_details[delegate_hotkey] = DelegatesDetails.from_json(delegates_details)
        return all_delegates_details
    else:
        return {}
    
def get_delegates_details(url: str) -> Optional[Dict[str, DelegatesDetails]]: 
    try:
        return _get_delegates_details_from_github(requests.get, url)
    except Exception:
        return None # Fail silently
