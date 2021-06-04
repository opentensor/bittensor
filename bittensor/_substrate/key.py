# Python Substrate Interface Library
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from hashlib import blake2b

from scalecodec import ScaleDecoder

RE_JUNCTION = r'(\/\/?)([^/]+)'
JUNCTION_ID_LEN = 32


class DeriveJunction:
    def __init__(self, chain_code, is_hard=False):
        self.chain_code = chain_code
        self.is_hard = is_hard

    @classmethod
    def from_derive_path(cls, path: str, is_hard=False):

        path_scale = ScaleDecoder.get_decoder_class('Bytes')
        path_scale.encode(path)

        if len(path) > JUNCTION_ID_LEN:
            chain_code = blake2b(path_scale.data.data, digest_size=32).digest()
        else:
            chain_code = bytes(path_scale.data.data.ljust(32, b'\x00'))

        return cls(chain_code=chain_code, is_hard=is_hard)


def extract_derive_path(derive_path: str):

    path_check = ''
    junctions = []
    paths = re.findall(RE_JUNCTION, derive_path)

    if paths:
        path_check = ''.join(''.join(path) for path in paths)

        for path_separator, path_value in paths:
            junctions.append(DeriveJunction.from_derive_path(
                path=path_value, is_hard=path_separator == '//')
            )

    if path_check != derive_path:
        raise ValueError('Reconstructed path "{}" does not match input'.format(path_check))

    return junctions

