""" Implementation of the bittensor tokenizer
"""
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

from transformers import AutoTokenizer
import bittensor
from bittensor.utils.tokenizer_utils import prep_tokenizer


class tokenizer:
    """ Implementation of the bittensor tokenizer
    """
    cached_tokenizer_for_version: dict = {}

    def __new__( cls, version: str = None ):
        if version == None:
            version = bittensor.__version__
        if version not in cls.cached_tokenizer_for_version:
            _tokenizer = cls.get_tokenizer_for_version( version )
            cls.cached_tokenizer_for_version[ version ] = _tokenizer
        else:
            _tokenizer = cls.cached_tokenizer_for_version[ version ]
        return _tokenizer
        
    # Tokenizer
    # NOTE (const): tokenizers are guaranteed to improve and expand as time progresses. We version the tokenizer here.
    # neurons must be aware that versions will increase and be ready to convert between tokenizers.
    # TODO (const): Add functionality to allow tokenizer conversion. i.e. for input token conversion.
    @classmethod
    def get_tokenizer_for_version( cls, version = bittensor.__version__ ):
        """ Return the GPT2 tokenizer with bittersor's special tokens
        """
        _tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=False)
        _tokenizer = prep_tokenizer(_tokenizer)
        return _tokenizer
