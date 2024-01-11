"""
The MIT License (MIT)
Copyright © 2023 Chris Wilson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the “Software”), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from typing import Optional, List, Dict
import bittensor as bt
import pydantic


"""
Represents a software version with major, minor, and patch components.
"""
class Version (pydantic.BaseModel):
    major_version: Optional[int] = None
    minor_version: Optional[int] = None
    patch_version: Optional[int] = None

"""
Extends the Bittensor Synapse with an additional version attribute, 
used for compatibility and version control in operations.
"""
class ScrapingSynapse ( bt.Synapse ):
    version: Optional[Version] = None

class RedditScrap(ScrapingSynapse):
    """
    RedditScrap class inherits from ScrapingSynapse.
    It is used to scrape data from Reddit.
    """
    # Required request input, filled by sending dendrite caller.
    scrap_input: Optional[Dict] = None

    # Optional request output, filled by receiving axon.
    # TODO: Add error handling for when scrap_output is None
    scrap_output: Optional[List[Dict]] = None

    def deserialize(self) -> List[Dict]:
        """
        Deserialize the scrap_output into a list of dictionaries.
        """
        # TODO: Add error handling for when scrap_output is None
        return self.scrap_output

class TwitterScrap(ScrapingSynapse):
    """
    TwitterScrap class inherits from ScrapingSynapse.
    It is used to scrape data from Twitter.
    """
    # Required request input, filled by sending dendrite caller.
    scrap_input: Optional[Dict] = None

    # Optional request output, filled by receiving axon.
    scrap_output: Optional[List[Dict]] = None

    def deserialize(self) -> List[Dict]:
        """
        Deserialize the scrap_output into a list of dictionaries.
        """
        # TODO: Add error handling for when scrap_output is None
        return self.scrap_output

class CheckMiner(ScrapingSynapse):
    """
    CheckMiner class inherits from ScrapingSynapse.
    It is used to check the miner's status.
    """
    # Required request input, send url_hash for check
    check_url_hash: str

    # Return response with tweet searched by url_hash
    # TODO: Add error handling for when check_output is None
    check_output: Optional[Dict] = None

    def deserialize(self) -> Dict:
        """
        Deserialize the check_output into a dictionary.
        """
        # TODO: Add error handling for when check_output is None
        return self.check_output

