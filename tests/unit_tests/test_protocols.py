# The MIT License (MIT)
# Copyright © 2023 crazydevlegend

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

import base64
import json
import bittensor as bt

from bittensor.subnets import TextToImage

def test_text_to_image_synapse():
    # Define a mock headers dictionary to use for testing
    headers = {
        "bt_header_axon_nonce": 111,
        "bt_header_dendrite_ip": "1.1.1.1",
        "bt_header_input_obj_text": base64.b64encode(json.dumps("cat sitting on a table").encode("utf-8")).decode("utf-8"),
        "bt_header_input_obj_negative_prompt": base64.b64encode(json.dumps("ugly").encode("utf-8")).decode("utf-8"),
        "timeout": 12,
        "header_size": 111,
        "total_size": 111,
        "computed_body_hash": "0xabcdef",
    }

    # Run the function to test
    sn5_synapse_test = TextToImage.from_headers(headers)

    # Check that the resulting object is an instance of TextToImage
    assert isinstance(sn5_synapse_test, TextToImage)

    # Check the properties of the resulting object
    assert sn5_synapse_test.text == "cat sitting on a table"
    assert sn5_synapse_test.negative_prompt == "ugly"

    assert sn5_synapse_test.axon.nonce == 111
    assert sn5_synapse_test.dendrite.ip == "1.1.1.1"
    assert sn5_synapse_test.timeout == 12
    assert sn5_synapse_test.name == "TextToImage"
    assert sn5_synapse_test.header_size == 111
    assert sn5_synapse_test.total_size == 111
    assert sn5_synapse_test.computed_body_hash == "0xabcdef"

def test_text_to_image_dendrite_call():
    # Define a mock headers dictionary to use for testing
    headers = {
        "bt_header_input_obj_text": base64.b64encode(json.dumps("cat sitting on a table").encode("utf-8")).decode("utf-8"),
        "bt_header_input_obj_negative_prompt": base64.b64encode(json.dumps("ugly").encode("utf-8")).decode("utf-8"),
        "timeout": 12,
    }

    # Run the function to test
    sn5_synapse_test = TextToImage.from_headers(headers)

    # Get metagraph and dendrite
    sn5 = bt.metagraph(5)
    d = bt.dendrite()

    # Call the dendrite
    sn5_out = d.query(sn5.axons[1], sn5_synapse_test)
    
    
    # Check that the resulting object is an instance of TextToImage
    assert isinstance(sn5_out, TextToImage)

    # Check the properties of the resulting object
    assert sn5_out.text == "cat sitting on a table"
    assert sn5_out.negative_prompt == "ugly"
    assert sn5_out.height == 512
    assert sn5_out.width == 512
    for image in sn5_out.images:    # Check the shape of each image in the list
        assert image.shape == [3, 512, 512]

        assert sn5_out.timeout == 12
    assert sn5_out.name == "TextToImage"
