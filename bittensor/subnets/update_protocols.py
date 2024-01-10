import json
import os
import requests

current_dir = os.path.dirname(os.path.realpath(__file__))

# Load the protocols list
with open(f"{current_dir}/protocols_list.json", "r") as file:
    protocols = json.load(file)["protocols"]

# Ensure the protocols directory exists
os.makedirs(f"{current_dir}/protocols", exist_ok=True)

for protocol in protocols:
    name = protocol["name"]
    url = protocol["url"]
    if url:
        # Extract the raw content URL from the provided URL
        # (This might need to be adjusted based on the URL format)
        content_url = url.replace("github.com", "raw.githubusercontent.com").replace(
            "/blob/", "/"
        )

        response = requests.get(content_url)
        if response.status_code == 200:
            # Write the content to a file
            with open(f"{current_dir}/protocols/{name}.py", "w") as file:
                file.write(response.text)
