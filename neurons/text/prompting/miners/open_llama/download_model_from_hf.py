import os
import requests
import argparse

from transformers import HfApi

def download_file(url, local_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

def main(model_name, local_folder):
    api = HfApi()

    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    model_info = api.model_info(model_name)

    for file in model_info.files:
        file_url = file.url
        file_name = file.filename
        local_path = os.path.join(local_folder, file_name)

        print(f"Downloading {file_name}...")
        download_file(file_url, local_path)

    print("All files downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="openlm-research/open_llama_7b_preview_300bt", help="Hugging Face model name")
    parser.add_argument("--local_folder", default="open_llama_7b_preview_300bt_transformers_weights", help="Local folder to save the model files")

    args = parser.parse_args()

    main(args.model_name, args.local_folder)