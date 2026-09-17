#!/usr/bin/env python3

# Source: https://huggingface.co/litert-community/deit_tiny_patch16_224

import os

from huggingface_hub import hf_hub_download

REPO = "litert-community/deit_tiny_patch16_224"
FILE = "model.tflite"
DEST = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    path = hf_hub_download(REPO, FILE, local_dir=DEST)
    print(f"Download model successful: {path}")


if __name__ == "__main__":
    main()
