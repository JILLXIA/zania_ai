"""Download the local embedding model once. Runtime never downloads models."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_REVISION = "aa8f8b060edb00e03bfdd08813a2949946c8ba55"


def prepare(directory: str):
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=directory,
        token=False,
        allow_patterns=["*.json", "*.txt", "model_optimized.onnx"],
    )
    print(f"Embedding model ready in {Path(directory).resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=".models")
    args = parser.parse_args()
    prepare(args.directory)
