"""Upload only the model files used by DTA inference to a Hugging Face model repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Hugging Face model repo, e.g. username/target-screening-models")
    parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    files = [(root / "vendor/PLM/checkpoint_step500000.pt", "PLM/checkpoint_step500000.pt")]
    for dataset in ("DAVIS-complete", "IC50", "Ki", "Kd"):
        for fold in range(1, 6):
            files.append((root / f"vendor/DTA_model/result/{dataset}/fold{fold}/best.pth",
                          f"DTA_model/result/{dataset}/fold{fold}/best.pth"))

    for local_path, repo_path in files:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        print(f"Uploading {repo_path}", flush=True)
        api.upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_path,
                        repo_id=args.repo_id, repo_type="model")
    print(f"Uploaded {len(files)} files to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
