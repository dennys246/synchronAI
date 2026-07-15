"""
Hugging Face Hub helpers for synchronAI.

Provides a minimal CLI for uploading/downloading model artifacts using access tokens.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

try:
    from huggingface_hub import HfApi, snapshot_download
except ModuleNotFoundError:  # Optional dependency; fail on first use.
    HfApi = None
    snapshot_download = None

from synchronai.utils.logging import get_logger, setup_logging


MODEL_REPOS = {
    "fnirs_diffusion": "dennys246/fNIRS_diffusion",
    "fnirs_synchrony": "dennys246/fNIRS_synchrony",
    "video_synchrony": "dennys246/video_synchrony",
    "audio_synchrony": "dennys246/audio_synchrony",
    "transcript_synchrony": "dennys246/transcript_synchrony",
}


def _resolve_repo_id(model: Optional[str], repo_id: Optional[str]) -> str:
    if repo_id:
        return repo_id
    if model and model in MODEL_REPOS:
        return MODEL_REPOS[model]
    raise ValueError("Provide --repo or a valid --model name.")


def _resolve_token(token: Optional[str], token_env: Optional[str] = None) -> str:
    if token:
        return token

    env_candidates = [token_env] if token_env else []
    env_candidates += ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
    for env_name in env_candidates:
        if not env_name:
            continue
        value = os.environ.get(env_name)
        if value:
            return value

    raise ValueError(
        "No Hugging Face access token found. Pass --token or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
    )


def _safe_import_hf():
    if HfApi is None or snapshot_download is None:
        raise ModuleNotFoundError(
            "huggingface_hub is required for upload/download. Install with `pip install huggingface_hub`."
        )
    return HfApi, snapshot_download


def upload_model_dir(
    *,
    local_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    token_env: Optional[str] = None,
    commit_message: Optional[str] = None,
    private: bool = False,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
    create_pr: bool = False,
) -> str:
    """
    Upload a directory of model artifacts to a HF Hub repo.
    """
    HfApi, _ = _safe_import_hf()
    token_value = _resolve_token(token, token_env)

    local_path = Path(local_dir)
    if not local_path.exists():
        raise FileNotFoundError(f"local_dir not found: {local_dir}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", token=token_value, private=private, exist_ok=True)
    result = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(local_path),
        token=token_value,
        commit_message=commit_message or "Upload synchronAI model artifacts",
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        create_pr=create_pr,
    )
    return str(result)


def download_model_dir(
    *,
    repo_id: str,
    local_dir: str,
    token: Optional[str] = None,
    token_env: Optional[str] = None,
    revision: Optional[str] = None,
    allow_patterns: Optional[Iterable[str]] = None,
    ignore_patterns: Optional[Iterable[str]] = None,
) -> str:
    """
    Download a HF Hub repo snapshot into a local directory.
    """
    _, snapshot_download = _safe_import_hf()
    token_value = _resolve_token(token, token_env)

    Path(local_dir).mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token_value,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
        ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
    )
    return path


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hugging Face Hub upload/download helpers.")
    parser.add_argument("--log-level", default="INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)
    upload = subparsers.add_parser("upload", help="Upload a directory to HF Hub.")
    upload.add_argument("--model", choices=sorted(MODEL_REPOS.keys()))
    upload.add_argument("--repo", help="Override repo id (e.g. user/model).")
    upload.add_argument("--path", required=True, help="Local directory to upload.")
    upload.add_argument("--token", default=None)
    upload.add_argument("--token-env", default=None)
    upload.add_argument("--private", action="store_true")
    upload.add_argument("--commit-message", default=None)
    upload.add_argument("--allow", nargs="*", default=None, help="Upload allow_patterns.")
    upload.add_argument("--ignore", nargs="*", default=None, help="Upload ignore_patterns.")
    upload.add_argument("--create-pr", action="store_true")

    download = subparsers.add_parser("download", help="Download a repo snapshot from HF Hub.")
    download.add_argument("--model", choices=sorted(MODEL_REPOS.keys()))
    download.add_argument("--repo", help="Override repo id (e.g. user/model).")
    download.add_argument("--path", required=True, help="Local directory for downloaded files.")
    download.add_argument("--revision", default=None)
    download.add_argument("--token", default=None)
    download.add_argument("--token-env", default=None)
    download.add_argument("--allow", nargs="*", default=None, help="Download allow_patterns.")
    download.add_argument("--ignore", nargs="*", default=None, help="Download ignore_patterns.")

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    logger = get_logger(__name__)

    repo_id = _resolve_repo_id(args.model, args.repo)
    if args.command == "upload":
        result = upload_model_dir(
            local_dir=args.path,
            repo_id=repo_id,
            token=args.token,
            token_env=args.token_env,
            commit_message=args.commit_message,
            private=args.private,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            create_pr=args.create_pr,
        )
        logger.info("Upload complete: %s", result)
        return

    if args.command == "download":
        path = download_model_dir(
            repo_id=repo_id,
            local_dir=args.path,
            token=args.token,
            token_env=args.token_env,
            revision=args.revision,
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
        )
        logger.info("Download complete: %s", path)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
