import argparse
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

logging.basicConfig(level=logging.INFO)

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent
GENERIC_VARIANT = "generic"
PLATFORM = platform.system().lower()


def set_github_step_summary(summary: str):
    logging.info(f"Appending to github summary: {summary}")
    step_summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
    with open(step_summary_file, "a") as f:
        f.write(summary + "\n")


def create_index_file(args: argparse.Namespace):
    logging.info("Creating index file")
    index_file_path = THEROCK_DIR / "third-party" / "indexer" / "indexer.py"
    build_dir = args.build_dir

    subprocess.run(
        [sys.executable, index_file_path, "-f", "*.tar.xz*", build_dir / "artifacts"]
    )


def create_log_index(args: argparse.Namespace):
    logging.info("Creating log index file")
    create_log_index_path = THEROCK_DIR / "build_tools" / "create_log_index.py"
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    subprocess.run(
        [
            sys.executable,
            create_log_index_path,
            f"--build-dir={str(build_dir)}",
            f"--amdgpu-family={amdgpu_family}",
        ]
    )


def upload_artifacts(args: argparse.Namespace, bucket_uri: str):
    logging.info("Uploading artifacts to S3")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family

    # Uploading artifacts to S3 bucket
    cmd = [
        "aws",
        "s3",
        "cp",
        build_dir / "artifacts",
        bucket_uri,
        "--recursive",
        "--no-follow-symlinks",
        "--exclude",
        "*",
        "--include",
        "*.tar.xz*",
    ]
    logging.info(f"Executing cmd {shlex.join(cmd)}")
    subprocess.run(cmd)

    # Uploading index.html to S3 bucket
    cmd = [
        "aws",
        "s3",
        "cp",
        build_dir / "artifacts" / "index.html",
        f"{bucket_uri}/index-{amdgpu_family}.html",
    ]
    logging.info(f"Executing cmd {shlex.join(cmd)}")
    subprocess.run(cmd)


def upload_logs(args: argparse.Namespace, bucket: str, bucket_uri: str):
    logging.info(f"Uploading logs to S3 for bucket {bucket}")
    build_dir = args.build_dir
    amdgpu_family = args.amdgpu_family
    upload_logs_s3_path = THEROCK_DIR / "build_tools" / "upload_logs_to_s3.py"
    s3_base_path = f"{bucket_uri}/logs/{amdgpu_family}"

    subprocess.run(
        [
            sys.executable,
            upload_logs_s3_path,
            f"--build-dir={build_dir}",
            f"--s3-base-path={s3_base_path}",
        ]
    )


def add_links_to_job_summary(args: argparse.Namespace, bucket: str, bucket_url: str):
    logging.info(f"Adding links to job summary to bucket {bucket}")
    build_dir = args.build_dir
    run_id = args.run_id
    amdgpu_family = args.amdgpu_family

    log_url = f"{bucket_url}/logs/{amdgpu_family}/index.html"
    set_github_step_summary(f"[Build Logs]({log_url})")
    if os.path.exists(build_dir / "artifacts" / "index.html"):
        artifact_url = f"{bucket_url}/index-{amdgpu_family}.html"
        set_github_step_summary(f"[Artifacts]({artifact_url})")
    else:
        logging.info("No artifacts index found. Skipping artifact link.")


def run(args: argparse.Namespace):
    repo = args.repo
    owner, repo_name = repo.split("/")
    run_id = args.run_id
    bucket = (
        "therock-artifacts"
        if repo_name == "TheRock" and owner == "ROCm"
        else "therock-artifacts-external"
    )
    # For external repos, we add an extra folder in the bucket because GitHub run IDs are unique per repo.
    external_repo_path = (
        "" if repo_name == "TheRock" and owner == "ROCm" else f"{owner}-{repo_name}/"
    )
    bucket_uri = f"s3://{bucket}/{external_repo_path}{run_id}-{PLATFORM}"
    bucket_url = f"https://{bucket}.s3.us-east-2.amazonaws.com/{external_repo_path}{run_id}-{PLATFORM}"

    create_index_file(args)
    create_log_index(args)
    upload_artifacts(args, bucket_uri)
    upload_logs(args, bucket, bucket_uri)
    add_links_to_job_summary(args, bucket, bucket_url)


def main(argv):
    parser = argparse.ArgumentParser(prog="artifact_upload")
    parser.add_argument("--repo", type=str, required=True)

    parser.add_argument("--run-id", type=str, required=True)

    parser.add_argument("--amdgpu-family", type=str, required=True)

    parser.add_argument("--build-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
