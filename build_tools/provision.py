#!/usr/bin/env python

import argparse
import sys
import os
import shutil
from fetch_artifacts import retrieve_base_artifacts, retrieve_enabled_artifacts, s3_bucket_exists
import urllib.request

def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def url_exists(url):
    try:
        response = urllib.request.urlopen(url)
        return response.status == 200
    except urllib.error.URLError:
        return False
    
def create_output_directory(args):
    output_dir_path = args.output_dir
    log(f"Creating directory {output_dir_path}")
    if os.path.isdir(output_dir_path):
        log(f"Directory {output_dir_path} already exists, removing existing directory and files")
        shutil.rmtree(output_dir_path)
    os.mkdir(output_dir_path)
    log(f"Created directory {output_dir_path}")
    
    
def retrieve_artifacts_by_ci(args):
    runner_id = args.runner_id
    output_dir = args.output_dir
    if not s3_bucket_exists(runner_id):
        print(f"S3 artifacts for {runner_id} does not exist. Exiting...")
        return
    
    args.all = True
    args.blas = False
    print(f"Retrieving artifacts by runner ID {runner_id}")
    retrieve_base_artifacts(args, runner_id, output_dir)
    

def retrieve_artifacts_by_release(args):
    release_id = args.release_id
    print(f"Retrieving artifacts by release ID {release_id}")
    
    
def run(args):
    create_output_directory(args)
    if args.runner_id:
        retrieve_artifacts_by_ci(args)
    
    if args.release_id:
        retrieve_artifacts_by_release(args)
    
        
def main(argv):
    parser = argparse.ArgumentParser(prog="provision")
    parser.add_argument(
        "--output-dir", type=str, default="./build", help="Path of the output directory for TheRock"
    )
    
    parser.add_argument(
        "--amdgpu-family", type=str, default="gfx94X-dcgpu", help="AMD GPU family to provision (please refer to this: https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake#L44-L81 for target choices)"
    )

    # This mutually exclusive group will ensure that only one argument is present
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--runner-id", type=str, help="GitHub runner ID of TheRock to provision"
    )
    
    group.add_argument(
        "--release-id", type=str, help="Github release ID of TheRock to provision"
    )
    
    
    
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main(sys.argv[1:])
