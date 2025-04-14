#!/usr/bin/env python

import argparse
import sys
import os
import shutil

def log(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()
    
def create_output_directory(args):
    output_dir_path = args.output_dir
    log(f"Creating directory {output_dir_path}")
    if os.path.isdir(output_dir_path):
        log(f"Directory {output_dir_path} already exists, removing existing directory and files")
        shutil.rmtree(output_dir_path)
    os.mkdir(output_dir_path)
    log(f"Created directory {output_dir_path}")
        
    
def run(args):
    create_output_directory(args)
    
    
def main(argv):
    parser = argparse.ArgumentParser(prog="provision")
    parser.add_argument(
        "--output-dir", type=str, default="./build", help="Path of the output directory for TheRock"
    )

    parser.add_argument(
        "--provision-id", type=str, default="nightly-release", help="Release or runner ID of TheRock to provision"
    )
    
    parser.add_argument(
        "--amdgpu-family", type=str, default="gfx94X", help="AMD GPU family to provision"
    )
    
    args = parser.parse_args(argv)
    run(args)

if __name__ == "__main__":
    main(sys.argv[1:])
