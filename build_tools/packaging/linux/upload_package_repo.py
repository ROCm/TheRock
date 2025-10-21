#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


"""Given ROCm artifacts directories, performs packaging to
create RPM and DEB packages and stores to OUTPUT folder

```
python ./build_tools/packaging/linux/upload_package_repo.py \
             --pkg-type deb \
             --s3-bucket therock-deb-rpm-test \
             --amdgpu-family gfx94X-dcgpu \
             --artifact-id 16418185899
```
"""

import os
import argparse
import subprocess
import boto3
import shutil
import datetime


# --- START: Index.html generation code ---

SVG_DEFS = """<svg xmlns="http://www.w3.org/2000/svg" style="display:none">
<defs>
  <symbol id="file" viewBox="0 0 265 323">
    <path fill="#4582ec" d="M213 115v167a41 41 0 01-41 41H69a41 41 0 01-41-41V39a39 39 0 0139-39h127a39 39 0 0139 39v76z"/>
    <path fill="#77a4ff" d="M176 17v88a19 19 0 0019 19h88"/>
    <path fill="#e3ecff" d="M126 90h89v25h-89zM81 120h164v25H81zM81 157h164v25H81zM81 195h164v25H81zM81 233h164v25H81z"/>
  </symbol>
  <symbol id="folder-shortcut" viewBox="0 0 265 216">
    <path fill="#4582ec" d="M18 54v-5a30 30 0 0130-30h75a28 28 0 0128 28v7h77a30 30 0 0130 30v84a30 30 0 01-30 30H33a30 30 0 01-30-30V54z"/>
    <path fill="#3a72e6" d="M127 43h-29a30 30 0 00-30 30v6H22a30 30 0 00-30 30v84a30 30 0 0030 30h215a30 30 0 0030-30V83a30 30 0 00-30-30h-78z"/>
  </symbol>
  <symbol id="go-up" viewBox="0 0 448 512">
    <path fill="#4582ec" d="M34.9 289.5L224 100.4l189.1 189.1c9.4 9.4 24.6 9.4 33.9 0l22.6-22.6c9.4-9.4 9.4-24.6 0-33.9L241 4.3c-9.4-9.4-24.6-9.4-33.9 0L12.3 233c-9.4 9.4-9.4 24.6 0 33.9l22.6 22.6c9.2 9.4 24.4 9.4 33.9-.1z"/>
  </symbol>
</defs>
</svg>
"""

HTML_TEMPLATE_HEAD = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{
    box-sizing: border-box;
  }}
  body {{
    background: #e3ecff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
      Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
    margin: 0;
    color: #222;
  }}
  header {{
    position: sticky;
    top: 0;
    background: white;
    border-bottom: 1px solid #eee;
    padding: 0.5rem 1rem;
  }}
  header h1 {{
    margin: 0;
  }}
  main {{
    max-width: 1024px;
    margin: 2rem auto;
  }}
  .listing table {{
    width: 100%;
    border-collapse: collapse;
  }}
  thead th {{
    text-align: left;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #eee;
  }}
  tbody tr {{
    border-bottom: 1px solid #eee;
  }}
  tbody tr.clickable:hover {{
    background-color: #cfe3ff;
  }}
  tbody tr.clickable > td:first-child {{
    cursor: pointer;
  }}
  td {{
    padding: 0.25rem 0.5rem 0.25rem 0.5rem;
    vertical-align: middle;
  }}
  .hideable {{
    display: none;
  }}
  a {{
    color: #008;
    text-decoration: none;
  }}
  a:hover {{
    text-decoration: underline;
  }}
  .file svg, .folder svg {{
    vertical-align: middle;
    margin-right: 0.5rem;
    fill: #4582ec;
  }}
  .name {{
    vertical-align: middle;
  }}
  .goup {{
    vertical-align: middle;
  }}
  @media (min-width: 640px) {{
    .hideable {{
      display: table-cell;
    }}
  }}
</style>
<title>artifacts</title>
</head>
<body>
{SVG_DEFS}
<header>
<h1>artifacts</h1>
</header>
<main>
<div class="listing">
<table aria-describedby="summary">
<thead>
<tr>
  <th></th>
  <th>Name</th>
  <th>Size</th>
  <th class="hideable">Modified</th>
  <th class="hideable"></th>
</tr>
</thead>
<tbody>
"""

HTML_TEMPLATE_FOOT = """
</tbody>
</table>
</div>
</main>
</body>
</html>
"""

def human_readable_size(size_bytes):
    if size_bytes == 0:
        return "0 bytes"
    units = ["bytes", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(units) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.0f} {units[i]}"

def generate_index_html(directory):
    entries = []
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.name.startswith('.'):
                    continue
                stat = entry.stat()
                entries.append({
                    'name': entry.name,
                    'is_dir': entry.is_dir(),
                    'size': stat.st_size if entry.is_file() else 0,
                    'mtime': datetime.datetime.fromtimestamp(stat.st_mtime),
                })
    except PermissionError:
        return 

    entries.sort(key=lambda e: (not e['is_dir'], e['name'].lower()))

    lines = []
    if os.path.abspath(directory) != os.path.abspath(os.sep):
        lines.append(f"""
<tr class="clickable">
  <td></td>
  <td><a href=".."><svg width="1.5em" height="1em" viewBox="0 0 448 512"><use xlink:href="#go-up"></use></svg><span class="goup">..</span></a></td>
  <td>&mdash;</td>
  <td class="hideable">&mdash;</td>
  <td class="hideable"></td>
</tr>
""")

    for e in entries:
        icon = "folder-shortcut" if e['is_dir'] else "file"
        size_disp = "&mdash;" if e['is_dir'] else human_readable_size(e['size'])
        mtime_str = e['mtime'].strftime("%a %b %d %H:%M:%S %Y")
        lines.append(f"""
<tr class="file clickable">
  <td></td>
  <td>
    <a href="{e['name']}">
      <svg width="1.5em" height="1em" viewBox="0 0 265 323"><use xlink:href="#{icon}"></use></svg>
      <span class="name">{e['name']}</span>
    </a>
  </td>
  <td data-order="{e['size']}">{size_disp}</td>
  <td class="hideable"><time datetime="{mtime_str}">{mtime_str}</time></td>
  <td class="hideable"></td>
</tr>""")
    html_content = HTML_TEMPLATE_HEAD + "".join(lines) + HTML_TEMPLATE_FOOT
    with open(os.path.join(directory, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_indexes_recursive(root_dir):
    for current_dir, dirs, files in os.walk(root_dir):
        generate_index_html(current_dir)

def run_command(cmd, cwd=None):
    """
    Function to execute commands in shell.
    """
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def find_package_dir():
    """
    Finds the default output dir for packages.
    Expects packages in ./output/packages
    """
    base_dir = os.path.join(os.getcwd(), "output", "packages")
    if not os.path.exists(base_dir):
        raise RuntimeError(f"Package directory not found: {base_dir}")
    print(f"Using package directory: {base_dir}")
    return base_dir


def create_deb_repo(package_dir, origin_name):
    """Function to create rpm repo
    It takes all the rpm files in the package_dir parameter
    And creates the deb package list using dpkg-scanpackages
    Package list is gzipped Packages.gz to pool/main foldre
    Also create Release meta package file needed for debian repo

    Parameters:
    package_dir : Folder to search for deb packages
    origin_name : S3 bucket to upload, used in meta data creation

    Returns: None
    """
    print("Creating APT repository...")
    dists_dir = os.path.join(package_dir, "dists", "stable", "main", "binary-amd64")
    release_dir = os.path.join(package_dir, "dists", "stable")
    pool_dir = os.path.join(package_dir, "pool", "main")

    os.makedirs(dists_dir, exist_ok=True)
    os.makedirs(pool_dir, exist_ok=True)
    for file in os.listdir(package_dir):
        if file.endswith(".deb"):
            shutil.move(os.path.join(package_dir, file), os.path.join(pool_dir, file))

    print(
        "Generating Packages file at repository root so 'Filename' paths are 'pool/...'." 
    )
    cmd = "dpkg-scanpackages -m pool/main /dev/null > dists/stable/main/binary-amd64/Packages"
    run_command(cmd, cwd=package_dir)
    run_command("gzip -9c Packages > Packages.gz", cwd=dists_dir)

    print("Creating Release file...")
    release_content = f"""\
Origin: {origin_name}
Label: {origin_name}
Suite: stable
Codename: stable
Version: 1.0
Date: {datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S UTC')}
Architectures: amd64
Components: main
Description: ROCm Repository
"""
    os.makedirs(release_dir, exist_ok=True)
    release_path = os.path.join(release_dir, "Release")
    with open(release_path, "w") as f:
        f.write(release_content)
    print(f"Wrote Release file to {release_path}")

    # Generate index.html recursively
    generate_index_html(package_dir)
    generate_indexes_recursive(package_dir)
    print(f"Generated index.html files recursively under {package_dir}")


def create_rpm_repo(package_dir):
    """Function to create rpm repo
    It takes all the rpm files in the package_dir parameter
    And creates the rpm repo using createrepo_c command inside x86_64 folder

    Parameters:
    package_dir : Folder to search for rpm packages

    Returns: None
    """
    print("Creating YUM/DNF repository...")

    arch_dir = os.path.join(package_dir, "x86_64")
    os.makedirs(arch_dir, exist_ok=True)
    for file in os.listdir(package_dir):
        if file.endswith(".rpm"):
            shutil.move(os.path.join(package_dir, file), os.path.join(arch_dir, file))
    run_command("createrepo_c .", cwd=arch_dir)
    print(f"Generated repodata/ in {arch_dir}")

    # Generate index.html recursively
    generate_index_html(package_dir)
    generate_indexes_recursive(package_dir)
    print(f"Generated index.html files recursively under {package_dir}")


def upload_to_s3(source_dir, bucket, prefix):
    """Function to upload the packges and repo files to the s3 bucket
    It upload the source_dir contents to s3://{bucket}/{prefix}/

    Parameters:
    source_dir : Folder with the packages and repo files
    bucket : S3 bucket
    prefix : S3 prefix

    Returns: None
    """
    s3 = boto3.client("s3")
    print(f"Uploading to s3://{bucket}/{prefix}/")

    for root, _, files in os.walk(source_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, source_dir)
            s3_key = os.path.join(prefix, rel_path).replace("\\", "/")
            print(f"Uploading: {local_path} → s3://{bucket}/{s3_key}")
            extra_args = {}
            if filename.endswith(".html"):
                extra_args['ContentType'] = 'text/html'
            s3.upload_file(local_path, bucket, s3_key, ExtraArgs=extra_args if extra_args else None)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkg-type",
        required=True,
        choices=["deb", "rpm"],
        help="Type of packages to process",
    )
    parser.add_argument("--s3-bucket", required=True, help="Target S3 bucket name")
    parser.add_argument(
        "--amdgpu-family", required=True, help="AMDGPU family identifier (e.g., gfx94X)"
    )
    parser.add_argument(
        "--artifact-id", required=True, help="Unique artifact ID or version tag"
    )
    args = parser.parse_args()

    package_dir = find_package_dir()
    s3_prefix = f"{args.amdgpu_family}_{args.artifact_id}/{args.pkg_type}"

    if args.pkg_type == "deb":
        create_deb_repo(package_dir, args.s3_bucket)
    else:
        create_rpm_repo(package_dir)

    upload_to_s3(package_dir, args.s3_bucket, s3_prefix)


if __name__ == "__main__":
    main()
