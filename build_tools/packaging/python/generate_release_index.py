#!/usr/bin/env python
"""Creates an HTML page for releases for `pip install --find-links` from a subdirectory in an S3 bucket.

Sample usage:

    ```bash
    ./build_tools/packaging/python/generate_release_index.py \
        --bucket=therock-dev-python \
        --endpoint=s3.us-east-2.amazonaws.com \
        --subdir=gfx110X-dgpu \
        --output=index.html
    ```

S3 buckets in `us-east-2` currently used are:

    * `therock-dev-python`
    * `therock-nightly-python`
"""

import argparse
import boto3
import html
import io
import sys
import textwrap
from urllib.parse import quote


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--bucket",
        required=True,
        help="S3 bucket in which the subdirectory is indexed",
    )
    p.add_argument(
        "--endpoint",
        default="s3.us-east-2.amazonaws.com",
        help="S3 endpoint used to form the URL",
    )
    p.add_argument(
        "--subdir",
        "--subdirectory",
        required=True,
        help="Subdirectory in S3 bucket which is indexed",
    )
    p.add_argument(
        "--output",
        default="-",
        help="The file to write the HTML to or '-' for stdout (the default)",
    )
    return p.parse_args()


def get_objects(bucket_name: str, subdir: str):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)

    all_objects = bucket.objects.filter(Prefix=subdir)

    objects = [obj.key.split(subdir + "/")[1] for obj in all_objects]
    if "index.html" in objects:
        objects.remove("index.html")

    return objects


def add_releases(objects: list, base_url: str, subdir: str, file: io.TextIOWrapper):

    file.write(
        f'    <h2> <a href="https://github.com/ROCm/TheRock">ROCm/TheRock</a> {subdir} packages</h2>\n'
    )

    for obj in objects:
        encoded_url = f"{base_url}/{quote(obj)}"  # URL-encode the object key
        name = html.escape(obj)
        file.write(f'    <a href="{encoded_url}">{name}</a><br>\n')

    file.write(
        "    <br><hr>\n"
        f'    This file was auto-generated with <a href="https://github.com/ROCm/TheRock/blob/main/build_tools/packaging/python/generate_release_index.py">generate_release_index.py</a>.\n'
    )


def main(args):
    objects = get_objects(args.bucket, args.subdir)
    url = f"https://{args.bucket}.{args.endpoint}/{args.subdir}"

    with sys.stdout if args.output == "-" else open(args.output, "w") as f:
        f.write(
            textwrap.dedent(
                """\
            <!DOCTYPE html>
            <html>
              <head>
                <meta charset="utf-8">
                <style>
                  * { padding: 0; margin: 10; }
                  body {
                      font-family: sans-serif;
                      text-rendering: optimizespeed;
                      background-color: #ffffff;
                  }
                  a {
                      color: #006ed3;
                      text-decoration: none;
                  }
                  a:hover {
                      color: #319cff;
                      text-decoration: underline;
                  }
                </style>
              </head>

              <body>
            """
            )
        )
        add_releases(objects, url, args.subdir, f)
        f.write(
            textwrap.dedent(
                """\
      </body>
    </html>
    """
            )
        )


if __name__ == "__main__":
    main(parse_arguments())
