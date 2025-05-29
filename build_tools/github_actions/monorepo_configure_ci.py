"""
This script determines which build flag and tests to run based on SUBTREES

Required environment variables:
  - SUBTREES
"""

from configure_ci import set_github_output
import json
from monorepo_map import monorepo_map
import os

SUBTREES = os.getenv("SUBTREES", "")


def run():
    subtrees = SUBTREES.split("\n")
    job = None
    for subtree in subtrees:
        if subtree in monorepo_map:
            job = monorepo_map.get(subtree)
            break
    set_github_output({"job": json.dumps(job)})


if __name__ == "__main__":
    run()
