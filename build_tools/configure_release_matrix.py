import os
import json
from configure_ci import set_github_output

# This file helps generate a release test matrix for test_release_packages.yml


def main(args):
    assets = json.loads(args.get("asset_files"))
    release_matrix = []
    for asset in assets:
        asset_name = asset.get("name", "")
        # for now, we can only run tests on gfx94X, so adding hard-coded value
        if "gfx94X" in asset_name:
            release_matrix.append(asset_name)

    set_github_output({"release_matrix": json.dumps(release_matrix)})


if __name__ == "__main__":
    args = {}
    args["asset_files"] = os.environ.get("ASSET_FILES", "[]")
    main(args)
