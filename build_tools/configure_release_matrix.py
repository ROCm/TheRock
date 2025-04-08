import os
import json
from configure_ci import set_github_output, amdgpu_family_info_matrix

# This file helps generate a release test matrix for test_release_packages.yml


def main(args):
    assets = json.loads(args.get("asset_files")).get("assets", [])
    today_date = args.get("today_date")
    release_matrix = []
    for asset in assets:
        asset_name = asset.get("name", "")
        # Test only today's nightly release packages
        # for now, we can only run tests on gfx94X since we only have a linux gfx94X test machine
        if today_date in asset_name and "gfx94X" in asset_name:
            target_info = amdgpu_family_info_matrix.get("gfx94X").get("linux")
            target_info["file_name"] = asset_name
            release_matrix.append(target_info)

    set_github_output({"release_matrix": json.dumps(release_matrix)})


if __name__ == "__main__":
    args = {}
    args["asset_files"] = os.environ.get("ASSET_FILES", "[]")
    args["today_date"] = os.environ.get("TODAY_DATE", "")
    main(args)
