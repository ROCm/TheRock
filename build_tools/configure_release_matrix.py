import os
import json
from configure_ci import set_github_output, amdgpu_family_info_matrix
from datetime import datetime

# This file helps generate a release test matrix for test_release_packages.yml

TODAY_DATE = datetime.today().strftime("%Y%m%d")


def main(args):
    assets = json.loads(args.get("asset_files")).get("assets", [])
    release_matrix = []
    for asset in assets:
        asset_name = asset.get("name", "")
        # Test only today's nightly release packages
        # for now, we can only run tests on gfx94X since we only have a linux gfx94X test machine
        if TODAY_DATE in asset_name and "gfx94X" in asset_name:
            target_info = amdgpu_family_info_matrix.get("gfx94X").get("linux")
            target_info["file_name"] = asset_name
            release_matrix.append(target_info)

    set_github_output(
        {
            "release_matrix": json.dumps(release_matrix),
            "today_date": json.dumps(TODAY_DATE),
        }
    )


if __name__ == "__main__":
    args = {}
    args["asset_files"] = os.environ.get("ASSET_FILES", "[]")
    main(args)
