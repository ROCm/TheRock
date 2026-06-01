with section("parse"):
    additional_commands = {
        "rocm_create_package": {
            "kwargs": {"NAME": 1, "DESCRIPTION": 1, "MAINTAINER": 1, "LDCONFIG_DIR": 1},
            "flags": ["LDCONFIG", "HEADER_ONLY"],
        }
    }

with section("format"):
    line_width = 100
    tab_size = 4
    dangle_parens = True
    command_case = "lower"
    enable_sort = True
    max_subgroups_hwrap = 4
    max_pargs_hwrap = 6
    max_lines_hwrap = 2

with section("markup"):
    first_comment_is_literal = True

with section("lint"):
    disabled_codes = ["C0103", "C0301", "W0106"]
