"""
This AMD GPU Family Matrix is the "source of truth" for GitHub workflows.

* Each entry determines which families and test runners are available to use
* Each group determines which entries run by default on workflow triggers

Data layout

amdgpu_family_info_matrix_all {
  <gpufamily-target>: {                         # string: cmake target for entire gpu family
     <target>: {                                # string: cmake target for single gpu architecture
        "linux": {
            "build": {
              "expect_failure":                 #         boolean:
            },                                  # platform <optional>
           "test": {                            #     test options
              "run_tests":                      #         boolean: True if the test should run
              "runs-on":                        #         string: Host name of the compute node where the test should run on
            }
            "release": {                        #     release options
               "push_on_success":               #         boolean: True if the release should be performed
               "bypass_tests_for_releases":     #         boolean: True if tests should be skipped for the release
            }
        }
        "windows": {
            "build": {
              "expect_failure":                 #         boolean:
            },                                  # platform <optional>
            "test": {                           #     test options
              "run_tests":                      #         boolean: True if the test should run
              "runs-on":                        #         string: Host name of the compute node where the test should run on
            }
            "release": {                        #     release options
               "push_on_success":               #         boolean: True if the release should be performed
               "bypass_tests_for_releases":     #         boolean: True if tests should be skipped for the release
            }
        }
    }
}

Generic targets of a family are "all", "dcgpu", "dgpu", ...
Cmake targets are defined in: cmake/therock_amdgpu_targets.cmake
"""

# blueprint = {
# "linux": {
#                 "build": {
#                   "expect_failure": False,
#                 },
#                 "test": {
#                     "run_tests": False,
#                     "runs-on": "",
#                 },
#                 "release": {
#                     "push_on_success": False,
#                     "bypass_tests_for_releases": False,
#                 }
#             },
#             "windows": {
#                 "build": {
#                     "expect_failure": False,
#                 },
#                 "test": {
#                     "run_tests": False,
#                     "runs-on": "",
#                 },
#                 "release": {
#                     "push_on_success": False,
#                     "bypass_tests_for_releases": False
#                 }
#             },
# }

amdgpu_family_predefined_groups = {
    # The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
    "amdgpu_presubmit": ["gfx94X-dcgpu", "gfx110X-dgpu", "gfx1151"],
    # The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
    "amdgpu_postsubmit": ["gfx950-dcgpu", "gfx120X-all"],
    # The 'nightly' matrix runs on 'schedule' triggers.
    "amdgpu_nightly_ci": ["gfx90X-dcgpu", "gfx101X-dgpu", "gfx103X-dgpu"],
}


amdgpu_family_info_matrix_all = {
    "gfx94X": {
        "dcgpu": {
            "linux": {
                "build": {},
                "test": {
                    "run_tests": True,
                    "runs-on": "linux-mi325-1gpu-ossci-rocm",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {"push_on_success": False, "bypass_tests_for_releases": ""},
            },
        },
    },
    "gfx110X": {
        "dgpu": {
            "linux": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
            "windows": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {"push_on_success": True, "bypass_tests_for_releases": True},
            },
        }
    },
    "gfx115x": {
        "gfx1151": {
            "linux": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
            "windows": {
                "build": {},
                "test": {
                    "run_tests": True,
                    "runs-on": "windows-strix-halo-gpu-rocm",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        }
    },
    "gfx950": {
        "dcgpu": {
            "linux": {
                "build": {},
                "test": {
                    "run_tests": True,
                    "runs-on": "linux-mi355-1gpu-ossci-rocm",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        }
    },
    "gfx120X": {
        "all": {
            "linux": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
            "windows": {
                "build": {},
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {"push_on_success": True, "bypass_tests_for_releases": True},
            },
        }
    },
    "gfx90X": {
        "dcgpu": {
            "linux": {
                "build": {
                    "expect_failure": False,
                },
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "expect_failure": False,
                },
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        }
    },
    "gfx101X": {
        "dgpu": {
            "linux": {
                "build": {
                    "expect_failure": False,
                },
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "expect_failure": False,
                },
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        }
    },
    "gfx103X": {
        "dgpu": {
            "linux": {
                "build": {
                    "expect_failure": True,
                },
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "expect_failure": True,
                },
                "test": {
                    "run_tests": False,
                    "runs-on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        }
    },
}
