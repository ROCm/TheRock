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
              "build_variants": []              #         list: build variant names (e.g., ["release", "asan"])
            },                                  # platform <optional>
           "test": {                            #     test options
              "run_tests":                      #         boolean: True if the test should run
              "runs_on":                        #         string: Host name of the compute node where the test should run on
              "benchmark_runs_on":              #         string: Host name of the compute node for benchmarks (optional)
            }
            "release": {                        #     release options
               "push_on_success":               #         boolean: True if the release should be performed
               "bypass_tests_for_releases":     #         boolean: True if tests should be skipped for the release
            }
        }
        "windows": {
            "build": {
              "expect_failure":                 #         boolean:
              "build_variants": []              #         list: build variant names
            },                                  # platform <optional>
            "test": {                           #     test options
              "run_tests":                      #         boolean: True if the test should run
              "runs_on":                        #         string: Host name of the compute node where the test should run on
              "benchmark_runs_on":              #         string: Host name of the compute node for benchmarks (optional)
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
#                   "build_variants": ["release"],
#                 },
#                 "test": {
#                     "run_tests": False,
#                     "runs_on": "",
#                     "sanity_check_only_for_family": False,
#                     "expect_pytorch_failure": False,
#                 },
#                 "release": {
#                     "push_on_success": False,
#                     "bypass_tests_for_releases": False,
#                 }
#             },
#             "windows": {
#                 "build": {
#                     "expect_failure": False,
#                     "build_variants": ["release"],
#                 },
#                 "test": {
#                     "run_tests": False,
#                     "runs_on": "",
#                     "benchmark_runs_on": "",
#                     "sanity_check_only_for_family": False,
#                     "expect_pytorch_failure": False,
#                 },
#                 "release": {
#                     "push_on_success": False,
#                     "bypass_tests_for_releases": False
#                 }
#             },
# }

amdgpu_family_predefined_groups = {
    # The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
    "amdgpu_presubmit": ["gfx94X-dcgpu", "gfx110X-all", "gfx1151"],
    # The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
    "amdgpu_postsubmit": ["gfx950-dcgpu", "gfx120X-all"],
    # The 'nightly' matrix runs on 'schedule' triggers.
    "amdgpu_nightly_ci": ["gfx90X-dcgpu", "gfx101X-dgpu", "gfx103X-dgpu"],
}

all_build_variants = {
    "linux": {
        "release": {
            "build_variant_label": "release",
            "build_variant_suffix": "",
            # TODO: Enable linux-release-package once capacity and rccl link
            # issues are resolved. https://github.com/ROCm/TheRock/issues/1781
            # "build_variant_cmake_preset": "linux-release-package",
            "build_variant_cmake_preset": "",
        },
        "asan": {
            "build_variant_label": "asan",
            "build_variant_suffix": "asan",
            "build_variant_cmake_preset": "linux-release-asan",
            "expect_failure": True,
        },
    },
    "windows": {
        "release": {
            "build_variant_label": "release",
            "build_variant_suffix": "",
            "build_variant_cmake_preset": "windows-release",
        },
    },
}


amdgpu_family_info_matrix_all = {
    "gfx94X": {
        "dcgpu": {
            "linux": {
                "build": {
                    "build_variants": ["release", "asan"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "linux-mi325-1gpu-ossci-rocm-frac",
                    # TODO: Add new benchmark_runs_on runner for benchmarks
                    "benchmark_runs_on": "linux-mi325-1gpu-ossci-rocm-frac",
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        },
    },
    "gfx110X": {
        "all": {
            "linux": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "linux-gfx110X-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "windows-gfx110X-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {"push_on_success": True, "bypass_tests_for_releases": True},
            },
        }
    },
    "gfx115x": {
        "gfx1150": {
            "linux": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    # TODO(#2614): Re-enable machine once it is stable
                    "run_tests": False,
                    "runs_on": "linux-gfx1150-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        },
        "gfx1151": {
            "linux": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "linux-gfx1151-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "windows-gfx1151-gpu-rocm",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        },
        "gfx1152": {
            "linux": {
                "build": {
                    "expect_failure": True,
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "expect_failure": True,
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        },
        "gfx1153": {
            "linux": {
                "build": {
                    "expect_failure": True,
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "linux-gfx1153-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "expect_failure": True,
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        },
    },
    "gfx950": {
        "dcgpu": {
            "linux": {
                "build": {
                    "build_variants": ["release", "asan"],
                },
                "test": {
                    "run_tests": True,
                    # Networking issue: https://github.com/ROCm/TheRock/issues/1660
                    # Label is "linux-mi355-1gpu-ossci-rocm"
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
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
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "linux-gfx1201-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                },
                "release": {"push_on_success": True, "bypass_tests_for_releases": True},
            },
        }
    },
    "gfx90X": {
        "dcgpu": {
            "linux": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                    "expect_pytorch_failure": True,
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
                    "expect_failure": True,
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                    "expect_pytorch_failure": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": "",
                    "benchmark_runs_on": "",
                    "expect_pytorch_failure": True,
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
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "linux-gfx1030-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": True,
                    "runs_on": "windows-gfx1030-gpu-rocm",
                    "benchmark_runs_on": "",
                    "sanity_check_only_for_family": True,
                    "expect_pytorch_failure": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
        }
    },
}
