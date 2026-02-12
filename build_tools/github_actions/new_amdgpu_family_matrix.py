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
              "runs_on": {                      #         dict: Host names of compute nodes
                  "test":                       #             string: test runner (optional)
                  "test-multi-gpu":             #             string: multi-gpu test runner (optional)
                  "benchmark":                  #             string: benchmark runner (optional)
              }
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
              "runs_on": {                      #         dict: Host names of compute nodes
                  "test":                       #             string: test runner (optional)
                  "test-multi-gpu":             #             string: multi-gpu test runner (optional)
                  "benchmark":                  #             string: benchmark runner (optional)
              }
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

##########################################################################################
# NOTE: when doing changes here, also check that they are done in amdgpu_family_matrix.py
##########################################################################################

amdgpu_family_predefined_groups = {
    # The 'presubmit' matrix runs on 'pull_request' triggers (on all PRs).
    "amdgpu_presubmit": ["gfx94X-dcgpu", "gfx110X-all", "gfx1151", "gfx120X-all"],
    # The 'postsubmit' matrix runs on 'push' triggers (for every commit to the default branch).
    "amdgpu_postsubmit": ["gfx950-dcgpu"],
    # The 'nightly' matrix runs on 'schedule' triggers.
    "amdgpu_nightly": [
        "gfx90X-dcgpu",
        "gfx101X-dgpu",
        "gfx103X-dgpu",
        "gfx1150",
        "gfx1152",
        "gfx1153",
    ],
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
                    "runs_on": {
                        "test": "linux-mi325-1gpu-ossci-rocm-frac",
                        "test-multi-gpu": "linux-mi325-8gpu-ossci-rocm",
                        # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
                        "benchmark": "linux-mi325-8gpu-ossci-rocm",
                    },
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
                    "runs_on": {},
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
                    # TODO(#2740): Re-enable machine once `amdsmi` test is fixed
                    "run_tests": False,
                    "runs_on": {
                        "test": "linux-gfx110X-gpu-rocm",
                    },
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
                    "runs_on": {
                        "test": "windows-gfx110X-gpu-rocm",
                    },
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
            },
        }
    },
    "gfx115X": {
        "gfx1150": {
            "linux": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    # TODO(#3199): Re-enable machine once it is stable
                    "run_tests": False,
                    "runs_on": {
                        "test": "linux-gfx1150-gpu-rocm",
                    },
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
                    "runs_on": {},
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
                    "runs_on": {
                        "test": "linux-gfx1151-gpu-rocm",
                        "oem": "linux-strix-halo-gpu-rocm-oem",
                    },
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
                    "runs_on": {
                        "test": "windows-gfx1151-gpu-rocm",
                        # TODO(#2754): Add new benchmark-runs-on runner for benchmarks
                        "benchmark": "windows-gfx1151-gpu-rocm",
                    },
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
                    "runs_on": {},
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
                    "runs_on": {},
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
                    # TODO(#2682): Re-enable machine once it is stable
                    "run_tests": False,
                    "runs_on": {
                        "test": "linux-gfx1153-gpu-rocm",
                    },
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
                    "runs_on": {},
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
                    "runs_on": {"test": "linux-mi355-1gpu-ossci-rocm"},
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
                    "runs_on": {},
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
                    "runs_on": {
                        "test": "linux-gfx120X-gpu-rocm",
                    },
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
                    # TODO(#2962): Re-enable machine once sanity checks work with this architecture
                    "run_tests": False,
                    "runs_on": {
                        "test": "windows-gfx120X-gpu-rocm",
                    },
                },
                "release": {
                    "push_on_success": True,
                    "bypass_tests_for_releases": True,
                },
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
                    "run_tests": True,
                    "runs_on": {
                        "test": "linux-gfx90X-gpu-rocm",
                    },
                    "sanity_check_only_for_family": True,
                },
                "release": {
                    "push_on_success": False,
                    "bypass_tests_for_releases": False,
                },
            },
            # TODO(#1927): Resolve error generating file `torch_hip_generated_int4mm.hip.obj`,
            # to enable PyTorch builds
            "windows": {
                "build": {
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": {},
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
                    "build_variants": ["release"],
                },
                "test": {
                    "run_tests": False,
                    "runs_on": {},
                    # TODO(#1926): Resolve bgemm kernel hip file generation error,
                    # to enable PyTorch builds
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
                    "runs_on": {},
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
                    # TODO(#2740): Re-enable machine once `amdsmi` test is fixed
                    "run_tests": False,
                    "runs_on": {
                        "test": "linux-gfx1030-gpu-rocm",
                    },
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
                    # TODO(#3200): Re-enable machine once it is stable
                    "run_tests": False,
                    "runs_on": {
                        "test": "windows-gfx1030-gpu-rocm",
                    },
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
