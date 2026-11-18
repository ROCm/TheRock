# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test subproject configuration, dependency discovery, and staged builds."""

import unittest

from therock_fingerprint_test_utils import (
    FingerprintTestCase,
    cmake_project_setup,
    write,
)


class SubprojectBuildTest(FingerprintTestCase):
    required_tools = ("cmake", "ninja")

    def setUp(self) -> None:
        super().setUp()
        self.source = self.root / "source with spaces"
        self.build = self.root / "build with spaces"
        self.provider_build = self.build / "provider" / "build"
        self.provider_stage = self.build / "provider" / "stage"
        self.provider_dist = self.build / "provider" / "dist"
        write(
            self.source / "CMakeLists.txt",
            f"""
            {cmake_project_setup("subproject_baseline")}
            set(PROVIDER_INSTALL_DESTINATION "" CACHE STRING
              "Provider install subdirectory")
            option(EXCLUDE_CONSUMER "Exclude consumer from the default build" OFF)
            set(_consumer_options)
            if(EXCLUDE_CONSUMER)
              list(APPEND _consumer_options EXCLUDE_FROM_ALL)
            endif()
            therock_cmake_subproject_declare(provider
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/provider"
              BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/provider"
              INSTALL_DESTINATION "${{PROVIDER_INSTALL_DESTINATION}}"
              DISABLE_AMDGPU_TARGETS
              NO_MERGE_COMPILE_COMMANDS
            )
            therock_cmake_subproject_provide_package(
              provider BaselineFixture lib/cmake/BaselineFixture)
            therock_cmake_subproject_activate(provider)

            therock_cmake_subproject_declare(consumer
              EXTERNAL_SOURCE_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/consumer"
              BINARY_DIR "${{CMAKE_CURRENT_BINARY_DIR}}/consumer"
              BUILD_DEPS provider
              DISABLE_AMDGPU_TARGETS
              NO_MERGE_COMPILE_COMMANDS
              ${{_consumer_options}}
            )
            therock_cmake_subproject_activate(consumer)
            """,
        )
        write(
            self.source / "provider" / "CMakeLists.txt",
            r"""
            cmake_minimum_required(VERSION 3.25)
            project(provider NONE)
            file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/events.txt" "configure\n")
            # Deliberately runs whenever the inner build is invoked. This lets
            # the test detect redundant outer build steps without timestamps.
            add_custom_target(payload ALL
              COMMAND "${CMAKE_COMMAND}"
                "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/build_payload.cmake"
              VERBATIM
            )
            install(FILES "${CMAKE_CURRENT_BINARY_DIR}/payload.txt"
              DESTINATION share)
            install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/BaselineFixtureConfig.cmake"
              DESTINATION lib/cmake/BaselineFixture)
            install(CODE
              "file(APPEND \"${CMAKE_CURRENT_BINARY_DIR}/events.txt\" \"install\n\")")
            """,
        )
        write(
            self.source / "provider" / "build_payload.cmake",
            r"""
            file(APPEND "${OUTPUT_DIR}/events.txt" "build\n")
            file(WRITE "${OUTPUT_DIR}/payload.txt" "provider payload\n")
            """,
        )
        write(
            self.source / "provider" / "BaselineFixtureConfig.cmake",
            r"""
            file(READ "${CMAKE_CURRENT_LIST_DIR}/../../../share/payload.txt"
              BaselineFixture_PAYLOAD)
            """,
        )
        write(
            self.source / "consumer" / "CMakeLists.txt",
            r"""
            cmake_minimum_required(VERSION 3.25)
            project(consumer NONE)
            find_package(BaselineFixture CONFIG REQUIRED)
            file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/dependency.txt"
              "${BaselineFixture_PAYLOAD}")
            install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dependency.txt"
              DESTINATION share)
            """,
        )

    def configure(self, *cmake_args: str) -> None:
        self.run_command(
            "cmake",
            "-S",
            self.source,
            "-B",
            self.build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTHEROCK_HARDENING_POLICY=OFF",
            *cmake_args,
        )

    def build_target(self, target: str = "all") -> None:
        self.run_command("cmake", "--build", self.build, "--target", target)

    def events(self) -> list[str]:
        return (
            (self.provider_build / "events.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )

    def test_configure_build_and_stage_run_in_order(self) -> None:
        self.configure()
        self.assertFalse((self.provider_build / "events.txt").exists())

        self.build_target("provider+configure")
        self.assertEqual(self.events(), ["configure"])
        self.assertFalse((self.provider_build / "payload.txt").exists())

        self.build_target("provider+build")
        self.assertEqual(self.events(), ["configure", "build"])
        self.assertEqual(
            (self.provider_build / "payload.txt").read_text(encoding="utf-8"),
            "provider payload\n",
        )
        self.assertFalse((self.provider_stage / "share/payload.txt").exists())

        self.build_target("provider+stage")
        self.assertEqual(self.events(), ["configure", "build", "install"])
        for directory in (self.provider_stage, self.provider_dist):
            with self.subTest(directory=directory):
                self.assertEqual(
                    (directory / "share/payload.txt").read_text(encoding="utf-8"),
                    "provider payload\n",
                )

    def test_build_dependency_package_is_available_during_configuration(self) -> None:
        self.configure()
        self.assertFalse((self.provider_dist / "share/payload.txt").exists())
        # Request only the consumer: TheRock must stage its provider first and
        # make the declared package discoverable to find_package().
        self.build_target("consumer+configure")
        self.assertEqual(self.events(), ["configure", "build", "install"])
        self.assertEqual(
            (self.build / "consumer/build/dependency.txt").read_text(encoding="utf-8"),
            "provider payload\n",
        )

    def test_duplicate_subproject_declaration_is_rejected(self) -> None:
        with (self.source / "CMakeLists.txt").open("a", encoding="utf-8") as cmake_file:
            cmake_file.write("\ntherock_cmake_subproject_declare(provider)\n")

        with self.assertRaisesRegex(
            AssertionError, "Cannot declare subproject 'provider'"
        ):
            self.configure()

    def test_duplicate_package_provider_is_rejected(self) -> None:
        with (self.source / "CMakeLists.txt").open("a", encoding="utf-8") as cmake_file:
            cmake_file.write(
                "\ntherock_cmake_subproject_provide_package(\n"
                "  consumer BaselineFixture lib/cmake/BaselineFixture)\n"
            )

        with self.assertRaisesRegex(
            AssertionError, "Duplicate package provided by consumer: BaselineFixture"
        ):
            self.configure()

    def test_build_dependency_is_not_copied_to_consumer_dist(self) -> None:
        self.configure()
        self.build_target("consumer+stage")

        # The provider must be usable, but only runtime dependencies belong in
        # the consumer's distribution. Its own installed output is retained.
        self.assertEqual(
            (self.provider_dist / "share/payload.txt").read_text(encoding="utf-8"),
            "provider payload\n",
        )
        consumer_dist = self.build / "consumer" / "dist"
        self.assertEqual(
            (consumer_dist / "share/dependency.txt").read_text(encoding="utf-8"),
            "provider payload\n",
        )
        self.assertFalse((consumer_dist / "share/payload.txt").exists())
        self.assertFalse((consumer_dist / "lib/cmake/BaselineFixture").exists())

    def test_custom_install_destination_is_preserved_in_stage_and_dist(self) -> None:
        self.configure("-DPROVIDER_INSTALL_DESTINATION=custom/prefix")
        self.build_target("provider+stage")

        for directory in (self.provider_stage, self.provider_dist):
            with self.subTest(directory=directory):
                self.assertEqual(
                    (directory / "custom/prefix/share/payload.txt").read_text(
                        encoding="utf-8"
                    ),
                    "provider payload\n",
                )
                self.assertFalse((directory / "share/payload.txt").exists())

    def test_excluded_project_requires_an_explicit_target(self) -> None:
        self.configure("-DEXCLUDE_CONSUMER=ON")
        self.build_target()

        # The default build does real work for the included provider, while
        # leaving the excluded consumer unconfigured and unstaged.
        self.assertEqual(self.events(), ["configure", "build", "install"])
        self.assertFalse((self.build / "consumer/build/CMakeCache.txt").exists())
        consumer_output = self.build / "consumer/stage/share/dependency.txt"
        self.assertFalse(consumer_output.exists())

        self.build_target("consumer+stage")
        self.assertEqual(
            consumer_output.read_text(encoding="utf-8"), "provider payload\n"
        )
        self.assertEqual(self.events(), ["configure", "build", "install"])

    def test_unchanged_stage_build_skips_completed_steps(self) -> None:
        self.configure()
        self.build_target("provider+stage")
        self.assertEqual(self.events(), ["configure", "build", "install"])

        self.build_target("provider+stage")
        self.assertEqual(self.events(), ["configure", "build", "install"])
        self.assertEqual(
            (self.provider_dist / "share/payload.txt").read_text(encoding="utf-8"),
            "provider payload\n",
        )


if __name__ == "__main__":
    unittest.main()
