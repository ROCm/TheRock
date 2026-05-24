# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generates a stub executable and saves it to the given output_file.

The stub executable will exec a child at the given path relative to its
origin. This emulates how a symlink to an executable would function and can
be used in place of a symlink (in case if symlinks are not tolerable in some
situation).

Example usage (creates a stub that invokes /bin/ls):
  python -m _therock_utils.exe_stub_gen /tmp/foobar_stub ../bin/ls
  /tmp/foobar_stub
"""

from pathlib import Path
import os
import platform
import subprocess
import sys
import tempfile


POSIX_EXE_STUB_TEMPLATE = r"""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char EXEC_RELPATH[] = "@EXEC_RELPATH@";

int main(int argc, char** argv) {
    // Use /proc/self/exe instead of dladdr(main): dladdr() fails when argv[0]
    // has no '/' (e.g. MLIR's ROCDL target passes bare "ld.lld" as argv[0]),
    // causing dli_fname to have no path component and strrchr to return NULL.
    char main_path[4096];
    ssize_t len = readlink("/proc/self/exe", main_path, sizeof(main_path) - 1);
    if (len == -1) {
        perror("could not readlink /proc/self/exe");
        return 1;
    }
    main_path[len] = '\0';

    char* last_slash = strrchr(main_path, '/');
    if (!last_slash) {
        fprintf(stderr, "could not find path component of main program: '%s'\n",
                main_path);
        return 1;
    }
    *last_slash = 0;

    // Compute the new target relative to the containing directory.
    char* target = malloc(
        strlen(main_path) + 1 /* slash */ + strlen(EXEC_RELPATH) + 1 /* nul */);
    strcpy(target, main_path);
    strcat(target, "/");
    strcat(target, EXEC_RELPATH);

    // Exec with altered target executable but preserving argv[0] as pointing
    // to the current program. This emulates how invocation via a symlink
    // works.
    int rc = execv(target, argv);
    if (rc == -1) {
        fprintf(stderr, "could not exec %s: ", target);
        perror(0);
        return 1;
    }
    return 0;
}
"""


def generate_exe_link_stub(output_file: Path, relative_link_to: str):
    if platform.system() == "Windows":
        raise NotImplementedError("generate_exe_link_stub NYI for Windows")

    # Generic Posix impl.
    with tempfile.TemporaryDirectory() as td:
        source_file = Path(td) / "stub.c"
        source_contents = POSIX_EXE_STUB_TEMPLATE.replace(
            "@EXEC_RELPATH@", relative_link_to
        )
        source_file.write_text(source_contents)
        cc = os.getenv("CC", "cc")
        subprocess.check_call(
            [cc, "-fPIE", "-o", str(output_file), str(source_file)]
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Expected {out_file} {relative_link_to}")
        sys.exit(1)
    generate_exe_link_stub(sys.argv[1], sys.argv[2])
