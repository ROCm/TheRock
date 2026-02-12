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


POSIX_EXE_STUB_TEMPLATE = r"""#define _GNU_SOURCE
#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char EXEC_RELPATH[] = "@EXEC_RELPATH@";

// Get the directory containing the main executable.
// main_addr should be a pointer to a function in the main executable (used for
// dladdr fallback).
// Returns a heap-allocated string that must be freed, or NULL on failure.
static char* get_main_dir(void* main_addr) {
    char* main_path = NULL;

#if defined(__linux__) || defined(__CYGWIN__)
    // On Linux, /proc/self/exe is the most reliable way to get the executable
    // path. This works even when the program is invoked via PATH without an
    // absolute path.
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len > 0) {
        exe_path[len] = '\0';
        // Resolve any symlinks in the path.
        char* real_path = realpath(exe_path, NULL);
        if (real_path) {
            main_path = real_path;
        } else {
            main_path = strdup(exe_path);
        }
    }
#endif

    // Fallback: use dladdr to get the path. This requires the executable to be
    // linked as PIE (-fPIE) and may return just a filename in some cases.
    if (!main_path) {
        Dl_info info;
        if (dladdr(main_addr, &info) && info.dli_fname) {
            // Try to resolve to an absolute path.
            char* real_path = realpath(info.dli_fname, NULL);
            if (real_path) {
                main_path = real_path;
            } else {
                main_path = strdup(info.dli_fname);
            }
        }
    }

    if (!main_path) {
        fprintf(stderr, "could not determine path of main program\n");
        return NULL;
    }

    // Extract the directory by finding the last slash.
    char* last_slash = strrchr(main_path, '/');
    if (!last_slash) {
        fprintf(stderr, "could not find path component of main program: '%s'\n",
                main_path);
        free(main_path);
        return NULL;
    }
    *last_slash = '\0';

    return main_path;
}

int main(int argc, char** argv) {
    char* main_dir = get_main_dir((void*)(intptr_t)main);
    if (!main_dir) {
        return 1;
    }

    // Compute the new target relative to the containing directory.
    char* target = malloc(
        strlen(main_dir) + 1 /* slash */ + strlen(EXEC_RELPATH) + 1 /* nul */);
    strcpy(target, main_dir);
    strcat(target, "/");
    strcat(target, EXEC_RELPATH);
    free(main_dir);

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
        # Must link as PIE so that the main executable is dynamic (i.e. dladdr
        # will work).
        subprocess.check_call(
            [cc, "-fPIE", "-o", str(output_file), str(source_file), "-ldl"]
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("ERROR: Expected {out_file} {relative_link_to}")
        sys.exit(1)
    generate_exe_link_stub(sys.argv[1], sys.argv[2])
