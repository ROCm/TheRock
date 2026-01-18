import subprocess
import os
import sys
from pathlib import Path

# using shell script here to test a fix for gitconfig corruptions
# on windows machines.
# https://github.com/ROCm/TheRock/issues/2929


def run_git_config(args):
    """Helper to run git config commands and return result."""
    return subprocess.run(
        ["git", "config", "--global"] + args, capture_output=True, text=True
    )


def get_global_gitconfig_path():
    try:
        # get all configs and their origins
        result = subprocess.check_output(
            ["git", "config", "--list", "--show-origin", "--global"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        loc_arr = result.strip().split("\n")

        # The output format is "file:<path> <key>=<value>"
        if loc_arr:
            for loc_line in loc_arr:
                print(loc_line)
                if loc_line.startswith("file:"):
                    # extract the path part and remove the "file:" prefix
                    path = loc_line.split(" ", 1)[0][5:]
                    if path.startswith("/etc/gitconfig"):
                        print(f"Skipping git config path: {path}")
                    else:
                        # Find the index of the last occurrence
                        index = path.rfind(".gitconfig")
                        # Slice the string from the beginning to the found index
                        if index != -1:
                            path = path[:index] + ".gitconfig"
                            print(f"Found git config path: {path}")
                            return path
                        else:
                            print(f"Invalid path: {path}")
                            return None
            print("Global gitconfig file not found or output format unexpected.")
            return None
        else:
            # The command may not return anything if the file doesn't exist yet
            return None
    except subprocess.CalledProcessError as e:
        # Handle cases where git command fails (e.g., git is not installed)
        print(f"Git command failed: {e.output}")
        return None
    except FileNotFoundError:
        print("Git is not installed or not in the system's PATH.")
        return None


def update_global_gitconfig_value_if_needed(gitconfig_key, val_new: str):
    ret = 0
    # val_old=$(git config --global --get "key")
    val_old = run_git_config(["--get", gitconfig_key])
    val_old = val_old.stdout.strip()

    # Check if update is needed
    if not val_old or val_old != val_new:
        print(
            f"Updating .gitconfig {gitconfig_key} key. Old value: '{val_old}', new value: '{val_new}'."
        )
        # git config --global $gitconfig_key $val_new
        res = run_git_config([gitconfig_key, val_new])
        if res.returncode != 0:
            print(
                f"git config {gitconfig_key} value update to {val_new} failed, error code: {res.returncode}"
            )
        else:
            print(f"git config {gitconfig_key} value updated to {val_new} succesfully")
    else:
        print(
            f"Skipping .gitconfig {gitconfig_key} value update skipped, value is already: '{val_new}'"
        )
    return ret


def force_update_global_gitconfig_value_if_needed(gitconfig_key, val_new: str):
    ret = update_global_gitconfig_value_if_needed(key_param, val_new_param)
    if ret != 0:
        path_str = get_global_gitconfig_path()
        if path_str and path_str.endswith(".gitconfig"):
            print(f"Deleting corrupted .gitconfig: {path_str}")
            path_obj = Path(path_str)
            # delete the file
            path_obj.unlink(missing_ok=True)
            print(f"Trying to update .gitconfig value again: {path_str}")
            ret = update_global_gitconfig_value_if_needed(key_param, val_new_param)
        else:
            print("Error, could not find global .gitconfig file")
            ret = 1
    return ret


if __name__ == "__main__":
    # Use the first command-line argument if it exists (sys.argv[1]),
    # otherwise default to the current working directory (os.getcwd()
    ret = 0
    if len(sys.argv) == 3:
        key_param = sys.argv[1]
        val_new_param = sys.argv[2]
        force_update_global_gitconfig_value_if_needed(key_param, val_new_param)
    else:
        # einval
        ret = 22
    sys.exit(ret)
