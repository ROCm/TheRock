import subprocess
import os
import sys

# using shell script here to play safety because we used to
# have corruptions in .gitconfig file on windows machines.
# https://github.com/ROCm/TheRock/issues/2929

def run_git_config(args):
    """Helper to run git config commands and return result."""
    return subprocess.run(["git", "config", "--global"] + args, 
                          capture_output=True, text=True)

def update_gitconfig_safe_directory(val_new_dir: str, new_fetch_val:str):
    # val_old_dir=$(git config --global --get safe.directory)
    result = run_git_config(["--get", "safe.directory"])
    val_old_dir = result.stdout.strip()

    # 1. Handle the safe.directory
    # Check if update is needed
    if not val_old_dir or val_old_dir != val_new_dir:
        print(f"Updating .gitconfig directory keyword. Old value: '{val_old_dir}', new value: '{val_new_dir}'.")

        # git config --global safe.directory "$val_new_dir"
        res = run_git_config(["safe.directory", val_new_dir])

        if res.returncode != 0:
            print(f"git config safe.directory value update failed with error code: {res.returncode}")
            print("trying git config again but now with --replace-all parameter to maintain compatibility")
            res = run_git_config(["--replace-all", "safe.directory", val_new_dir])

            if res.returncode != 0:
                print(f"git config safe.directory value update failed with error code: {res.returncode}")
                sys.exit(res.returncode)
            else:
                print("git config --replace-all safe.directory value update ok")
        else:
            print("git config safe.directory value update ok")
    else:
        print(f"Skipping .gitconfig safe.directory value write, value is already ok: '{val_new_dir}'")

    # 2. Handle fetch.parallel value write
    # Get current fetch.parallel value
    res_fetch = run_git_config(["--get", "fetch.parallel"])
    current_fetch_val = res_fetch.stdout.strip()

    if current_fetch_val != new_fetch_val:
        print(f"Updating fetch.parallel. Current: '{current_fetch_val}', New: {new_fetch_val}.")
        run_git_config(["fetch.parallel", new_fetch_val])
    else:
        print(f"fetch.parallel is already set to {new_fetch_val}.")

if __name__ == "__main__":
    # Use the first command-line argument if it exists (sys.argv[1]),
    # otherwise default to the current working directory (os.getcwd()
    new_dir_param = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    # then use second value for fetch.parallel thread count
    fetch_parallel_count = sys.argv[2] if len(sys.argv) > 2 else 10
    update_gitconfig_safe_directory(new_dir_param, fetch_parallel_count)
    sys.exit(0)

