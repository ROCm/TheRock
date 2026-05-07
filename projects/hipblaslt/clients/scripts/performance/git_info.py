# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
This module is used to create git info file,
if it is not found in output_folder_location specified by the end user.
"""


from pathlib import Path
import git
import os


def get_merge_id(git_log: list):

    """
    This method is used to get merge id and pull id from a git log.

    Args
    ------
    git_log : string consisting of all the information about commit id.

    Returns
    -------
    merge_id, pull_id : Merge id , Pull id for the latest commit.

    """

    merge_id = ""
    pull_id = ""

    try:
        merge_id = git_log[0][7:]
        for i in git_log:
            if "merge pull request #" in i.lower():
                pull_id = i.lower().split()[3][1:]

    except Exception as e:
        print("No Merge requests found.")

    return merge_id, pull_id


def create_github_file(filename: str, dir_name: str = None) -> str:
    """
    This method creates a github information file in the specified directory.

    This function creates four different parameters of a
    git repo { Branch Name , Merge Id ,  Pull Id ,  Git Hash }

    To run this function you must be in a project/package or in any git repo folder
    (or) you need to give the location of a git repo when prompted.

    Args
    -----
    filename : Absolute filename in which the path is to be created
    dir_name : Absolute path where the git hash to be generated (optional)

    Returns
    -------
    filename : Absolute filename of created file

    Raises
    ------
    InvalidGitRepositoryError : If the package is running outside of a git repo
    IOError (Is a directory error) : If the specified filename ( input ) should be a file
                                     and not a directory

    Usage
    -----
    CLI syntax
    ----------
    linux : python3 -m pts_amd create_git_info < filename >
    windows : python -m pts_amd create_git_info < filename >

    """

    # Declaring the git info parameters as empty strings
    branch_name = "None"
    merge_id = "None"
    pull_id = "None"
    git_hash = "None"

    try:
        cwd = os.getcwd()
        if dir_name:
            if os.path.isdir(Path(dir_name)):
                os.chdir(Path(dir_name))
        try:
            repo = git.Repo(Path(dir_name))
            os.chdir(cwd)
        except (git.InvalidGitRepositoryError, git.exc.NoSuchPathError, Exception) as err:
            print(err)
            os.chdir(cwd)
            repo = git.Repo(search_parent_directories=True)
        # Retrieving the values, when package/project is running inside any git repo
        # Check if the repo is in a detached HEAD state
        if repo.head.is_detached:
            print("Repository is in a detached HEAD state.")
            # You can use repo.head.commit to get the current commit
            git_hash = repo.head.commit.hexsha
            branches_output = repo.git.branch('--contains', git_hash)
            branches = [
                branch.strip().lstrip("* ")  # Remove leading "* " for the current branch
                for branch in branches_output.split("\n")
                if branch.strip()  # Exclude empty lines
            ]
            branch_name = branches[0] if branches else "None"
        else:
            # Safe to access active_branch
            branch_name = repo.active_branch.name
            git_hash = repo.head.object.hexsha
            git_logs = repo.git.log("--grep=Merge", "--max-count=1")
            if len(git_logs) > 0:
                merge_id, pull_id = get_merge_id(git_logs.split("\n"))

    except (git.InvalidGitRepositoryError, git.exc.NoSuchPathError):
        print(
            "Could not find any git files from github repo in the specified output folder location."
        )
        return filename

    # If all parameters are captured successfully then those parameters are written in a file.
    try:
        with open(filename, "w") as file:
            file.write(f"Branch Name : {branch_name}\n")
            file.write(f"Merge Id : {merge_id}\n")
            file.write(f"Pull Id : {pull_id}\n")
            file.write(f"Git Hash : {git_hash}\n")

    except Exception as error:
        print(error)
        return filename

    return filename
