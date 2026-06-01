# Migration from Single Repo to Monorepo

## Introduction
This document outlines the process for migrating from a single library repository to this monorepo. It covers the necessary steps to ensure a smooth transition, including pre-conditions, conflict resolution, and changes to repository management.

## Pre-conditions
To ensure consistency and maintainability during the migration, the following pre-conditions must be satisfied:

1. **Identify Next Repo to Migrate:**
   - Please refer to the main [README.md](/README.md) on the order of repositories being migrated.
   - This is usually discussed in advance in meetings with the technical leads of that project.

2. **Identify Branches and Pull Requests:**
   - Determine branches and active pull requests that will be affected by the migration.
   - Typically, this is limited to the pull requests targeting `develop` and `release-staging` branches.
   - Any point-fixes for previous releases will not be migrated over.

3. **Pause Merges:**
   - There are GitHub Actions that automatically synchronize changes from the individual repos to the monorepo.
   - These automated actions need to be paused by disabling the workflow on the GitHub UI.
   - develop branch workflow: https://github.com/ROCm/rocm-libraries/actions/workflows/update-subtrees.yml
   - release-staging branch workflow: https://github.com/ROCm/rocm-libraries/actions/workflows/update-release-staging-subtree.yml
   - Announce the pause to key stakeholders and ask them to propagate the news.

## Migration Process

### Step 1: Pull Request Management

1. **Automated Import of Pull Requests:**
   - Pull requests without merge conflicts will be automatically imported with a GitHub Action, only executable by maintainers and admins.
   - This GitHub action will create a feature branch on the monorepo, pulling in the changes from the PR on the original repo using `git subtree`.
   - These imported pull requests will have the `imported pr` label applied.
   - After running the action successfully, close the PR on the original repo.
   - GitHub Action: https://github.com/ROCm/rocm-libraries/actions/workflows/pr-import.yml
   - Example Imported Pull Request on Monorepo: https://github.com/ROCm/rocm-libraries/pull/206
   - Corresponding Pull Request on Original Repo: https://github.com/ROCm/Tensile/pull/2135

2. **Conflict Resolution:**
   - For pull requests with merge conflicts, add a comment explaining the merge conflict and blocking issue preventing import.
   - Collaborate with contributors to import these PRs after the migration period, or the contributor can reopen the pull request themselves on the monorepo.

3. **NPI Development:**
   - Repeat this import process for the monorepo on GitHub EMU for npi work.

### Step 2: Issue and Comment Import

1. **Issue Import:**
   - Import all open issues from both public and EMU repositories with a GitHub Action, only executable by maintainers and admins.
   - Comments are copied over in the imported issues.
   - GitHub Action: https://github.com/ROCm/rocm-libraries/actions/workflows/issue-import.yml
   - Ensure issue status and labels are preserved during migration.
   - Look for any weird unicode characters that get mangled during the automated import.
   - After running the action successfully, close the issue with a comment on the original repo.
   - Example Imported Issue on Monorepo: https://github.com/ROCm/rocm-libraries/issues/100
   - Corresponding Issue on Original Repo: https://github.com/ROCm/rocThrust/issues/501

### Step 3: Path-Based Commit History

1. **Use of Git Filter-Repo:**
   - Utilize `git filter-repo` at the migration point to add path-based commit history.
   - As changing the contents of a commit will change the output the hash function, commit SHA will change.
   - The filter-repo tool is used to add a snippet at the end of the old commit to refer to the old commit SHA.
   - It is not possible to preserve the same commit SHA if the metadata is changed to point to new paths, as the hash function output changes.
   - Example directory view: https://github.com/ROCm/rocm-libraries/commits/develop/projects/rocrand/library
   - Example commit view: https://github.com/ROCm/rocm-libraries/commit/ea8b6884a0f2a0ec80ff7811bc5ec042600790e9

2. **command sequence example**

Some steps are added to ensure you have latest checked out, in case you're copy-pasting and already have the repositories checked out beforehand.
```
python3 -m pip install --user git-filter-repo
git clone git@github.com:ROCm/hipBLAS-common.git
pushd hipBLAS-common
git checkout develop
git pull origin
git checkout -b filtered/hipblas-common
git filter-repo --path-rename '':'projects/hipblas-common/' --commit-callback "original_hash = commit.original_id.decode(); original_message = commit.message.decode(); new_message = f'{original_message}\\n\\n[ROCm/hipBLAS-common commit: {original_hash}]' if original_message.strip() else f'[ROCm/hipBLAS-common commit: {original_hash}]'; commit.message = new_message.encode()" --force
git remote add monorepo git@github.com:ROCm/rocm-libraries.git
git push monorepo filtered/hipblas-common
popd
git clone git@github.com:ROCm/rocm-libraries.git
git checkout develop
git pull origin
git branch backup/develop-hipblas-common
git checkout filtered/hipblas-common
git checkout -b preserved/hipblas-common
git merge origin/develop --allow-unrelated-histories
# Set merge commit message to "Import path-preserved history of hipblas-common into the monorepo."
git push --set-upstream origin preserved/hipblas-common
git checkout develop
git reset --hard preserved/hipblas-common
git push origin develop
# Double check contents. Make sure no stray developers merged PRs on either repo during this period. Manually pull in those PRs for these exceptional cases.
# Delete the temporary branches created in this sequence.
```

### Step 4: CI/CD Triggers

1. **CI/CD Trigger Points:**
   - Modify the existing CI/CD systems to be triggered off changes to this project in the monorepo.

### Step 5: Repository Adjustments

1. **Default Branch Deprecation:**
   - Change the default branch of the original repository with a clear deprecation notice.
   - Example: https://github.com/ROCm/rocPRIM/tree/develop_deprecated

2. **Disable Dependabot Updates:**
   - Cease automatic dependency updates in the old repository to streamline the focus on the monorepo.
   - Clear the contents in this file on the original repo: https://github.com/ROCm/rocPRIM/blob/develop_deprecated/.github/dependabot.yml
   - In the original repo settings, go to Security -> Advanced Security and disable all the Dependabot settings.

3. **Protection Rules:**
   - Use branch protection to make the new default branch with the deprecation notice read-only.
   - Create a ruleset for the `develop` branch to also be restrictive, but allow the assistant-librarian bot exceptions to push patches to the original repository.

### Step 6: Source of Truth Declaration

1. **Update repos-config.json:**
   - Update the true/false values in the [`repos-config.json`](/.github/repos-config.json) file that automated workflows use to determine which way the source gets synchronized..
   - `auto_subtree_pull` should now be false, `auto_subtree_push` should now be true for this migrated project. `monorepo_source_of_truth` should be true to reflect the new source of truth is the monorepo.
   - Make this change on both the `develop` and `release-staging` branches.
   - https://github.com/ROCm/rocm-libraries/blob/develop/.github/repos-config.json
   - https://github.com/ROCm/rocm-libraries/blob/release-staging/rocm-rel-7.0/.github/repos-config.json

2. **Update the monorepo README.md:**
   - Update the migration status on the monorepo's main readme to indicate the migration has been completed.
   - https://github.com/ROCm/rocm-libraries/blob/develop/README.md

## Post-Migration Activities

1. **Re-enable synchronization jobs:**
   - Re-enable any automated workflows that were paused.

2. **Communication:**
   - Communicate to key stakeholders the successful completion of the migration.
   - Continue daily meetings and active written communications to offer support for any issues that arise.

3. **Automated Patching of Original Repos:**
    - During the migration period, when a pull request is merged on the monorepo, the contents of the pull request will be split into patches to be pushed onto the original repos.
    - This supports potential pull requests that touch multiple projects.
    - Example pull request on the monorepo: https://github.com/ROCm/rocm-libraries/pull/230
    - Corresponding patches on the original repos:
        - https://github.com/ROCm/hipCUB/commit/50438ec4971def627729ea3d9dc1485e52b09e48
        - https://github.com/ROCm/hipRAND/commit/74afe303def580290a8e5b149ea13ae739bc4c61
        - https://github.com/ROCm/rocPRIM/commit/0514a7bfdd44b324654b53f885dec928af61279a
        - https://github.com/ROCm/rocRAND/commit/39fe7d9dca493765573c3c8be275328547ea2abe
        - https://github.com/ROCm/rocThrust/commit/cdcc666a4c42770fcb7d9fde7c71c243b53c476e

4. **Monitoring:**
   - Monitor the monorepo for any issues or discrepancies.
   - If the automated patching for a PR failed to make it to the original repo, use this GitHub Action: https://github.com/ROCm/rocm-libraries/actions/workflows/pr-merge-sync-patches-manual.yml

## Conclusion

This migration process aims to assist the ROCm development teams transition from many repos to a monorepo by addressing the topics above. By following these outlined steps, we aim to maintain and improve the quality of our development workflow post-migration.
