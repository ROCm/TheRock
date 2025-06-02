#!/usr/bin/env bash

# GitHub doesn't currently support deleting old workflows unless you delete
# all previous runs of it. This scripts will delete all the runs for a given
# workflow. Use with caution!!! It will prompt before deletion.
#
# Usage: delete-workflow.sh <org/repo> <workflow-name (aka path to workflow
# file e.g. blahblah.yml )>
#
# PREREQUISITES: Install https://github.com/cli/cli and authenticate with it.
#
#
# Inspired by and mostly copied from:
# https://github.com/orgs/community/discussions/26256

set -oe pipefail

REPOSITORY=${1}
WORKFLOW_NAME=${2}

# Validate arguments
if [[ -z "${REPOSITORY}" ]]; then
  echo "Repository is required"
  exit 1
fi

if [[ -z "${WORKFLOW_NAME}" ]]; then
  echo "Workflow name is required"
  exit 1
fi

echo "Getting all completed runs for workflow ${WORKFLOW_NAME} in ${REPOSITORY}"

RUNS=${(
  gh api \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${REPOSITORY}/actions/workflows/${WORKFLOW_NAME}/runs" \
    --paginate \
    --jq '.workflow_runs[] | select(.conclusion != "") | .id'
) || RUNS=0


if [[ ${RUNS} -eq 0 ]]; then
  echo "No runs found for ${WORKFLOW_NAME} in ${REPOSITORY}"
  exit
fi

echo "Found $(echo '${RUNS}' | wc -l) completed runs for workflow ${WORKFLOW_NAME}"

echo "Would you like to continue:"
select answer in "yes" "no"; do
  case ${answer} in
    yes)
      break
      ;;
    no)
      exit
      ;;
  esac
done

for RUN in ${RUNS}; do
  gh run delete --repo "${REPOSITORY}" "${RUN}" || echo "Failed to delete run ${RUN}"
  sleep 0.1
done
