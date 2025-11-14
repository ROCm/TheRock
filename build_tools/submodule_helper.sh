#!/usr/bin/env bash
set -euo pipefail

BRANCH_INPUT="${1:-}"
COMPONENTS="${2:-}"
NO_PIN_CK="${3:-true}"
EXTRA_FLAGS="${4:-}"

echo "===> Starting submodule bump process"
echo "Inputs:"
echo "  Branch input: $BRANCH_INPUT"
echo "  Components: $COMPONENTS"
echo "  No pin CK: $NO_PIN_CK"
echo "  Extra flags: $EXTRA_FLAGS"
echo

git config --global user.name "therockbot"
git config --global user.email "therockbot@amd.com"

DATE=$(date +%Y%m%d)
SAFE_COMPONENTS=$(echo "$COMPONENTS" | tr ',' '-' | tr ' ' '-' | tr -cd '[:alnum:]-')

if [ -z "$BRANCH_INPUT" ]; then
  BRANCH_NAME="bump-${SAFE_COMPONENTS:-submodule}-$DATE"
else
  BRANCH_NAME="${BRANCH_INPUT}-${SAFE_COMPONENTS:-submodule}-$DATE"
fi

echo "Generated branch name: $BRANCH_NAME"
echo "$BRANCH_NAME" > branch_name.txt
echo "$DATE" > date.txt

CMD="python3 ./build_tools/bump_submodules.py --push-branch --branch-name $BRANCH_NAME"

if [ -n "$COMPONENTS" ]; then
  CMD="$CMD --components $COMPONENTS"
fi
if [ "$NO_PIN_CK" == "true" ]; then
  CMD="$CMD --no-pin-ck"
fi
if [ -n "$EXTRA_FLAGS" ]; then
  CMD="$CMD $EXTRA_FLAGS"
fi

echo "Running: $CMD"
eval $CMD

# === Prepare PR body with submodule diff ===
echo "Preparing PR body..."
{
  echo "Automated submodule bump"
  echo "This PR was automatically created by a GitHub App to update submodules."
  echo
  echo "## Submodule changes"
  echo
  git fetch origin main
  git diff origin/main "$BRANCH_NAME" | \
    grep -E '^diff --git |^\+\+\+|^---|^\+Subproject commit|^-Subproject commit' | \
    sed \
      -e 's/^diff --git/### diff --git/g' \
      -e 's/^---/Old path:/g' \
      -e 's/^\+\+\+/New path:/g' \
      -e 's/^-Subproject commit/  Old commit:/g' \
      -e 's/^\+Subproject commit/  New commit:/g'
} > pr_body.txt

echo "PR body prepared at pr_body.txt"
