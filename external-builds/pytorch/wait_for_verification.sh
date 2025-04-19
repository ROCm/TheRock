#!/bin/bash
set -xeuo pipefail

if [ -z "$GH_TOKEN" ]; then
  echo "❌ GH_TOKEN is not set. Cannot authenticate with GitHub CLI."
  exit 1
fi

if [ -z "$DISPATCH_ID" ]; then
  echo "❌ DISPATCH_ID is required."
  exit 1
fi

REPO=${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}

echo "🔍 Waiting for workflow 'verify_docker_image.yml' with dispatch ID $DISPATCH_ID..."

RUN_ID=""

for attempt in {1..30}; do
  echo "⏱️  Polling attempt $attempt..."

  if [ -z "$RUN_ID" ]; then
    RUN_ID=$(gh api "/repos/${REPO}/actions/runs?event=workflow_dispatch&per_page=10" \
      --jq ".workflow_runs[] | .id" |
      while read run_id; do
        gh api "/repos/${REPO}/actions/runs/$run_id/jobs" \
          --jq ".jobs[].steps[]? | select(.name == \"Identifier $DISPATCH_ID\") | \"$run_id\"" || true
      done | head -n 1)

    if [ -z "$RUN_ID" ]; then
      echo "⚠️  No matching workflow run found yet. Retrying in 30 seconds..."
      sleep 30
      continue
    fi

    echo "✅ Found run: $RUN_ID"
  fi

  STATUS=$(gh run view "$RUN_ID" --json status -q '.status')
  CONCLUSION=$(gh run view "$RUN_ID" --json conclusion -q '.conclusion')

  echo "🔄 Run ID: $RUN_ID | Status: $STATUS | Conclusion: $CONCLUSION"

  if [[ "$STATUS" == 'completed' ]]; then
    if [[ "$CONCLUSION" == 'success' ]]; then
      echo "✅ Verification passed for $DISPATCH_ID"
      exit 0
    else
      echo "❌ Verification failed for $DISPATCH_ID"
      gh run view "$RUN_ID" --log || echo "⚠️ Could not fetch logs. Check manually."
      exit 1
    fi
  fi

  echo "⏳ Workflow still in progress. Sleeping for 30 seconds..."
  sleep 30
done

echo "❌ Timed out after waiting 15 minutes for verification."
exit 1
