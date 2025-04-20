#!/usr/bin/env bash
set -xeuo pipefail

if [ -z "${GH_TOKEN:-}" ] || [ -z "${DISPATCH_ID:-}" ] || [ -z "${START_TIME:-}" ] || [ -z "${GITHUB_REPOSITORY:-}" ]; then
  echo "❌ Required environment variables: GH_TOKEN, DISPATCH_ID, START_TIME, GITHUB_REPOSITORY"
  exit 1
fi

echo "🔍 Waiting for run matching DISPATCH_ID=$DISPATCH_ID, after $START_TIME..."

RUN_ID=""

for attempt in {1..30}; do
  echo "⏱️  Polling attempt $attempt..."

  if [ -z "$RUN_ID" ]; then
    RUN_ID=$(gh api "/repos/${GITHUB_REPOSITORY}/actions/runs?event=workflow_dispatch&per_page=20&created=>=$START_TIME" \
      --jq ".workflow_runs[] | .id" |
      while read run_id; do
        gh api "/repos/${GITHUB_REPOSITORY}/actions/runs/$run_id/jobs" \
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

  if [[ "$STATUS" == "completed" ]]; then
    if [[ "$CONCLUSION" == "success" ]]; then
      echo "✅ Verification passed for $DISPATCH_ID"
      exit 0
    else
      echo "❌ Verification failed for $DISPATCH_ID"
      gh run view "$RUN_ID" --log || echo "⚠️ Could not fetch logs."
      exit 1
    fi
  fi

  echo "⏳ Workflow still in progress. Sleeping for 30 seconds..."
  sleep 30
done

echo "❌ Timed out after waiting 15 minutes for verification.."
exit 1
