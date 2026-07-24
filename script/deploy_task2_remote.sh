#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-usc-e2}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/sihengzhao/hsea26-hw5-task2}"
HEAD_SHA=$(git -C "$LOCAL_ROOT" rev-parse HEAD)

if ! git -C "$LOCAL_ROOT" diff --quiet || ! git -C "$LOCAL_ROOT" diff --cached --quiet; then
  echo "Tracked local changes must be committed before deployment" >&2
  exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
BUNDLE="$TMP_DIR/task2.bundle"
TASK1_REFS="$TMP_DIR/task1-reference-artifacts.tar.gz"

git -C "$LOCAL_ROOT" bundle create "$BUNDLE" HEAD
(
  cd "$LOCAL_ROOT"
  find results/adaptec1 -type f \
    \( -name initial_population.npz -o -path '*/placements/*.pl' \) \
    -path 'results/adaptec1/task1_*' -print0 \
    | tar --null -czf "$TASK1_REFS" --files-from=-
)

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT' '$REMOTE_ROOT/archive'"
scp -q "$BUNDLE" "$REMOTE_HOST:$REMOTE_ROOT/task2.bundle"
scp -q "$TASK1_REFS" "$REMOTE_HOST:$REMOTE_ROOT/task1-reference-artifacts.tar.gz"

ssh "$REMOTE_HOST" bash -s -- "$REMOTE_ROOT" "$HEAD_SHA" <<'REMOTE'
set -euo pipefail
REMOTE_ROOT=$1
HEAD_SHA=$2
NEXT="$REMOTE_ROOT/repo.next"
CURRENT="$REMOTE_ROOT/repo"
rm -rf "$NEXT"
git clone -q "$REMOTE_ROOT/task2.bundle" "$NEXT"
git -C "$NEXT" checkout -q --detach "$HEAD_SHA"

# Reuse the previously verified ignored DREAMPlace source without retaining the
# old non-Git code tree. It is verified against the committed source manifest
# below before the new checkout becomes active.
if [[ -d "$CURRENT/thirdparty/DREAMPlace_source" ]]; then
  mkdir -p "$NEXT/thirdparty"
  cp -a "$CURRENT/thirdparty/DREAMPlace_source" "$NEXT/thirdparty/"
fi

# Preserve historical results outside the active checkout. They remain useful
# for audit history but cannot satisfy current fingerprint gates.
if [[ -d "$CURRENT/results" ]]; then
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  mv "$CURRENT/results" "$REMOTE_ROOT/archive/results-$stamp"
fi
rm -rf "$CURRENT"
mv "$NEXT" "$CURRENT"
mkdir -p "$CURRENT/results/adaptec1"
tar -xzf "$REMOTE_ROOT/task1-reference-artifacts.tar.gz" -C "$CURRENT"
rm -f "$REMOTE_ROOT/task2.bundle" "$REMOTE_ROOT/task1-reference-artifacts.tar.gz"
REMOTE

# Benchmarks are ignored data and are copied from the audited local instance.
ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/repo/benchmarks/ispd2005/adaptec1'"
rsync -az --delete \
  "$LOCAL_ROOT/benchmarks/ispd2005/adaptec1/" \
  "$REMOTE_HOST:$REMOTE_ROOT/repo/benchmarks/ispd2005/adaptec1/"

# Fall back to a source upload only if no prior verified source was available.
if ! ssh "$REMOTE_HOST" "test -d '$REMOTE_ROOT/repo/thirdparty/DREAMPlace_source'"; then
  rsync -az --delete \
    --exclude='/.git/' \
    --exclude='/build/' \
    --exclude='/install/' \
    --exclude='/**/__pycache__/' \
    --exclude='/**/*.pyc' \
    "$LOCAL_ROOT/thirdparty/DREAMPlace_source/" \
    "$REMOTE_HOST:$REMOTE_ROOT/repo/thirdparty/DREAMPlace_source/"
fi

ssh "$REMOTE_HOST" bash -s -- "$REMOTE_ROOT" "$HEAD_SHA" <<'REMOTE'
set -euo pipefail
REMOTE_ROOT=$1
HEAD_SHA=$2
REPO="$REMOTE_ROOT/repo"
actual=$(git -C "$REPO" rev-parse HEAD)
[[ "$actual" == "$HEAD_SHA" ]]
[[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]
python3 "$REPO/script/hash_dreamplace_source.py" \
  "$REPO/thirdparty/DREAMPlace_source" \
  --verify "$REPO/experiments/task2_smoke/dreamplace_source_manifest.json"
printf 'remote_commit=%s\n' "$actual"
printf 'task1_initial_populations=%s\n' \
  "$(find "$REPO/results/adaptec1" -path '*/task1_*/*' -name initial_population.npz | wc -l)"
printf 'task1_saved_placements=%s\n' \
  "$(find "$REPO/results/adaptec1" -path '*/task1_*/*' -path '*/placements/*.pl' | wc -l)"
REMOTE

echo "Deployed exact commit $HEAD_SHA to $REMOTE_HOST:$REMOTE_ROOT/repo"
