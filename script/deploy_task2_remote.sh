#!/usr/bin/env bash
set -euo pipefail
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-usc-e2}"
REMOTE_ROOT="${REMOTE_ROOT:-~/hsea26-hw5-task2}"

ssh "$REMOTE_HOST" "mkdir -p $REMOTE_ROOT/repo $REMOTE_ROOT/{logs,manifests,build,cache,tmp,conda-pkgs,pip-cache,home,dreamplace-install}"

rsync -az --partial --delete --info=stats2,progress2 \
  --exclude='/.git/' \
  --exclude='/results/' \
  --exclude='/assets/' \
  --exclude='/*.typ' \
  --include='/benchmarks/' \
  --include='/benchmarks/ispd2005/' \
  --include='/benchmarks/ispd2005/adaptec1/***' \
  --exclude='/benchmarks/***' \
  --exclude='/.task2_runtime/' \
  --exclude='/.pytest_cache/' \
  --exclude='/**/__pycache__/' \
  --exclude='/**/*.pyc' \
  --exclude='/experiments/task1_analysis/' \
  --exclude='/experiments/task1_runs/' \
  --exclude='/experiments/task1_remote_logs/' \
  --exclude='/experiments/task2_analysis/' \
  --exclude='/experiments/task2_runs/*/' \
  --include='/experiments/task2_smoke/dreamplace_source_manifest.json' \
  --exclude='/experiments/task2_smoke/***' \
  --exclude='/**/.git' \
  --exclude='/**/.git/' \
  --exclude='/thirdparty/DREAMPlace_source/build/' \
  --exclude='/thirdparty/DREAMPlace_source/install/' \
  "$LOCAL_ROOT/" "$REMOTE_HOST:$REMOTE_ROOT/repo/"

ssh "$REMOTE_HOST" "find $REMOTE_ROOT/repo/benchmarks -mindepth 1 -maxdepth 1 ! -name ispd2005 -exec rm -rf {} +; find $REMOTE_ROOT/repo/benchmarks/ispd2005 -mindepth 1 -maxdepth 1 ! -name adaptec1 -exec rm -rf {} +"

LOCAL_MANIFEST_SHA=$(sha256sum "$LOCAL_ROOT/experiments/task2_smoke/dreamplace_source_manifest.json" | awk '{print $1}')
REMOTE_MANIFEST_SHA=$(ssh "$REMOTE_HOST" "sha256sum $REMOTE_ROOT/repo/experiments/task2_smoke/dreamplace_source_manifest.json" | awk '{print $1}')
if [[ "$LOCAL_MANIFEST_SHA" != "$REMOTE_MANIFEST_SHA" ]]; then
  echo "Remote source manifest differs after upload" >&2
  exit 1
fi

ssh "$REMOTE_HOST" "python3 $REMOTE_ROOT/repo/script/hash_dreamplace_source.py $REMOTE_ROOT/repo/thirdparty/DREAMPlace_source --verify $REMOTE_ROOT/repo/experiments/task2_smoke/dreamplace_source_manifest.json"

echo "Deployed to $REMOTE_HOST:$REMOTE_ROOT"
