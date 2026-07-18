#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda env create -f "$ROOT/environment-hsea.yml" || conda env update -f "$ROOT/environment-hsea.yml" --prune
printf '\nActivate with: conda activate hsea26-hw5\n'
