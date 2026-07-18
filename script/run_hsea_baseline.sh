#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT"
cd "$ROOT"
OUT="${1:-results/hsea_baseline_2026}"

conda run -n hsea26-hw5 python -m src.hsea.cli search-task1 \
  --benchmark=adaptec1 --n-macro=128 --grid=64 --routing-bins=16 --n-cpu=8 \
  --trials=6 --seeds=1 --max-evaluations=64 --output="$OUT/search_task1"

conda run -n hsea26-hw5 python -m src.hsea.cli search-task2 \
  --benchmark=adaptec1 --n-macro=128 --grid=64 --routing-bins=16 --n-cpu=8 \
  --trials=4 --seeds=1 --max-evaluations=64 --output="$OUT/search_task2"

for seed in 1 2 3; do
  conda run -n hsea26-hw5 python -m src.hsea.cli run \
    --algorithm=ea --config="$OUT/search_task1/best_config.json" --seed="$seed" \
    --benchmark=adaptec1 --grid=224 --routing-bins=24 --n-cpu=8 \
    --max-evaluations=72 --output="$OUT/formal/ea/seed_$seed"
  conda run -n hsea26-hw5 python -m src.hsea.cli run \
    --algorithm=nsga2 --config="$OUT/search_task2/best_nsga2_config.json" --seed="$seed" \
    --benchmark=adaptec1 --grid=224 --routing-bins=24 --n-cpu=8 \
    --max-evaluations=72 --output="$OUT/formal/nsga2/seed_$seed"
  conda run -n hsea26-hw5 python -m src.hsea.cli run \
    --algorithm=moead --config="$OUT/search_task2/best_moead_config.json" --seed="$seed" \
    --benchmark=adaptec1 --grid=224 --routing-bins=24 --n-cpu=8 \
    --max-evaluations=72 --output="$OUT/formal/moead/seed_$seed"
done

conda run -n hsea26-hw5 python -m src.hsea.cli analyze \
  --root="$OUT/formal" --output="$OUT/formal/analysis.csv"
conda run -n hsea26-hw5 python -m src.hsea.report --root="$OUT"
