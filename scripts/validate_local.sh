#!/usr/bin/env bash
# Validates project structure without requiring CUDA (run locally before push).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Structure check ==="
required=(
  README.md INSTALL Makefile run.sh
  include/cli_parser.h include/image_processor.h
  src/main.cpp src/cli_parser.cpp src/image_processor.cu
  scripts/fetch_dependencies.sh scripts/generate_sample_images.py
)

for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "MISSING: $file"
    exit 1
  fi
  echo "OK: $file"
done

echo ""
echo "=== Dependency check ==="
bash scripts/fetch_dependencies.sh
test -f lib/stb_image.h
test -f lib/stb_image_write.h
echo "OK: stb headers"

echo ""
echo "=== Sample data check ==="
python3 scripts/generate_sample_images.py --output /tmp/cuda_batch_test_input --count 4
test -f /tmp/cuda_batch_test_input/sample_0000.png
echo "OK: image generator"

echo ""
if command -v nvcc >/dev/null 2>&1; then
  echo "=== CUDA build ==="
  make build
  echo "OK: built bin/batch_image_processor.exe"
  echo ""
  echo "=== CUDA run (4 images) ==="
  mkdir -p data/input data/output results
  cp /tmp/cuda_batch_test_input/*.png data/input/
  ./bin/batch_image_processor.exe --input data/input --output data/output \
    --log results/execution_log.txt --mode both --threads 256 --max-images 4
  test -f results/execution_log.txt
  echo "OK: GPU run complete"
else
  echo "SKIP: nvcc not found (build/run on Coursera GPU IDE)"
fi

echo ""
echo "All local checks passed."
