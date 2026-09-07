#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/input data/output results bin

if [[ ! -d data/input ]] || [[ -z "$(ls -A data/input 2>/dev/null || true)" ]]; then
  echo "Preparing MNIST dataset..."
  python3 scripts/prepare_mnist_dataset.py --output data/input --count 128
fi

echo "Running batch GPU image processor on $(ls data/input | wc -l | tr -d ' ') images"
./bin/batch_image_processor.exe \
  --input data/input \
  --output data/output \
  --log results/execution_log.txt \
  --mode both \
  --threads 256 \
  2>&1 | tee results/console_output.txt

echo "Done. See results/execution_log.txt and data/output/"

python3 scripts/package_proof_artifacts.py --root .
