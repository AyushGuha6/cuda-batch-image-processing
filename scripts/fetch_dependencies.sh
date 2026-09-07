#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_DIR="${ROOT_DIR}/lib"

mkdir -p "${LIB_DIR}"

if [[ ! -f "${LIB_DIR}/stb_image.h" ]]; then
  echo "Downloading stb_image.h"
  curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" \
    -o "${LIB_DIR}/stb_image.h"
fi

if [[ ! -f "${LIB_DIR}/stb_image_write.h" ]]; then
  echo "Downloading stb_image_write.h"
  curl -fsSL "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" \
    -o "${LIB_DIR}/stb_image_write.h"
fi

echo "Dependencies ready in ${LIB_DIR}"
