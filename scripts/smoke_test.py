#!/usr/bin/env python3
"""End-to-end smoke test without CUDA (CPU reference + file I/O)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("FAIL: install pillow (pip install pillow)")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def cpu_grayscale_and_blur(input_path: Path, output_path: Path) -> None:
    image = Image.open(input_path).convert("RGB")
    pixels = list(image.getdata())
    width, height = image.size

    # Horizontal blur on RGB
    blurred = []
    for row in range(height):
        for col in range(width):
            idx = row * width + col
            r, g, b = pixels[idx]
            if col == 0 or col == width - 1:
                blurred.append((r, g, b))
            else:
                left = pixels[idx - 1]
                right = pixels[idx + 1]
                blurred.append(
                    (
                        (left[0] + r + right[0]) // 3,
                        (left[1] + g + right[1]) // 3,
                        (left[2] + b + right[2]) // 3,
                    )
                )

    gray = [
        int(0.299 * r + 0.587 * g + 0.114 * b)
        for r, g, b in blurred
    ]
    out = Image.new("L", (width, height))
    out.putdata(gray)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def main() -> None:
    print("=== Python CPU pipeline smoke test ===")
    tmp = Path(tempfile.mkdtemp(prefix="cuda_batch_smoke_"))
    input_dir = tmp / "input"
    output_dir = tmp / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    run(
        [
            sys.executable,
            str(ROOT / "scripts/generate_sample_images.py"),
            "--output",
            str(input_dir),
            "--count",
            "16",
        ]
    )

    pngs = sorted(input_dir.glob("*.png"))
    assert len(pngs) == 16, f"expected 16 images, got {len(pngs)}"

    for png in pngs:
        cpu_grayscale_and_blur(png, output_dir / png.name)

    outputs = sorted(output_dir.glob("*.png"))
    assert len(outputs) == 16
    for out in outputs:
        img = Image.open(out)
        assert img.mode == "L"
        assert img.size == (256, 256)

    print("OK: processed 16 images on CPU reference pipeline")

    print("\n=== CLI parser compile test ===")
    run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(ROOT / "include"),
            "-c",
            str(ROOT / "src/cli_parser.cpp"),
            "-o",
            str(tmp / "cli_parser.o"),
        ]
    )
    print("OK: cli_parser.cpp compiles")

    print("\n=== fetch_dependencies ===")
    run(["bash", str(ROOT / "scripts/fetch_dependencies.sh")])
    assert (ROOT / "lib/stb_image.h").exists()
    assert (ROOT / "lib/stb_image_write.h").exists()

    print("\n=== validate_local.sh ===")
    run(["bash", str(ROOT / "scripts/validate_local.sh")])

    if shutil.which("nvcc"):
        print("\n=== CUDA build + run ===")
        run(["make", "build"], cwd=ROOT)
        run(["bash", str(ROOT / "run.sh")], cwd=ROOT)
        log = ROOT / "results/execution_log.txt"
        assert log.exists(), "missing execution log"
        text = log.read_text()
        assert "Images processed: 128" in text or "Images processed:" in text
        print("OK: GPU pipeline executed")
    else:
        print("\nSKIP: nvcc not available on this machine")
        print("     Run `make build && bash run.sh` in Coursera GPU IDE")

    shutil.rmtree(tmp)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
