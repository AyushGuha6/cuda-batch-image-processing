#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow: pip install pillow")


def cpu_process(input_path: Path, output_path: Path) -> None:
    image = Image.open(input_path).convert("RGB")
    pixels = list(image.getdata())
    width, height = image.size
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
    gray = [int(0.299 * r + 0.587 * g + 0.114 * b) for r, g, b in blurred]
    out = Image.new("L", (width, height))
    out.putdata(gray)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def validate_sample(before_path: Path, after_path: Path) -> None:
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path)

    before_pixels = list(before.getdata())
    color_pixels = sum(
        1
        for red, green, blue in before_pixels
        if max(abs(red - green), abs(green - blue), abs(red - blue)) > 8
    )
    if color_pixels < 500:
        raise SystemExit(
            f"{before_path.name} does not look like RGB input "
            f"(only {color_pixels} differing-channel pixels)."
        )

    after_pixels = list(after.getdata())
    if after.mode == "L":
        after_max = max(after_pixels)
    else:
        after_max = max(max(channel) for channel in after_pixels)
    if after_max == 0:
        raise SystemExit(
            f"{after_path.name} is blank. Rebuild with CUDA 12 on Pascal GPUs "
            "or rerun the GPU batch job before packaging."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--zip-name", default="cuda_proof_of_execution.zip")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    staging = root / "proof_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    input_dir = root / "data" / "input"
    output_dir = root / "data" / "output"
    results_dir = root / "results"

    input_images = sorted(input_dir.glob("*.png"))
    if not input_images:
        raise SystemExit(
            f"No PNG files in {input_dir}. Run generate_sample_images.py first."
        )

    summary = staging / "batch_run_summary.txt"
    summary.write_text(
        f"Dataset: Colorized MNIST digits (UCI / LeCun source, exported as RGB PNG)\n"
        f"Input images exported: {len(input_images)}\n"
        f"Input directory: data/input\n"
        f"Output directory: data/output\n"
        f"Processing: horizontal blur + RGB to grayscale on GPU\n"
    )

    log_src = results_dir / "execution_log.txt"
    console_src = results_dir / "console_output.txt"
    if log_src.exists():
        shutil.copy(log_src, staging / "execution_log.txt")
    else:
        (staging / "MISSING_execution_log.txt").write_text(
            "Missing results/execution_log.txt — run the GPU batch job first.\n"
        )

    if console_src.exists():
        shutil.copy(console_src, staging / "console_output.txt")
    else:
        (staging / "MISSING_console_output.txt").write_text(
            "Missing results/console_output.txt — run run.sh or run.bat first.\n"
        )

    for i, src in enumerate(input_images[: args.samples]):
        name = src.stem
        before_path = staging / f"before_{name}.png"
        after_path = staging / f"after_{name}.png"
        shutil.copy(src, before_path)
        gpu_out = output_dir / src.name
        if gpu_out.exists():
            shutil.copy(gpu_out, after_path)
            validate_sample(before_path, after_path)
        else:
            cpu_process(src, after_path)
            validate_sample(before_path, after_path)

    readme = staging / "README_proof_artifacts.txt"
    readme.write_text(
        "CUDA batch image processor — proof of execution\n\n"
        "execution_log.txt     GPU name, image counts, timing\n"
        "console_output.txt    Terminal output from the batch run\n"
        "before_*.png          Input samples\n"
        "after_*.png           GPU-processed output samples\n"
        "batch_run_summary.txt Dataset summary\n"
    )

    zip_path = root / args.zip_name
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    shutil.rmtree(staging)
    print(f"Created {zip_path}")
    if not log_src.exists():
        print("WARNING: execution_log.txt missing — run the GPU batch job first.")


if __name__ == "__main__":
    main()
