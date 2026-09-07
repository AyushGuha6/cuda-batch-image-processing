#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow: pip install pillow")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_dir = root / "data" / "input"
    output_dir = root / "data" / "output"

    for index in range(3):
        before = Image.open(input_dir / f"mnist_{index:04d}.png")
        after = Image.open(output_dir / f"mnist_{index:04d}.png")
        before_pixels = list(before.convert("RGB").getdata())
        color_pixels = sum(
            1
            for red, green, blue in before_pixels
            if max(abs(red - green), abs(green - blue), abs(red - blue)) > 8
        )
        after_pixels = list(after.getdata())
        after_max = max(after_pixels)
        print(
            f"sample {index}: before RGB color_px={color_pixels} "
            f"extrema={before.getextrema()}"
        )
        print(
            f"           after  {after.mode} max={after_max} "
            f"extrema={after.getextrema()}"
        )
        if color_pixels < 500:
            raise SystemExit("Before image does not look like RGB input.")
        if after_max == 0:
            raise SystemExit("After image is blank.")

    print("Output verification passed.")


if __name__ == "__main__":
    main()
