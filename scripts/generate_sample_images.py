#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import random

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required. Install with: pip install pillow")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/input")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)

    for index in range(args.count):
        pixels = [
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            for _ in range(args.width * args.height)
        ]
        image = Image.new("RGB", (args.width, args.height))
        image.putdata(pixels)
        filename = os.path.join(args.output, f"sample_{index:04d}.png")
        image.save(filename)

    print(f"Generated {args.count} images in {args.output}")


if __name__ == "__main__":
    main()
