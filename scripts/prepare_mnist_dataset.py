#!/usr/bin/env python3

from __future__ import annotations

import argparse
import colorsys
import gzip
import os
import struct
import urllib.request

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required. Install with: pip install pillow")

MNIST_TRAIN_URL = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
)
CACHE_DIR = "data/mnist_cache"


def download_mnist_train(cache_path: str) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.isfile(cache_path):
        return
    print("Downloading MNIST training images...")
    urllib.request.urlretrieve(MNIST_TRAIN_URL, cache_path)
    print(f"Saved to {cache_path}")


def read_mnist_images(gz_path: str) -> list[Image.Image]:
    images: list[Image.Image] = []
    with gzip.open(gz_path, "rb") as handle:
        magic, count, rows, cols = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected MNIST magic number: {magic}")
        pixel_count = rows * cols
        for index in range(count):
            raw = handle.read(pixel_count)
            if len(raw) != pixel_count:
                raise ValueError(f"Truncated MNIST file at image {index}")
            images.append(Image.frombytes("L", (cols, rows), raw))
    return images


def colorize_digit(gray_image: Image.Image, image_index: int) -> Image.Image:
    width, height = gray_image.size
    rgb = Image.new("RGB", (width, height))
    rgb_pixels = []
    base_hue = (image_index * 0.137) % 1.0
    for row in range(height):
        for col in range(width):
            value = gray_image.getpixel((col, row))
            if value < 12:
                rgb_pixels.append((0, 0, 0))
                continue
            hue = (base_hue + (col / max(width, 1)) * 0.15) % 1.0
            saturation = 0.55 + 0.45 * (value / 255.0)
            intensity = value / 255.0
            red, green, blue = colorsys.hsv_to_rgb(hue, saturation, intensity)
            rgb_pixels.append(
                (int(red * 255), int(green * 255), int(blue * 255))
            )
    rgb.putdata(rgb_pixels)
    return rgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export colorized MNIST digits as RGB PNG files."
    )
    parser.add_argument("--output", default="data/input")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    cache_file = os.path.join(CACHE_DIR, "train-images-idx3-ubyte.gz")
    download_mnist_train(cache_file)
    images = read_mnist_images(cache_file)

    end_index = args.start_index + args.count
    if end_index > len(images):
        raise SystemExit(
            f"Requested {args.count} images from index {args.start_index}, "
            f"but MNIST train set has {len(images)} images."
        )

    os.makedirs(args.output, exist_ok=True)
    for offset, image in enumerate(images[args.start_index : end_index]):
        color = colorize_digit(image, offset).resize(
            (args.width, args.height), Image.Resampling.NEAREST
        )
        filename = os.path.join(args.output, f"mnist_{offset:04d}.png")
        color.save(filename)

    print(
        f"Exported {args.count} colorized MNIST RGB images to {args.output} "
        f"({args.width}x{args.height} PNG)"
    )


if __name__ == "__main__":
    main()
