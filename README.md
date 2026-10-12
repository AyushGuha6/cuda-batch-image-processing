# CUDA Batch Image Processing

> A GPU batch processor that applies custom CUDA kernels (blur and grayscale) to a folder of PNG images. Final project for Course 3 of a GPU Specialization course.

## Overview

The program loads a directory of images on the CPU, sends them to the GPU, runs custom CUDA kernels on every pixel, and writes the results back to disk. It shows how a simple image-processing step maps onto GPU threads, and how time splits between GPU kernel work and host input/output.

## My contribution

I wrote the CUDA kernels, the C++ host code, the build scripts, and the supporting Python scripts.

## Kernels

| Kernel | What it does |
|---|---|
| `HorizontalBlurKernel` | 3-tap horizontal box blur on RGB channels. Edge pixels are left unchanged. |
| `RgbToGrayscaleKernel` | Grayscale conversion using ITU-R BT.601 luma weights. |

The kernels are written directly in CUDA. This project does not use NVIDIA libraries such as NPP or cuFFT.

## Tools

C++, CUDA, Python (Pillow), Make. Code follows the Google C++ Style Guide.

## Input data

By default, the program uses MNIST digits colorized into 256×256 RGB PNG images (128 images). It can also use a synthetic image generator or your own PNG or JPG files.

## Run options

| Option | Meaning |
|---|---|
| `--input` | Folder of input images |
| `--output` | Folder for processed images |
| `--log` | Path for the execution log |
| `--mode` | `grayscale`, `blur`, or `both` |
| `--threads` | CUDA threads per block (default 256) |
| `--max-images` | Limit on images processed (default: all) |

## Build and run

Requires an NVIDIA GPU, the CUDA Toolkit, and Python 3 with Pillow.

```bash
# Linux
make setup
make build

# Windows
build.bat
```

See `START_HERE.txt` for the full walkthrough, `WINDOWS.md` for Windows notes, and `INSTALL` for setup.

## Testing

Tested on an NVIDIA GeForce GTX 1070 Ti (Pascal, compute capability 6.1). The build targets compute capabilities 6.1, 7.5, and 8.6. GPU kernel time and host input/output time are tracked separately in an execution log, which is generated when you run the program. No benchmark results are published in this repository.

## Limitations

- Only two operations: a horizontal 3-tap blur and a grayscale conversion.
- The blur leaves edge pixels unchanged.
- No performance benchmarks or CPU comparison are included.
- Needs an NVIDIA GPU to run.
