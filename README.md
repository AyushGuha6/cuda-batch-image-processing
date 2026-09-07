# CUDA Batch Image Processor

GPU batch processor for PNG images. Loads a directory of images on the host,
runs custom CUDA kernels for optional horizontal blur and RGB-to-grayscale
conversion, and writes processed files back to disk.

Repository: https://github.com/AyushGuha06/CudaCourse2026

## Layout

```
include/     Headers (cli_parser, image_processor, platform_compat)
src/         main.cpp, cli_parser.cpp, image_processor.cu
lib/         stb_image headers and Windows dirent shim
scripts/     Setup, sample data, proof packaging
bin/         Built executable (not committed)
data/        Input/output images (generated locally)
results/     Execution logs (generated locally)
```

## Requirements

- NVIDIA GPU and CUDA toolkit (`nvcc`)
- Python 3 with Pillow (`pip install pillow`)
- Linux: `curl`, `make`, `bash`
- Windows: Visual Studio Build Tools, see [WINDOWS.md](WINDOWS.md)

## Build

Linux / Coursera GPU environment:

```bash
make setup
make build
```

Windows (from project root, x64 Native Tools prompt or after `build.bat` sets up MSVC):

```cmd
build.bat
```

Manual compile:

```bash
nvcc -Iinclude -Ilib --std=c++17 -O2 \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  src/main.cpp src/cli_parser.cpp src/image_processor.cu \
  -o bin/batch_image_processor.exe
```

## Generate test images

Default dataset: **MNIST** handwritten digits (UCI / LeCun), exported as **colorized RGB**
PNG files so the grayscale kernel has real color input to convert. The script downloads
the training set once and writes 128 images scaled to 256×256:

```bash
python3 scripts/prepare_mnist_dataset.py --output data/input --count 128
```

Alternative synthetic images (no download):

```bash
python3 scripts/generate_sample_images.py --output data/input --count 128
```

You can also place your own `.png` or `.jpg` files in the input directory.

## Run

Full pipeline (build, generate data if needed, process, log output):

```bash
bash run.sh
```

Windows:

```cmd
run.bat
```

Manual run:

```bash
./bin/batch_image_processor.exe \
  --input data/input \
  --output data/output \
  --log results/execution_log.txt \
  --mode both \
  --threads 256
```

### Command-line options

| Flag | Description | Default |
|------|-------------|---------|
| `--input <dir>` | Input image directory | `data/input` |
| `--output <dir>` | Output directory | `data/output` |
| `--log <file>` | Execution log path | `results/execution_log.txt` |
| `--mode <name>` | `grayscale`, `blur`, or `both` | `both` |
| `--threads <n>` | CUDA threads per block | `256` |
| `--max-images <n>` | Max images to process (`-1` = all) | `-1` |
| `--help` | Print usage | |

## Kernels

- `HorizontalBlurKernel` — 3-tap horizontal box blur on RGB, edges unchanged
- `RgbToGrayscaleKernel` — ITU-R BT.601 luma weights (0.299R + 0.587G + 0.114B)

Host code handles directory iteration, CUDA memory, and timing. GPU kernel
time and host I/O time are tracked separately in the execution log.

## Proof-of-execution package

After a GPU run:

```bash
python3 scripts/package_proof_artifacts.py --root .
```

Creates `cuda_proof_of_execution.zip` with logs, console capture, and labeled
before/after sample images for submission.

## Style

Code follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html):
`snake_case` functions, `kConstant` names, 2-space indent, include guards,
and `enum class` for processing modes.
