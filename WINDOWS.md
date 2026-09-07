# Windows Setup

## Requirements

1. NVIDIA GPU with current driver (`nvidia-smi`)
2. [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) — verify with `nvcc --version`
3. Python 3 — `pip install -r requirements.txt`
4. Visual Studio 2022 Build Tools (C++ workload)

## Quick start

```cmd
cd path\to\cuda-batch-image-processor
build.bat
run.bat
```

`run.bat` generates 128 test images, runs the GPU processor, and creates
`cuda_proof_of_execution.zip`.

## Manual commands

```cmd
mkdir bin data\input data\output results

python scripts\prepare_mnist_dataset.py --output data/input --count 128

bin\batch_image_processor.exe --input data/input --output data/output ^
  --log results/execution_log.txt --mode both --threads 256

python scripts\package_proof_artifacts.py --root .
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `nvcc` not found | Reopen terminal after CUDA install; check PATH |
| Build errors | Use x64 Native Tools Command Prompt for VS |
| Black output images | GTX 1070 Ti needs **CUDA 12.x** (CUDA 13 dropped Pascal). Run **`cuda_12.8.1_installer.exe` as Administrator** from the project folder, then `build.bat` and `run.bat` |
| `python` not found | Reinstall Python with "Add to PATH" |

Repository: https://github.com/AyushGuha06/CudaCourse2026
