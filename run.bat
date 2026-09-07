@echo off
setlocal
cd /d "%~dp0"

if not exist bin\batch_image_processor.exe (
  echo Run build.bat first.
  exit /b 1
)

mkdir data\input 2>nul
mkdir data\output 2>nul
mkdir results 2>nul

echo === Preparing 128 MNIST images ===
python scripts\prepare_mnist_dataset.py --output data/input --count 128
if errorlevel 1 exit /b 1

echo === Running GPU batch processor ===
bin\batch_image_processor.exe --input data/input --output data/output --log results/execution_log.txt --mode both --threads 256 > results\console_output.txt 2>&1
type results\console_output.txt

echo.
echo === Packaging proof artifacts ===
python scripts\package_proof_artifacts.py --root .
if errorlevel 1 exit /b 1

echo.
echo DONE. Upload cuda_proof_of_execution.zip to Coursera.
echo Log: results\execution_log.txt
endlocal
