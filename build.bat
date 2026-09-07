@echo off
setlocal
cd /d "%~dp0"

if not exist lib\stb_image.h (
  echo stb headers missing in lib\
  echo Run scripts\fetch_dependencies.sh from Git Bash, or copy stb_image.h manually.
  exit /b 1
)

if not exist bin mkdir bin

set "CUDA_PATH="
for /f "delims=" %%i in ('dir /b /ad "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*" 2^>nul') do (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i"
    goto :cuda_found
  )
)
for /f "delims=" %%i in ('dir /b /ad /o-n "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
  if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i\bin\nvcc.exe" (
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i"
    goto :cuda_found
  )
)
echo CUDA toolkit not found. Install from https://developer.nvidia.com/cuda-downloads
exit /b 1

:cuda_found
set "PATH=%CUDA_PATH%\bin;%PATH%"

for /f "tokens=*" %%v in ('nvcc --version ^| findstr /C:"release"') do set NVCC_RELEASE=%%v
echo Using %CUDA_PATH%
echo %NVCC_RELEASE%

set "ARCH_FLAGS=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86"
echo %NVCC_RELEASE% | findstr /C:"release 12" >nul && (
  set "ARCH_FLAGS=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86"
)
echo %NVCC_RELEASE% | findstr /C:"release 11" >nul && (
  set "ARCH_FLAGS=-gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75"
)

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
  echo Visual Studio Build Tools not found.
  exit /b 1
)

echo === Building ===
nvcc -Iinclude -Ilib -I"%CUDA_PATH%\include" --std=c++17 -O2 %ARCH_FLAGS% ^
  src\main.cpp src\cli_parser.cpp src\image_processor.cu ^
  -o bin\batch_image_processor.exe
if errorlevel 1 exit /b 1

echo Build OK: bin\batch_image_processor.exe
endlocal
