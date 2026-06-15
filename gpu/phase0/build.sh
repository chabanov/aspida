#!/usr/bin/env bash
# Phase-0 FFI build: nvcc compiles the CUDA kernel, gnatmake compiles the Ada
# main and links it against the CUDA object + the CUDA runtime.
#
# Requires: nvcc on PATH (DO "NVIDIA AI/ML Ready" image), and the Alire GNAT
# toolchain (gnatmake) on PATH.
set -euo pipefail

CUDA="${CUDA_HOME:-/usr/local/cuda}"
LIBDIR="$CUDA/lib64"

echo ">> nvcc: $(command -v nvcc) ($("$CUDA"/bin/nvcc --version 2>/dev/null | tail -1 || nvcc --version | tail -1))"
echo ">> gnatmake: $(command -v gnatmake)"

nvcc -O2 -c vadd.cu -o vadd.o
gnatmake -O2 vadd_main.adb -largs vadd.o -L"$LIBDIR" -lcudart -lstdc++

echo ">> running:"
./vadd_main
