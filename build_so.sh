#!/bin/bash
set -e
GG=/root/fattn_bench/llama.cpp
cd /root/aspida-integ
nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu/gpu_matvec.cu \
  -o libaspidagpu.so \
  -I$GG/ggml/include \
  -L$GG/build/bin -lggml -lggml-base -lggml-cuda \
  -Xlinker -rpath -Xlinker "\$ORIGIN"
