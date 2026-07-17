#!/bin/bash
# Build libaspidagpu.so (the CUDA shim).
#
# Links against upstream llama.cpp's ggml for two prefill paths: fattn-mma
# (fattn_ggml.cuh) and mul_mat_id (moe_ggml.cuh). Stock upstream, no patches.
#
# Pinned dependency — prod (the GPU host, an NVIDIA GPU) links this exact tree:
#   git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
#   git checkout 505b1ed15ca80e2a19f12ff4ac365e40fb374053
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
#         -DCMAKE_CUDA_ARCHITECTURES=89 -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
#   cmake --build build -j
# CUDA_ARCHITECTURES=89 is the an NVIDIA GPU (Ada); set it to your card's SM.
# ggml's CUDA API drifts, so a different commit may not compile — repin
# deliberately, don't float. Point GG at the tree; it need not be on this box.
set -e
GG=${GG:-/root/fattn_bench/llama.cpp}
cd "$(dirname "$0")"
nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu/gpu_matvec.cu \
  -o libaspidagpu.so \
  -I$GG/ggml/include \
  -L$GG/build/bin -lggml -lggml-base -lggml-cuda \
  -Xlinker -rpath -Xlinker "\$ORIGIN"
