#!/bin/bash
# pipefail so a compile error piped through `tail` still fails the build
# (else g++ silently links a stale .o and "succeeds" with old code).
set -e
set -o pipefail
SD=/root/fattn_bench/stable-diffusion.cpp
B=$SD/build-pic
export PATH=/usr/local/cuda/bin:$PATH
cd /opt/imggen-native

cat > exports.map <<'MAP'
{ global: aspida_img_*; local: *; };
MAP

nvcc -c -O3 -std=c++17 -Xcompiler -fPIC \
  -I$SD -I$SD/include -I$SD/thirdparty -I$SD/ggml/include \
  aspida_imggen.cpp -o aspida_imggen.o 2>&1 | tail -20

# whole-archive on ALL static archives so ggml backend registrations
# (ggml_backend_cpu_reg / cuda_reg, pulled only via the registry) are kept.
g++ -shared -fPIC -o libaspida_imggen.so \
  -Wl,--version-script=exports.map \
  aspida_imggen.o \
  -Wl,--whole-archive \
  $B/libstable-diffusion.a \
  $B/ggml/src/ggml-cuda/libggml-cuda.a \
  $B/ggml/src/libggml-cpu.a \
  $B/ggml/src/libggml-base.a \
  $B/ggml/src/libggml.a \
  -Wl,--no-whole-archive \
  -L/usr/local/cuda/lib64 -lcudart -lcublas -lcuda \
  -lpthread -ldl -lm -lgomp -lstdc++ 2>&1 | tail -25

echo "=== exported (should be ONLY aspida_img_*): ==="
nm -D --defined-only libaspida_imggen.so | grep " T " | head
echo "ggml leaked: $(nm -D --defined-only libaspida_imggen.so | grep -c ' T ggml_')"
echo "aspida_img exported: $(nm -D --defined-only libaspida_imggen.so | grep -c ' T aspida_img_')"
