#!/bin/bash
# Build the isolated image daemon. It dynamically links libaspida_imggen.so
# (the sd.cpp wrapper), so sd.cpp lives ONLY in this process — never in the
# LLM engine. Run build_imggen.sh first so libaspida_imggen.so exists.
set -e
D=/opt/imggen-native
cd "$D"
gcc -O2 -std=c11 -o aspida_imgd "$D/aspida_imgd.c" \
  -L"$D" -laspida_imggen -Wl,-rpath,"$D" \
  -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 \
  -lpthread -ldl -lm
echo "built $D/aspida_imgd"
ldd "$D/aspida_imgd" | grep -iE "imggen|cuda" | head
