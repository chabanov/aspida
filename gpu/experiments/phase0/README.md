# Stream B · Phase 0 — Ada ↔ C ↔ CUDA FFI proof

Proves the foundation for the GPU backend: the pure-Ada engine can drive its own
CUDA kernels over the C ABI. A self-contained CUDA vector-add (`vadd.cu`,
kernel + host wrapper) is called from Ada (`vadd_main.adb`) via `pragma Import`.

## Result (validated)
Run on a DigitalOcean **NVIDIA H200** droplet (CUDA 13.1, driver 590.48, GNAT
14.2.1 via Alire):

```
gcc -c -O2 vadd_main.adb
gnatbind -x vadd_main.ali
gnatlink vadd_main.ali -O2 vadd.o -L/usr/local/cuda/lib64 -lcudart -lstdc++
c[0]    = 0.00000E+00
c[1023] = 3.06900E+03  (expected 3069.0)
FFI OK: Ada -> C -> CUDA vector add correct on GPU
```

So the toolchains coexist and the Ada → C → CUDA path (alloc / H2D / launch /
D2H) works end to end. This `vadd` host wrapper is the seam real kernels
(quant-GEMM, attention, RMSNorm, RoPE, SwiGLU) plug into in Phase 1.

## How to reproduce
On an NVIDIA GPU box (DO image "NVIDIA AI/ML Ready" = image slug
`gpu-h100x1-base`) with nvcc + the Alire GNAT toolchain on PATH:

```
bash build.sh
```

## Notes
- 4000 Ada ($0.76/hr) and other cheap GPUs were not deployable at the time
  (no region capacity); H200/nyc2 was the only available NVIDIA size. Retry the
  cheaper sizes for routine Phase-1 dev (`doctl compute size list`).
- These files are CUDA-specific and are **not** part of the macOS/Linux gpr
  builds — they are Stream-B scaffolding, built standalone via `build.sh`.
- See `../../docs/gpu_linux_plan.md` for the full plan and cost model.
