# Stream B — GPU backend for the Aspida engine

The pure-Ada engine offloads its quantized mat-vecs (the ~95% decode cost) to
own CUDA kernels, while keeping all orchestration (GGUF load, tokenizer, the
forward graph, sampling, chat templates, crypto) in Ada. Validated end to end
on real Llama-3.3-70B.

## Status (validated on DigitalOcean NVIDIA H100/H200)
- **Kernels** (`experiments/phase1/`): Q4_K + Q6_K dequant & matvec (bit-exact vs
  CPU), RMSNorm / RoPE / SiLU / GQA-attention (≤1e-6), full transformer layer
  (2.3e-7), throughput ~2.25 tok/s un-tiled (~180× CPU at 70B scale). See
  `experiments/phase1/README.md`.
- **Full integration** (`gpu_matvec.cu` + `src/llm/llm_gpu.{ads,adb}`): the Ada
  `LLM_Llama` forward routes its 8 matvecs per layer through a CUDA shim
  (`aspida_gpu_matvec`) loaded at runtime via `dlopen` — so a CPU build/host is
  completely unaffected (no link-time CUDA dependency; `LLM_GPU.Available` is
  False ⇒ pure-Ada fallback). Weights are uploaded to VRAM once (cached by host
  pointer) and stay resident.

  **End-to-end result** — `ASPIDA_GPU=1 ./obj/llama_probe model.gguf "The capital of France is"`:
  ```
  completion: ' Paris .
  A. The Eiffel'
  wall clock (load + 8 tokens): 51 s   vs   CPU ~17 min / 4 tokens
  ```
  Correct output (identical to CPU — the kernels are bit-exact), ~20× faster
  end-to-end on the real engine.

## How to run (on an NVIDIA GPU box, image `gpu-h100x1-base`)
```
nvcc -O3 --fmad=false -arch=native -shared -Xcompiler -fPIC gpu/gpu_matvec.cu -o libaspidagpu.so
gprbuild -P probe.gpr -XOS=linux            # build the engine (Alire GNAT)
ASPIDA_GPU=1 ASPIDA_GPU_LIB=$PWD/libaspidagpu.so LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./obj/llama_probe model.gguf "The capital of France is" 8
```
Unset `ASPIDA_GPU` → pure CPU (unchanged).

## Next (perf)
The matvecs go one-at-a-time with a per-call x↑/y↓ round-trip. The big levers:
keep activations resident on the device and run a whole layer's matmuls without
host round-trips; tiled/batched quant-GEMM + FP16; move RMSNorm/RoPE/attention
on-device too. Correctness is already locked (bit-exact kernels); this is pure
throughput work.

Total GPU spend bringing Stream B to here: ~$12 (all boxes hourly, destroyed
after each run; DO GPU capacity fluctuates — a bounded create-retry loop lands one).
