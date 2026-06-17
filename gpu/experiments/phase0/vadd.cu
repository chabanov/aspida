// Phase-0 FFI proof: a self-contained CUDA kernel + C-ABI host wrapper that
// Ada calls via pragma Import. Vector add c = a + b on the GPU.
//
// Compile:  nvcc -O2 -c vadd.cu -o vadd.o
// The Ada side links vadd.o with -lcudart (see build.sh).
#include <cuda_runtime.h>

extern "C" {

__global__ void vadd_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Host wrapper (C ABI): alloc device buffers, H2D copy, launch, D2H copy.
// This is the seam the Ada engine will call; real kernels (quant-GEMM,
// attention) replace vadd_kernel later.
void vadd(const float *a, const float *b, float *c, int n) {
    float *da = 0, *db = 0, *dc = 0;
    size_t sz = (size_t) n * sizeof(float);
    cudaMalloc((void **) &da, sz);
    cudaMalloc((void **) &db, sz);
    cudaMalloc((void **) &dc, sz);
    cudaMemcpy(da, a, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sz, cudaMemcpyHostToDevice);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vadd_kernel<<<blocks, threads>>>(da, db, dc, n);
    cudaDeviceSynchronize();
    cudaMemcpy(c, dc, sz, cudaMemcpyDeviceToHost);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

}
