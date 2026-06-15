// Stream B · Phase 1 — Q4_K GPU kernels (dequant + quant matvec), mirroring the
// CPU engine's LLM_Dequant.Dequant_Q4_K / QMatVec exactly so results can be
// validated against it.
//
//   block_q4_K = 144 bytes / 256 elems:
//     d(fp16,2) dmin(fp16,2) scales[12] qs[128]
//   value(group g, lane l) = d*sc*(nibble) - dmin*m   (min subtracted)
//   weight tensor: ne0 = in_dim, ne1 = out_dim; row-major over out rows,
//   each row = (in_dim/256) blocks. y[o] = sum_i dequant(row o)[i] * x[i].
//
// Build & self-test:  see build.sh   (reads reference files produced by
// gen_ref.adb on the CPU engine and checks the GPU output against them).
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ float f16(const uint8_t *p) {
    __half h;
    *reinterpret_cast<uint16_t *>(&h) = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    return __half2float(h);
}

// get_scale_min_k4 (llama.cpp): extract 6-bit scale d and min m for sub-block j.
__device__ __forceinline__ void gsm(const uint8_t *sc, int j, int *d, int *m) {
    if (j < 4) {
        *d = sc[j] & 63;
        *m = sc[j + 4] & 63;
    } else {
        *d = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *m = (sc[j + 4] >> 4)   | ((sc[j]     >> 6) << 4);
    }
}

// Dequantize one 144-byte super-block into 256 floats (block layout order).
__device__ void deq_block(const uint8_t *b, float *out) {
    float d    = f16(b);
    float dmin = f16(b + 2);
    const uint8_t *sc = b + 4;
    const uint8_t *qs = b + 16;
    for (int g = 0; g < 4; ++g) {
        int s1, m1, s2, m2;
        gsm(sc, 2 * g,     &s1, &m1);
        gsm(sc, 2 * g + 1, &s2, &m2);
        float d1 = d * s1, mm1 = dmin * m1;
        float d2 = d * s2, mm2 = dmin * m2;
        const uint8_t *q = qs + g * 32;
        for (int l = 0; l < 32; ++l) out[64 * g + l]      = d1 * (q[l] & 0x0F) - mm1;
        for (int l = 0; l < 32; ++l) out[64 * g + 32 + l] = d2 * (q[l] >> 4)   - mm2;
    }
}

__global__ void k_dequant(const uint8_t *blocks, float *out, int nblocks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nblocks) deq_block(blocks + (size_t) i * 144, out + (size_t) i * 256);
}

// One thread per output row: dequant the row's blocks on the fly, ascending dot
// with x (matches the CPU scalar reduction order; --fmad=false keeps rounding).
__global__ void k_matvec(const uint8_t *w, const float *x, float *y,
                         int in_dim, int out_dim) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    int nblk = in_dim / 256;
    size_t bpr = (size_t) nblk * 144;
    const uint8_t *row = w + (size_t) o * bpr;
    float tmp[256];
    float acc = 0.0f;
    for (int blk = 0; blk < nblk; ++blk) {
        deq_block(row + (size_t) blk * 144, tmp);
        int base = blk * 256;
        for (int l = 0; l < 256; ++l) acc += tmp[l] * x[base + l];
    }
    y[o] = acc;
}

extern "C" void q4k_dequant_host(const uint8_t *blocks, float *out, int nblocks) {
    uint8_t *db; float *dout;
    cudaMalloc(&db, (size_t) nblocks * 144);
    cudaMalloc(&dout, (size_t) nblocks * 256 * sizeof(float));
    cudaMemcpy(db, blocks, (size_t) nblocks * 144, cudaMemcpyHostToDevice);
    int t = 64, b = (nblocks + t - 1) / t;
    k_dequant<<<b, t>>>(db, dout, nblocks);
    cudaDeviceSynchronize();
    cudaMemcpy(out, dout, (size_t) nblocks * 256 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(db); cudaFree(dout);
}

extern "C" void q4k_matvec_host(const uint8_t *w, const float *x, float *y,
                                int in_dim, int out_dim) {
    size_t wbytes = (size_t) (in_dim / 256) * 144 * out_dim;
    uint8_t *dw; float *dx, *dy;
    cudaMalloc(&dw, wbytes);
    cudaMalloc(&dx, (size_t) in_dim * sizeof(float));
    cudaMalloc(&dy, (size_t) out_dim * sizeof(float));
    cudaMemcpy(dw, w, wbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, (size_t) in_dim * sizeof(float), cudaMemcpyHostToDevice);
    int t = 128, b = (out_dim + t - 1) / t;
    k_matvec<<<b, t>>>(dw, dx, dy, in_dim, out_dim);
    cudaDeviceSynchronize();
    cudaMemcpy(y, dy, (size_t) out_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dw); cudaFree(dx); cudaFree(dy);
}

// ---- self-test harness: compare GPU kernels against CPU-engine references ----
static void *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(sz);
    if (fread(p, 1, sz, f) != (size_t) sz) { exit(2); }
    fclose(f); if (n) *n = sz; return p;
}

int main(int argc, char **argv) {
    int in_dim = argc > 1 ? atoi(argv[1]) : 2560;
    int out_dim = argc > 2 ? atoi(argv[2]) : 512;
    int rc = 0;

    // 1) dequant: one block vs exact CPU reference (bit-exact expected).
    {
        size_t nb; uint8_t *blk = (uint8_t *) slurp("dq_in.bin", &nb);
        float *exp = (float *) slurp("dq_exp.bin", 0);
        float *got = (float *) malloc(256 * sizeof(float));
        q4k_dequant_host(blk, got, 1);
        double maxd = 0; int bad = 0;
        for (int i = 0; i < 256; ++i) {
            double d = fabs((double) got[i] - exp[i]);
            if (d > maxd) maxd = d;
            if (d > 0.0) bad++;
        }
        printf("[dequant] max|gpu-cpu|=%.3e exact_mismatches=%d/256 -> %s\n",
               maxd, bad, maxd == 0.0 ? "BIT-EXACT" : (maxd < 1e-5 ? "OK(~)" : "FAIL"));
        if (maxd >= 1e-5) rc = 1;
        free(blk); free(exp); free(got);
    }

    // 2) matvec: real Q4_K tensor x deterministic vector vs CPU MatVec (tol).
    {
        size_t wn; uint8_t *w = (uint8_t *) slurp("mv_w.bin", &wn);
        float *x = (float *) slurp("mv_x.bin", 0);
        float *exp = (float *) slurp("mv_y.bin", 0);
        float *got = (float *) malloc((size_t) out_dim * sizeof(float));
        q4k_matvec_host(w, x, got, in_dim, out_dim);
        double maxa = 0, maxr = 0;
        for (int o = 0; o < out_dim; ++o) {
            double a = fabs((double) got[o] - exp[o]);
            double r = a / (fabs((double) exp[o]) + 1e-6);
            if (a > maxa) maxa = a; if (r > maxr) maxr = r;
        }
        printf("[matvec ] out=%d in=%d  max_abs=%.3e max_rel=%.3e -> %s\n",
               out_dim, in_dim, maxa, maxr, maxr < 1e-3 ? "OK" : "FAIL");
        if (maxr >= 1e-3) rc = 1;
        free(w); free(x); free(exp); free(got);
    }

    printf(rc == 0 ? "PHASE1 KERNELS: PASS\n" : "PHASE1 KERNELS: FAIL\n");
    return rc;
}
