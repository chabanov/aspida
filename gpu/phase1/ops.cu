// Stream B · Phase 1 (increment 2) — RMSNorm, RoPE, SiLU, attention GPU kernels.
// RMSNorm/RoPE checked against CPU-engine fixtures (gen_ref.adb); SiLU and the
// single-token GQA attention checked against an in-harness C reference.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

// ---- RMSNorm: y = x / sqrt(mean(x^2)+eps) * w  (eps 1e-6, ascending sum) ----
__global__ void k_rmsnorm(const float *x, const float *w, float *y, int n) {
    double ss = 0.0;                  // single-thread ascending reduction (matches CPU)
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    float rms = sqrtf((float) (ss / n) + 1e-6f);
    for (int i = 0; i < n; ++i) y[i] = x[i] / rms * w[i];
}

// ---- RoPE (NeoX): pair i with i+dim/2; theta = pos / base^(2i/dim) / ff[i] ----
__global__ void k_rope(const float *x, const float *ff, float *y,
                       int dim, int pos, float base) {
    int half = dim / 2;
    for (int i = 0; i < half; ++i) {
        float theta = (float) pos / powf(base, (float) (2 * i) / (float) dim);
        theta /= ff[i];
        float c = cosf(theta), s = sinf(theta);
        float x1 = x[i], x2 = x[i + half];
        y[i]        = x1 * c - x2 * s;
        y[i + half] = x2 * c + x1 * s;
    }
}

__global__ void k_silu(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + expf(-x[i]));
}

// ---- single-token GQA causal attention (decode step) ----
// Q[nh*hd], Kc/Vc[seq*nkv*hd]; head h uses kv head h/(nh/nkv). scale applied to
// scores; softmax over all seq positions (cache = past+current). O[nh*hd].
__global__ void k_attn(const float *Q, const float *Kc, const float *Vc, float *O,
                       int nh, int nkv, int hd, int seq, float scale) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= nh) return;
    int kv = h / (nh / nkv);
    const float *q = Q + h * hd;
    extern __shared__ float sh[];   // not used per-thread; keep simple w/ local
    float scores[4096];             // seq <= 4096 for this test
    float mx = -1e30f;
    for (int s = 0; s < seq; ++s) {
        const float *k = Kc + ((size_t) s * nkv + kv) * hd;
        float dp = 0.0f;
        for (int j = 0; j < hd; ++j) dp += q[j] * k[j];
        dp *= scale;
        scores[s] = dp;
        if (dp > mx) mx = dp;
    }
    float den = 0.0f;
    for (int s = 0; s < seq; ++s) { scores[s] = expf(scores[s] - mx); den += scores[s]; }
    for (int j = 0; j < hd; ++j) {
        float acc = 0.0f;
        for (int s = 0; s < seq; ++s) {
            const float *v = Vc + ((size_t) s * nkv + kv) * hd;
            acc += scores[s] / den * v[j];
        }
        O[h * hd + j] = acc;
    }
}

// ----------------------------------------------------------------------------
static float *slurp(const char *p, int *n) {
    FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *b = (float *) malloc(sz);
    if (fread(b, 1, sz, f) != (size_t) sz) exit(2);
    fclose(f); if (n) *n = (int) (sz / 4); return b;
}
static float maxabs(const float *a, const float *b, int n, double *mr) {
    double ma = 0, mrr = 0;
    for (int i = 0; i < n; ++i) {
        double d = fabs((double) a[i] - b[i]);
        if (d > ma) ma = d;
        double r = d / (fabs((double) b[i]) + 1e-6);
        if (r > mrr) mrr = r;
    }
    if (mr) *mr = mrr; return (float) ma;
}

int main() {
    int rc = 0;

    // RMSNorm vs CPU fixture.
    {
        int n; float *x = slurp("rms_x.bin", &n);
        float *w = slurp("rms_w.bin", 0), *exp = slurp("rms_y.bin", 0);
        float *got = (float *) malloc(n * 4), *dx, *dw, *dy;
        cudaMalloc(&dx, n*4); cudaMalloc(&dw, n*4); cudaMalloc(&dy, n*4);
        cudaMemcpy(dx, x, n*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w, n*4, cudaMemcpyHostToDevice);
        k_rmsnorm<<<1,1>>>(dx, dw, dy, n);
        cudaDeviceSynchronize(); cudaMemcpy(got, dy, n*4, cudaMemcpyDeviceToHost);
        double mr; float ma = maxabs(got, exp, n, &mr);
        printf("[rmsnorm] n=%d max_abs=%.3e max_rel=%.3e -> %s\n", n, ma, mr, mr<1e-4?"OK":"FAIL");
        if (mr>=1e-4) rc=1;
    }
    // RoPE vs CPU fixture (base 5e5, freq_factors).
    {
        int n; float *x = slurp("rope_in.bin", &n);
        float *ff = slurp("rope_ff.bin", 0), *exp = slurp("rope_out.bin", 0);
        float *got = (float *) malloc(n*4), *dx, *dff, *dy;
        cudaMalloc(&dx, n*4); cudaMalloc(&dff, (n/2)*4); cudaMalloc(&dy, n*4);
        cudaMemcpy(dx, x, n*4, cudaMemcpyHostToDevice);
        cudaMemcpy(dff, ff, (n/2)*4, cudaMemcpyHostToDevice);
        k_rope<<<1,1>>>(dx, dff, dy, n, 7, 500000.0f);
        cudaDeviceSynchronize(); cudaMemcpy(got, dy, n*4, cudaMemcpyDeviceToHost);
        double mr; float ma = maxabs(got, exp, n, &mr);
        printf("[rope   ] dim=%d max_abs=%.3e max_rel=%.3e -> %s\n", n, ma, mr, mr<1e-3?"OK":"FAIL");
        if (mr>=1e-3) rc=1;
    }
    // SiLU vs C reference.
    {
        int n = 4096; float *x=(float*)malloc(n*4), *exp=(float*)malloc(n*4), *got=(float*)malloc(n*4);
        for (int i=0;i<n;++i){ x[i]=((i%21)-10)*0.3f; exp[i]=x[i]/(1.0f+expf(-x[i])); }
        float *dx,*dy; cudaMalloc(&dx,n*4); cudaMalloc(&dy,n*4);
        cudaMemcpy(dx,x,n*4,cudaMemcpyHostToDevice);
        k_silu<<<(n+127)/128,128>>>(dx,dy,n);
        cudaDeviceSynchronize(); cudaMemcpy(got,dy,n*4,cudaMemcpyDeviceToHost);
        double mr; float ma=maxabs(got,exp,n,&mr);
        printf("[silu   ] n=%d max_abs=%.3e -> %s\n", n, ma, ma<1e-5?"OK":"FAIL");
        if (ma>=1e-5) rc=1;
    }
    // Attention vs C reference (GQA 8/2, hd 128, seq 40).
    {
        int nh=8, nkv=2, hd=128, seq=40; float scale=1.0f/sqrtf((float)hd);
        int q_n=nh*hd, kv_n=seq*nkv*hd;
        float *Q=(float*)malloc(q_n*4), *K=(float*)malloc(kv_n*4), *V=(float*)malloc(kv_n*4);
        float *exp=(float*)malloc(q_n*4), *got=(float*)malloc(q_n*4);
        for(int i=0;i<q_n;++i) Q[i]=sinf(0.01f*i);
        for(int i=0;i<kv_n;++i){ K[i]=cosf(0.013f*i); V[i]=sinf(0.007f*i); }
        // C reference
        for(int h=0;h<nh;++h){ int kv=h/(nh/nkv); const float*q=Q+h*hd;
            float sc[64], mx=-1e30f;
            for(int s=0;s<seq;++s){ const float*k=K+((size_t)s*nkv+kv)*hd; float dp=0; for(int j=0;j<hd;++j) dp+=q[j]*k[j]; dp*=scale; sc[s]=dp; if(dp>mx)mx=dp; }
            float den=0; for(int s=0;s<seq;++s){ sc[s]=expf(sc[s]-mx); den+=sc[s]; }
            for(int j=0;j<hd;++j){ float a=0; for(int s=0;s<seq;++s){ const float*v=V+((size_t)s*nkv+kv)*hd; a+=sc[s]/den*v[j]; } exp[h*hd+j]=a; } }
        float *dQ,*dK,*dV,*dO; cudaMalloc(&dQ,q_n*4); cudaMalloc(&dK,kv_n*4); cudaMalloc(&dV,kv_n*4); cudaMalloc(&dO,q_n*4);
        cudaMemcpy(dQ,Q,q_n*4,cudaMemcpyHostToDevice); cudaMemcpy(dK,K,kv_n*4,cudaMemcpyHostToDevice); cudaMemcpy(dV,V,kv_n*4,cudaMemcpyHostToDevice);
        k_attn<<<1,nh>>>(dQ,dK,dV,dO,nh,nkv,hd,seq,scale);
        cudaDeviceSynchronize(); cudaMemcpy(got,dO,q_n*4,cudaMemcpyDeviceToHost);
        double mr; float ma=maxabs(got,exp,q_n,&mr);
        // Outputs are O(1); judge by ABS error (rel blows up on near-zero elems).
        int ok = ma<1e-4;
        printf("[attn   ] nh=%d nkv=%d hd=%d seq=%d max_abs=%.3e max_rel=%.3e -> %s\n", nh,nkv,hd,seq,ma,mr,ok?"OK":"FAIL");
        if (!ok) rc=1;
    }
    printf(rc==0?"PHASE1 OPS: PASS\n":"PHASE1 OPS: FAIL\n");
    return rc;
}
