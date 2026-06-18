// train_kernels2.cu — GPU backward kernels for softmax and RMSNorm, matching
// Train.Softmax/Attention and Train.RMSNorm_Backward exactly. Self-validating
// against a double-precision CPU reference.
//   nvcc -O3 -arch=native train_kernels2.cu -o train_kernels2 && ./train_kernels2

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// row-softmax Jacobian-vector: dS[r,n] = P[r,n] * (dP[r,n] - sum_k P[r,k]*dP[r,k])
__global__ void softmax_bwd(const float* P, const float* dP, float* dS,
                            int R, int N) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  float dot = 0.f;
  for (int n = 0; n < N; ++n) dot += P[r * N + n] * dP[r * N + n];
  for (int n = 0; n < N; ++n)
    dS[r * N + n] = P[r * N + n] * (dP[r * N + n] - dot);
}

// RMSNorm backward (per row), matching Train.RMSNorm_Backward:
//   ri = 1/sqrt(mean(x^2)+eps);  n_j = x_j*ri
//   dGamma_j += dy_j * n_j ; dn_j = dy_j*gamma_j ; c = sum_j dn_j*x_j
//   dx_j = ri*dn_j - (ri^3/D)*x_j*c
__global__ void rmsnorm_bwd(const float* X, const float* Gamma, const float* DY,
                            float* DX, float* DGamma, int R, int D, float eps) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= R) return;
  float ms = 0.f;
  for (int j = 0; j < D; ++j) { float v = X[r * D + j]; ms += v * v; }
  float ri = 1.f / sqrtf(ms / D + eps);
  float ri3 = ri * ri * ri;
  float c = 0.f;
  for (int j = 0; j < D; ++j) c += (DY[r * D + j] * Gamma[j]) * X[r * D + j];
  for (int j = 0; j < D; ++j) {
    float dn = DY[r * D + j] * Gamma[j];
    DX[r * D + j] = ri * dn - (ri3 / D) * X[r * D + j] * c;
    atomicAdd(&DGamma[j], DY[r * D + j] * (X[r * D + j] * ri));
  }
}

// ---- CPU references (double accumulation) ----
static void cpu_softmax_bwd(const float*P,const float*dP,float*dS,int R,int N){
  for(int r=0;r<R;r++){double dot=0;for(int n=0;n<N;n++)dot+=(double)P[r*N+n]*dP[r*N+n];
    for(int n=0;n<N;n++)dS[r*N+n]=(float)((double)P[r*N+n]*((double)dP[r*N+n]-dot));}
}
static void cpu_rmsnorm_bwd(const float*X,const float*G,const float*DY,float*DX,float*DG,int R,int D,float eps){
  for(int j=0;j<D;j++)DG[j]=0;
  for(int r=0;r<R;r++){double ms=0;for(int j=0;j<D;j++){double v=X[r*D+j];ms+=v*v;}
    double ri=1.0/sqrt(ms/D+eps), ri3=ri*ri*ri, c=0;
    for(int j=0;j<D;j++)c+=((double)DY[r*D+j]*G[j])*X[r*D+j];
    for(int j=0;j<D;j++){double dn=(double)DY[r*D+j]*G[j];
      DX[r*D+j]=(float)(ri*dn-(ri3/D)*X[r*D+j]*c);
      DG[j]+=(float)((double)DY[r*D+j]*((double)X[r*D+j]*ri));}}
}
static double mrel(const float*a,const float*b,int n){double md=0,mb=0;
  for(int i=0;i<n;i++){double d=fabs((double)a[i]-b[i]);if(d>md)md=d;double bb=fabs((double)b[i]);if(bb>mb)mb=bb;}
  return md/(mb+1e-12);}
static void fill(float*p,int n){for(int i=0;i<n;i++)p[i]=(float)((rand()/(double)RAND_MAX)*2.0-1.0);}
static void fillpos(float*p,int n){for(int i=0;i<n;i++)p[i]=(float)(0.2+rand()/(double)RAND_MAX);} // gamma>0

static int t_softmax(int R,int N){
  // build a valid prob matrix P (softmax of random logits) on host
  int sz=R*N; float*P=(float*)malloc(sz*4),*dP=(float*)malloc(sz*4),*hS=(float*)malloc(sz*4),*rS=(float*)malloc(sz*4);
  for(int r=0;r<R;r++){double m=-1e30,s=0;float*row=P+r*N;
    for(int n=0;n<N;n++){row[n]=(float)((rand()/(double)RAND_MAX)*4-2);}
    for(int n=0;n<N;n++)if(row[n]>m)m=row[n];
    for(int n=0;n<N;n++){row[n]=(float)exp(row[n]-m);s+=row[n];}
    for(int n=0;n<N;n++)row[n]/=s;}
  fill(dP,sz);
  float*dPp,*Pp,*dSp; cudaMalloc(&Pp,sz*4);cudaMalloc(&dPp,sz*4);cudaMalloc(&dSp,sz*4);
  cudaMemcpy(Pp,P,sz*4,cudaMemcpyHostToDevice);cudaMemcpy(dPp,dP,sz*4,cudaMemcpyHostToDevice);
  softmax_bwd<<<(R+127)/128,128>>>(Pp,dPp,dSp,R,N);cudaDeviceSynchronize();
  cudaMemcpy(hS,dSp,sz*4,cudaMemcpyDeviceToHost);
  cpu_softmax_bwd(P,dP,rS,R,N); double e=mrel(hS,rS,sz);
  printf("  softmax_bwd [%3dx%3d]  err=%.2e  %s\n",R,N,e,e<1e-3?"OK":"FAIL");
  cudaFree(Pp);cudaFree(dPp);cudaFree(dSp);free(P);free(dP);free(hS);free(rS);
  return e<1e-3;
}
static int t_rms(int R,int D){
  int szX=R*D; float*X=(float*)malloc(szX*4),*DY=(float*)malloc(szX*4),*G=(float*)malloc(D*4);
  float*hDX=(float*)malloc(szX*4),*rDX=(float*)malloc(szX*4),*hDG=(float*)malloc(D*4),*rDG=(float*)malloc(D*4);
  fill(X,szX);fill(DY,szX);fillpos(G,D);
  float*Xp,*DYp,*Gp,*DXp,*DGp; cudaMalloc(&Xp,szX*4);cudaMalloc(&DYp,szX*4);cudaMalloc(&Gp,D*4);
  cudaMalloc(&DXp,szX*4);cudaMalloc(&DGp,D*4);
  cudaMemcpy(Xp,X,szX*4,cudaMemcpyHostToDevice);cudaMemcpy(DYp,DY,szX*4,cudaMemcpyHostToDevice);
  cudaMemcpy(Gp,G,D*4,cudaMemcpyHostToDevice);cudaMemset(DGp,0,D*4);
  rmsnorm_bwd<<<(R+127)/128,128>>>(Xp,Gp,DYp,DXp,DGp,R,D,1e-6f);cudaDeviceSynchronize();
  cudaMemcpy(hDX,DXp,szX*4,cudaMemcpyDeviceToHost);cudaMemcpy(hDG,DGp,D*4,cudaMemcpyDeviceToHost);
  cpu_rmsnorm_bwd(X,G,DY,rDX,rDG,R,D,1e-6f);
  double e1=mrel(hDX,rDX,szX),e2=mrel(hDG,rDG,D);
  printf("  rmsnorm_bwd [%3dx%3d]  dX=%.2e dGamma=%.2e  %s\n",R,D,e1,e2,(e1<1e-3&&e2<1e-3)?"OK":"FAIL");
  cudaFree(Xp);cudaFree(DYp);cudaFree(Gp);cudaFree(DXp);cudaFree(DGp);
  free(X);free(DY);free(G);free(hDX);free(rDX);free(hDG);free(rDG);
  return e1<1e-3&&e2<1e-3;
}
int main(){srand(7);
  printf("=== Aspida GPU backward: softmax + RMSNorm vs CPU ===\n");
  int ok=1;
  ok&=t_softmax(64,96); ok&=t_softmax(20,49);
  ok&=t_rms(128,64); ok&=t_rms(17,33);
  printf(ok?"\nRESULT: PASS (GPU == CPU)\n":"\nRESULT: FAIL\n");
  return ok?0:1;
}
