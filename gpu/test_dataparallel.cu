// test_dataparallel.cu — Step 7: data-parallel all-reduce CORRECTNESS, validated
// at $0 (two replicas in one process on one GPU, real sum-all-reduce of the
// gradient accumulators between micro and apply). Proves the distributed math:
// two nodes on different data shards + summed gradients == a single node trained
// on the combined data. The only thing a real two-droplet run adds is moving the
// accumulator blob over TCP (timing measured separately).
//
//   nvcc -O3 -arch=native test_dataparallel.cu -L. -laspidastudent -o test_dataparallel
//   LD_LIBRARY_PATH=. ./test_dataparallel

#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
  void* stu_create(int,int,int,int,int,int);
  void  stu_set_data(void*,const int*,const int*);
  float stu_micro(void*);
  void  stu_apply(void*,float,int);
  int   stu_nparams(void*);
  void  stu_get_acc(void*,float*);
  void  stu_set_acc(void*,const float*);
  void  stu_free(void*);
}

int main(){
  const int V=32,D=16,F=32,S=6,L=2,H=2;
  void* R0=stu_create(V,D,F,S,L,H);   // node 0
  void* R1=stu_create(V,D,F,S,L,H);   // node 1  (identical init — deterministic)
  void* Rr=stu_create(V,D,F,S,L,H);   // reference: single node, both shards
  int N=stu_nparams(R0);
  printf("data-parallel all-reduce: 2 replicas, %d params, summed grads\n", N);
  float *g0=(float*)malloc((size_t)N*4),*g1=(float*)malloc((size_t)N*4),*gs=(float*)malloc((size_t)N*4);

  const int id0[6]={3,7,3,11,20,7}, t0[6]={5,1,9,0,14,2};   // shard 0
  const int id1[6]={2,9,4,15,8,1},  t1[6]={6,3,2,11,7,0};   // shard 1
  const float lr=0.005f;

  for(int it=0;it<60;it++){
    stu_set_data(R0,id0,t0); stu_micro(R0);          // node 0 local grad
    stu_set_data(R1,id1,t1); stu_micro(R1);          // node 1 local grad
    stu_get_acc(R0,g0); stu_get_acc(R1,g1);          // ---- all-reduce (SUM) ----
    for(int i=0;i<N;i++) gs[i]=g0[i]+g1[i];
    stu_set_acc(R0,gs); stu_set_acc(R1,gs);          // both nodes get the global grad
    stu_apply(R0,lr,2); stu_apply(R1,lr,2);          // identical update -> stay in sync
    stu_set_data(Rr,id0,t0); stu_micro(Rr);          // reference: same two shards, one node
    stu_set_data(Rr,id1,t1); stu_micro(Rr);
    stu_apply(Rr,lr,2);
  }

  stu_set_data(R0,id0,t0); float l0=stu_micro(R0);
  stu_set_data(R1,id0,t0); float l1=stu_micro(R1);
  stu_set_data(Rr,id0,t0); float lr_=stu_micro(Rr);
  printf("  loss R0=%.6e  R1=%.6e  Rref=%.6e\n", l0,l1,lr_);
  float d01=fabsf(l0-l1), d0r=fabsf(l0-lr_);
  printf("  |R0-R1|=%.3e (replicas in sync)  |R0-Rref|=%.3e (DP == single-node)\n", d01, d0r);
  stu_free(R0); stu_free(R1); stu_free(Rr);
  bool ok = d01<1e-4f && d0r<1e-4f;
  printf("RESULT: %s (data-parallel all-reduce numerically == single-node on combined data)\n",
         ok?"PASS":"FAIL");
  return ok?0:1;
}
