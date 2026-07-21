#include "/opt/aspida-new/gpu/vit_engine.cuh"
// Global C entries for the Ada engine.
extern "C" void aspida_vit(const float*pv,int gh,int gw,float*out){ viteng::aspida_vit_forward(pv,gh,gw,out); cudaDeviceSynchronize(); }
// Inject visual tokens into a prefill embedding buffer at image_pad positions.
// pHb[P,dim] (device); vtok[npos,dim] (device); pos[npos] (device) = token indices.
__global__ void k_vit_inject(float*pHb,const float*vtok,const int*pos,int npos,int dim){
  int i=blockIdx.x; if(i>=npos)return; int p=pos[i]; for(int d=threadIdx.x;d<dim;d+=blockDim.x) pHb[(size_t)p*dim+d]=vtok[(size_t)i*dim+d];
}
extern "C" void aspida_vit_inject(float*pHb,const float*vtok_dev,const int*pos_dev,int npos,int dim,void*stream){
  k_vit_inject<<<npos,256,0,(cudaStream_t)stream>>>(pHb,vtok_dev,pos_dev,npos,dim);
}

// ── Vision injection state for the prefill path ──────────────────────────────
static float* g_vis_vtok=nullptr; static int* g_vis_pos=nullptr; static int g_vis_npos=0,g_vis_active=0,g_vis_cap=0;
// vtok_host [npos,2048], pos_host [npos] = global token positions of <|image_pad|>.
extern "C" void aspida_gpu_set_vision(int npos,const int*pos_host,const float*vtok_host){
  const int DIM=2048;
  if(npos>g_vis_cap){ if(g_vis_vtok)cudaFree(g_vis_vtok); if(g_vis_pos)cudaFree(g_vis_pos);
    cudaMalloc(&g_vis_vtok,(size_t)npos*DIM*4); cudaMalloc(&g_vis_pos,(size_t)npos*4); g_vis_cap=npos; }
  cudaMemcpy(g_vis_vtok,vtok_host,(size_t)npos*DIM*4,cudaMemcpyHostToDevice);
  cudaMemcpy(g_vis_pos,pos_host,(size_t)npos*4,cudaMemcpyHostToDevice);
  g_vis_npos=npos; g_vis_active=1;
}
extern "C" void aspida_gpu_clear_vision(){ g_vis_active=0; }
__global__ void k_vis_inject_chunk(float*pHb,const float*vtok,const int*gpos,int npos,int pos_start,int P,int dim){
  int i=blockIdx.x; if(i>=npos)return; int local=gpos[i]-pos_start; if(local<0||local>=P)return;
  for(int d=threadIdx.x;d<dim;d+=blockDim.x) pHb[(size_t)local*dim+d]=vtok[(size_t)i*dim+d];
}
// Called inside prefill after the token-embed gather; no-op unless vision is set.
extern "C" void aspida_gpu_vision_inject_chunk(float*pHb,int pos_start,int P,int dim,void*stream){
  if(!g_vis_active||g_vis_npos<=0)return;
  k_vis_inject_chunk<<<g_vis_npos,256,0,(cudaStream_t)stream>>>(pHb,g_vis_vtok,g_vis_pos,g_vis_npos,pos_start,P,dim);
  cudaStreamSynchronize((cudaStream_t)stream);
}
