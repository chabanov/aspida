#include "/opt/aspida-new/gpu/vit_engine.cuh"
// Global C entries for the Ada engine.
extern "C" void aspida_vit(const float*pv,int gh,int gw,float*out){ viteng::aspida_vit_forward(pv,gh,gw,out); }
// Inject visual tokens into a prefill embedding buffer at image_pad positions.
// pHb[P,dim] (device); vtok[npos,dim] (device); pos[npos] (device) = token indices.
__global__ void k_vit_inject(float*pHb,const float*vtok,const int*pos,int npos,int dim){
  int i=blockIdx.x; if(i>=npos)return; int p=pos[i]; for(int d=threadIdx.x;d<dim;d+=blockDim.x) pHb[(size_t)p*dim+d]=vtok[(size_t)i*dim+d];
}
extern "C" void aspida_vit_inject(float*pHb,const float*vtok_dev,const int*pos_dev,int npos,int dim,void*stream){
  k_vit_inject<<<npos,256,0,(cudaStream_t)stream>>>(pHb,vtok_dev,pos_dev,npos,dim);
}
