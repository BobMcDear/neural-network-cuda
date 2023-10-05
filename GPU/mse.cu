#include "mse.h"


__global__
void mse_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        atomicAdd(&out[sz_out], fdividef(powf(inp[ind]-out[ind], 2), sz_out));
    }
}


__global__
void mse_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        inp[ind] = fdividef(2*(inp[ind]-out[ind]), sz_out);
    }
}


MSE_GPU::MSE_GPU(int _sz_out){
    sz_out = _sz_out;

    n_blocks = (sz_out + block_size - 1) / block_size;
}


void MSE_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
}


void MSE_GPU::_forward(float *_inp, float *_out){
    _out[sz_out] = 0.0f;

    mse_forward_gpu<<<n_blocks, block_size>>>(_inp, _out, sz_out);
    cudaDeviceSynchronize();
}


void MSE_GPU::backward(){
    mse_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}
