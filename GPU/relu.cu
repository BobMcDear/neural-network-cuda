#include "relu.h"


__global__
void relu_forward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        out[ind] = fmaxf(0, inp[ind]);
    }
}


__global__
void relu_backward_gpu(float *inp, float *out, int sz_out){
    int ind = blockDim.x*blockIdx.x + threadIdx.x;

    if (ind < sz_out){
        inp[ind] = (0 < inp[ind]) * out[ind];
    }
}


ReLU_GPU::ReLU_GPU(int _sz_out){
    sz_out = _sz_out;

    n_blocks = (sz_out + block_size - 1) / block_size;
}


void ReLU_GPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    relu_forward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();
}


void ReLU_GPU::backward(){
    relu_backward_gpu<<<n_blocks, block_size>>>(inp, out, sz_out);
    cudaDeviceSynchronize();

    cudaFree(out);
}
