#include <iostream>

#include "utils.h"
#include "../CPU/linear.h"
#include "../GPU/linear.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in, n_out;
    int sz_inp, sz_weights, sz_out;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<10; i++){
        std::cout << "Iteration " << i << std::endl;

        bs = random_int(32, 256);
        n_in = random_int(32, 64);
        n_out = random_int(1, 32);
        
        sz_inp = bs*n_in;
        sz_weights = n_in*n_out;
        sz_out = bs*n_out;

        inp_cpu = new float[sz_inp];
        out_cpu = new float[sz_out];
    
        cudaMallocManaged(&inp_gpu, sz_inp*sizeof(float));
        cudaMallocManaged(&out_gpu, sz_out*sizeof(float));
    
        fill_array(inp_cpu, sz_inp);
        set_eq(inp_gpu, inp_cpu, sz_inp);
        
        Linear_CPU lin_cpu(bs, n_in, n_out);
        Linear_GPU lin_gpu(bs, n_in, n_out);
        set_eq(lin_gpu.weights, lin_cpu.weights, sz_weights);
    
        lin_cpu.forward(inp_cpu, out_cpu);
        lin_gpu.forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl; 
        test_res(lin_cpu.out, lin_gpu.out, sz_out);    
    }
    
    return 0;
}
