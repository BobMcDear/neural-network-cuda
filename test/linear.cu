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

        get_data(inp_cpu, out_cpu, inp_gpu, out_gpu, bs, n_in, n_out);

        Linear_CPU lin_cpu(bs, n_in, n_out);
        Linear_GPU lin_gpu(bs, n_in, n_out);
        set_eq(lin_gpu.weights, lin_cpu.weights, sz_weights);
    
        lin_cpu.forward(inp_cpu, out_cpu);
        lin_gpu.forward(inp_gpu, out_gpu);
        test_res(out_cpu, out_gpu, sz_out);    
    }
    
    return 0;
}