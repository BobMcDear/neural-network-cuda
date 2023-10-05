#include <iostream>

#include "../CPU/linear.h"
#include "../GPU/linear.h"
#include "../utils/utils.h"


int main(){
    int bs, n_in, n_out;
    int sz_inp, sz_weights, sz_out;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<8; i++){
        std::cout << "Iteration " << i+1 << std::endl;

        bs = random_int(32, 256);
        n_in = random_int(32, 64);
        n_out = random_int(1, 32);

        sz_inp = bs*n_in;
        sz_weights = n_in*n_out;
        sz_out = bs*n_out;

        inp_cpu = new float[sz_inp];
        cudaMallocManaged(&inp_gpu, sz_inp*sizeof(float));

        out_cpu = new float[sz_out];
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

        lin_cpu.update();
        lin_gpu.update();

        std::cout << "Result of the update" << std::endl;
        std::cout << "Weights" << std::endl;
        test_res(lin_cpu.weights, lin_gpu.weights, sz_weights);
        std::cout << "Bias" << std::endl;
        test_res(lin_cpu.bias, lin_gpu.bias, n_out);

        lin_cpu.backward();
        lin_gpu.backward();

        std::cout << "Result of the backward pass" << std::endl;
        test_res(lin_cpu.inp, lin_gpu.inp, sz_inp);
    }

    return 0;
}
