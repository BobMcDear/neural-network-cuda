#include <iostream>

#include "../CPU/mse.h"
#include "../GPU/mse.h"
#include "../utils/utils.h"


int main(){
    int bs;
    float *inp_cpu, *out_cpu, *inp_gpu, *out_gpu;

    for (int i=0; i<8; i++){
        std::cout << "Iteration " << i+1 << std::endl;

        bs = random_int(32, 2048);

        inp_cpu = new float[bs];
        cudaMallocManaged(&inp_gpu, bs*sizeof(float));

        out_cpu = new float[bs+1];
        cudaMallocManaged(&out_gpu, (bs+1)*sizeof(float));

        fill_array(inp_cpu, bs);
        set_eq(inp_gpu, inp_cpu, bs);

        fill_array(out_cpu, bs+1);
        set_eq(out_gpu, out_cpu, bs+1);

        MSE_CPU mse_cpu(bs);
        MSE_GPU mse_gpu(bs);

        mse_cpu.forward(inp_cpu, out_cpu);
        mse_gpu.forward(inp_gpu, out_gpu);
        mse_cpu._forward(inp_cpu, out_cpu);
        mse_gpu._forward(inp_gpu, out_gpu);

        std::cout << "Result of the forward pass" << std::endl;
        std::cout << mse_cpu.out[bs]-mse_gpu.out[bs] << std::endl;

        mse_cpu.backward();
        mse_gpu.backward();

        std::cout << "Result of the backward pass" << std::endl;
        test_res(mse_cpu.inp, mse_gpu.inp, bs);
    }

    return 0;
}
