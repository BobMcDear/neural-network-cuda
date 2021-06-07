#include "module.h"
#include "utils.h"
#include "../CPU/linear_cpu.h"
#include "../CPU/mse_cpu.h"
#include "../CPU/relu_cpu.h"
#include "../CPU/sequential_cpu.h"
#include "../GPU/linear_gpu.h"
#include "../GPU/mse_gpu.h"
#include "../GPU/relu_gpu.h"
#include "../GPU/sequential_gpu.h"

#include <chrono>
#include <iostream>
using namespace std;


int main(){
    /*
    int bs = 8, n_in = 9, n_hidden = 10, n_out = 1;
    int sz_inp = bs*n_in, sz_hidden = bs*n_hidden, sz_out = bs*n_out, sz_weight1 = n_in*n_hidden, sz_weight2 = n_hidden*n_out;

    float *inp = new float[sz_inp], *out = new float[sz_out], *targ = new float[sz_out];
    
    fill_array(inp, sz_inp, 1);
    fill_array(targ, sz_out, 2);

    cout << "inp:" << endl;
    //print_array(inp, sz_inp);

    cout << "targ:" << endl;
    //print_array(targ, sz_out);

    Linear_CPU* lin1 = new Linear_CPU(bs, n_in, n_hidden);
    ReLU_CPU* relu1 = new ReLU_CPU(sz_hidden);
    Linear_CPU* lin2 = new Linear_CPU(bs, n_hidden, n_out);
    ReLU_CPU* relu2 = new ReLU_CPU(sz_out);


    std::vector<Module*> layers = {lin1, relu1, lin2, relu2};
    Sequential_CPU seq(layers);
    MSE_CPU mse(sz_out);

    cout << "lin1->weights:" << endl;
    //print_array(lin1->weights, sz_weight1);

    cout << "lin2->weights:" << endl;
    //print_array(lin2->weights, sz_weight2);

    seq.forward(inp, out);
    mse.forward(relu2->out, targ);

    mse.backward();
    seq.update();

    cout << "Updated lin2->weights:" << endl;
    //print_array(lin2->weights, n_hidden*n_out);

    cout << "Updated lin2->bias:" << endl;
    //print_array(lin2->bias, n_out);
    
    cout << "Updated lin1->weights:" << endl;
    //print_array(lin1->weights, n_in*n_hidden);

    cout << "Updated lin1->bias:" << endl;
    //print_array(lin1->bias, n_hidden);
    */


    int bs = 128, n_in = 768, n_hidden1 = 100, n_hidden2 = 50, n_out = 1;
    int sz_inp = bs*n_in, sz_weight1 = n_in*n_hidden1, sz_weight2 = n_hidden1*n_hidden2, sz_weight3 = n_hidden2*n_out, sz_out = bs*n_out;

    float *inp_cpu = new float[sz_inp], *out_cpu = new float[sz_out], *targ_cpu = new float[sz_out];
    float *inp_gpu, *out_gpu, *targ_gpu;
    
    cudaMallocManaged(&inp_gpu, sz_inp*sizeof(float));
    cudaMallocManaged(&out_gpu, sz_out*sizeof(float));
    cudaMallocManaged(&targ_gpu, sz_out*sizeof(float));

    fill_array(inp_cpu, sz_inp);
    fill_array(targ_cpu, sz_out);

    set_eq(inp_gpu, inp_cpu, sz_inp);
    set_eq(targ_gpu, targ_cpu, sz_out);

    //print_array(inp_cpu, sz_inp);
    //print_array(targ_cpu, sz_out);

    Linear_CPU* lin1_cpu = new Linear_CPU(bs, n_in, n_hidden1);
    ReLU_CPU* relu1_cpu = new ReLU_CPU(bs*n_hidden1);
    Linear_CPU* lin2_cpu = new Linear_CPU(bs, n_hidden1, n_hidden2);
    ReLU_CPU* relu2_cpu = new ReLU_CPU(bs*n_hidden2);
    Linear_CPU* lin3_cpu = new Linear_CPU(bs, n_hidden2, n_out);

    Linear_GPU* lin1_gpu = new Linear_GPU(bs, n_in, n_hidden1);
    ReLU_GPU* relu1_gpu = new ReLU_GPU(bs*n_hidden1);
    Linear_GPU* lin2_gpu = new Linear_GPU(bs, n_hidden1, n_hidden2);
    ReLU_GPU* relu2_gpu = new ReLU_GPU(bs*n_hidden2);
    Linear_GPU* lin3_gpu = new Linear_GPU(bs, n_hidden2, n_out);


    set_eq(lin1_gpu->weights, lin1_cpu->weights, sz_weight1);
    set_eq(lin2_gpu->weights, lin2_cpu->weights, sz_weight2);
    set_eq(lin3_gpu->weights, lin3_cpu->weights, sz_weight3);

    //print_array(lin1_cpu->weights, sz_weight1);
    //print_array(lin2_cpu->weights, sz_weight2);
    //print_array(lin3_cpu->weights, sz_weight3);


    std::vector<Module*> layers_cpu = {lin1_cpu, relu1_cpu, lin2_cpu, relu2_cpu, lin3_cpu};
    Sequential_CPU seq_cpu(layers_cpu);

    MSE_CPU mse_cpu(sz_out);

    chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i=0; i<32; i++){
        seq_cpu.forward(inp_cpu, out_cpu);
        mse_cpu.forward(lin3_cpu->out, targ_cpu);

        mse_cpu.backward();
        seq_cpu.update();
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    cout << "CPU time difference = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;

    
    std::vector<Module*> layers_gpu = {lin1_gpu, relu1_gpu, lin2_gpu, relu2_gpu, lin3_gpu};
    Sequential_GPU seq_gpu(layers_gpu);

    MSE_GPU mse_gpu(sz_out);

    chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
    for (int i=0; i<32; i++){
        seq_gpu.forward(inp_gpu, out_gpu);
        mse_gpu.forward(lin3_gpu->out, targ_gpu);

        mse_gpu.backward();
        seq_gpu.update();
    }
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();

    cout << "GPU time difference = " << chrono::duration_cast<chrono::milliseconds>(end1 - begin1).count() << "[µs]" << endl;

    //print_array(lin1_cpu->weights, sz_weight1);
    //print_array(lin2_cpu->weights, sz_weight2);
    //print_array(lin3_cpu->weights, sz_weight3);

    //test_res(lin1_gpu->weights, lin1_cpu->weights, sz_weight1);
    //test_res(lin2_gpu->weights, lin2_cpu->weights, sz_weight2);
    //test_res(lin3_gpu->weights, lin3_cpu->weights, sz_weight3);
    
    //test_res(lin1_gpu->bias, lin1_cpu->bias, n_hidden1);
    //test_res(lin2_gpu->bias, lin2_cpu->bias, n_hidden2);
    //test_res(lin3_gpu->bias, lin3_cpu->bias, n_out);

    return 0;
}
