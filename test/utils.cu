#include "utils.h"
#include "../utils/utils.h"


void get_data(float *inp_cpu, float *out_cpu, float *inp_gpu, float *out_gpu, int bs, int n_in, int n_out){
    int sz_inp = bs*n_in, sz_out = bs*n_out;

    inp_cpu = new float[sz_inp];
    out_cpu = new float[sz_out];

    cudaMallocManaged(&inp_gpu, sz_inp*sizeof(float));
    cudaMallocManaged(&out_gpu, sz_out*sizeof(float));

    fill_array(inp_cpu, sz_inp);
    set_eq(inp_gpu, inp_cpu, sz_inp);
}


int random_int(int min, int max){
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}
