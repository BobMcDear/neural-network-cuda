#include "relu.h"


void relu_forward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        out[i] = (0 < inp[i]) ? inp[i] : 0;
    }
}


void relu_backward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        inp[i] = (0 < inp[i]) * out[i];
    }
}


ReLU_CPU::ReLU_CPU(int _sz_out){
    sz_out = _sz_out;
}


void ReLU_CPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;

    relu_forward_cpu(inp, out, sz_out);
}


void ReLU_CPU::backward(){
    relu_backward_cpu(inp, out, sz_out);
    delete[] out;
}
