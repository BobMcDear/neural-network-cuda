#include "mse.h"


void mse_forward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        out[sz_out] += (inp[i]-out[i])*(inp[i]-out[i])/sz_out;
    }
}


void mse_backward_cpu(float *inp, float *out, int sz_out){
    for (int i=0; i<sz_out; i++){
        inp[i] = 2*(inp[i]-out[i])/sz_out;
    }
}


MSE_CPU::MSE_CPU(int _sz_out){
    sz_out = _sz_out;
}


void MSE_CPU::forward(float *_inp, float *_out){
    inp = _inp;
    out = _out;
}


void MSE_CPU::_forward(float *_inp, float *_out){
    _out[sz_out] = 0.0f;

    mse_forward_cpu(_inp, _out, sz_out);
}


void MSE_CPU::backward(){
    mse_backward_cpu(inp, out, sz_out);
}
