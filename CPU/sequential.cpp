#include "sequential.h"


void sequential_forward(float *inp, std::vector<Module*> layers, float *out){
    int sz_out;
    float *curr_out;

    for (int i=0; i<layers.size(); i++){
        Module *layer = layers[i];

        sz_out = layer->sz_out;

        curr_out = new float[sz_out];
        layer->forward(inp, curr_out);

        inp = new float[sz_out];
        inp = curr_out;
    }
}


void sequetial_update(std::vector<Module*> layers){
    for (int i=layers.size()-1; 0<=i; i--){
        Module *layer = layers[i];

        layer->update();
        layer->backward();
    }
}


Sequential_CPU::Sequential_CPU(std::vector<Module*> _layers){
    layers = _layers;
}


void Sequential_CPU::forward(float *inp, float *out){
    sequential_forward(inp, layers, out);
}


void Sequential_CPU::update(){
    sequetial_update(layers);
}
