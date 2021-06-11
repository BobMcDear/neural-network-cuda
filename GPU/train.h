#ifndef TRAIN_GPU_H
#define TRAIN_GPU_H


#include "sequential.h"


void train_gpu(Sequential_GPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs);


#endif
