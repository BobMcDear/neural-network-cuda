#ifndef TRAIN_CPU_H
#define TRAIN_CPU_H


#include "sequential.h"


void train_cpu(Sequential_CPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs);


#endif
