#ifndef TRAIN_CPU_H
#define TRAIN_CPU_H

#include "sequential_cpu.h"
#include "linear_cpu.h"

void train(Sequential_CPU seq, float *inp, float *targ, int bs, int n_in, int n_epochs);

#endif
