#include <cmath>
#include <iostream>
#include <random>

#include "utils.h"


float max_diff(float *res1, float *res2, int n){
    float diff, r = 0;

    for (int i=0; i<n; i++){
        diff = abs(res1[i]-res2[i]);
        r = (r < diff) ? diff : r;
    }

    return r;
}


int n_zeros(float *a, int n){
    int r = 0;

    for (int i=0; i<n; i++){
        r += (!a[i]);
    }
    
    return r;
}


void fill_array(float *a, int n){
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, 1.0f); 

    for (int i=0; i<n; i++){
        a[i] = dist(gen);
    }
}


void test_res(float *res1, float *res2, int n){
    int n_res1_zeros = n_zeros(res1, n), n_res2_zeros = n_zeros(res2, n);
    float mx = max_diff(res1, res2, n);

    std::cout << "Number of zeros of res1: " << n_res1_zeros << std::endl;
    std::cout << "Number of zeros of res2: " << n_res2_zeros << std::endl;
    std::cout << "Maximum difference: " << mx << std::endl;
    std::cout << "*********" << std::endl;
}


void print_array(float *a, int n){
    for (int i=0; i<n; i++){
        std::cout << a[i] << std::endl;
    }
    std::cout << "*********" << std::endl;
}


void init_zero(float *a, int n){
    for (int i=0; i<n; i++){
        a[i] = 0.0f;
    }
}


void set_eq(float *a, float *b, int n){
    for (int i=0; i<n; i++){
        a[i] = b[i];
    }
}


void kaiming_init(float *w, int n_in, int n_out){
    float std = sqrt(2/(float) n_in);
    
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> dist(0.0f, std); 

    for (int i=0; i<n_in*n_out; i++){
        w[i] = dist(gen);
    }
}


int random_int(int min, int max){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(gen);
}
