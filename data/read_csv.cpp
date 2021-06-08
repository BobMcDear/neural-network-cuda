#include <iostream>
#include <fstream>

#include "read_csv.h"

void read_csv(float *inp, std::string name){
    std::ifstream file(name);
    std::string line;

    while(std::getline(file, line, '\n')){
        *inp = std::stof(line);
        inp++;
    }
}

/*
int main(){
    float *inp = new float[10001];
    read_csv("test.csv", inp);
    
    for (int i=0; i<10001; i++){
        std::cout << inp[i] << std::endl;
    }
    
    
    return 0;
}*/
