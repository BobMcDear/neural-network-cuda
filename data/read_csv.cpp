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
