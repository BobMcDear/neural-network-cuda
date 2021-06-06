#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

void read_csv(std::string filename, float *inp, float *targ){
    std::ifstream myFile(filename);

    if(!myFile.is_open()) throw std::runtime_error("Could not open file");



    while(std::getline(myFile, line, ','))
    {
        std::cout << line << std::endl;
        /*
        // Create a stringstream of the current line
        std::stringstream ss(line);
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val){
            
            // Add the current integer to the 'colIdx' column's values vector
            result.at(colIdx).second.push_back(val);
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }*/
    }

    // Close file
    myFile.close();

    return result;
}

int main() {
    // Read three_cols.csv and ones.csv
    std::vector<std::pair<std::string, std::vector<int>>> three_cols = read_csv("data.csv");
    for ( auto it = three_cols.begin(); it != three_cols.end(); it++ ){
        auto pClass1 = it->first;
        auto pClass2 = it->second;
        std::cout << pClass2.at(0) << std::endl;
        //std::cout << pClass2 << std::endl;
   }

    return 0;
}