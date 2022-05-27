#include "read.h"

using namespace std;

void read_input(vector<vector<float> > &x_train, vector<int> &y_train){
    ifstream in("x_train.txt");
    string line;
    int i = 0;
    
    while (getline(in, line)){
        float value;
        stringstream ss(line);
        
        x_train.push_back(vector<float>());
        
        while (ss >> value){
            x_train[i].push_back(value);
        }
        ++i;
    }
    
    in.close();
    
    in.open("y_train.txt");
    i = 0;
    
    while (getline(in, line)){
        int value;
        stringstream ss(line);
        
        while (ss >> value){
            y_train.push_back(value);
        }
        ++i;
    }
    
}
