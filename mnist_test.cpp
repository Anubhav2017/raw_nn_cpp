#include <vector>
#include "read.h"
#include "nn.cpp"
using namespace std;

int main(){

    vector<vector <float> > x_train;

    vector<int> y_train;

    read_input(x_train, y_train);

    cout << x_train.size()<< endl;
    
    
    Neural_Network nn(784);

    nn.add_conv(5,1,784,1,5,1);
    nn.add_relu(3900);
    nn.add_fcc(3900,10);

    
    // nn.add_conv(5,1,28,28,5,5);
    // nn.add_relu(2880);
    // nn.add_fcc(2880,10);
    
    // cout<<'\n';
    
    nn.train_cel(x_train,y_train,0.005,32,10);


}
