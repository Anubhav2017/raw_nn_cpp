#include <vector>
#include "read.h"

using namespace std;

int main(){

    vector<vector<float> > x_train;

    vector<int> y_train;

    read_input(x_train, y_train);


    cout << x_train.size() << '\n';
    cout << y_train.size() << '\n';


}