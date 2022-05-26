#include <vector>
#include "layers.h"
#include<iostream>
#include<cmath>

using namespace std;

int main(){
    vector<float> labels= {0,1,0};
    vector<float> preds= {0.1,0.8,0.1};
    vector<float> grads= {0.1,0,0.1};

    float loss = cross_entropy_derivative(preds,grads,1,1);

    cout << loss << endl;
    for(int i=0;i<3;i++){
        cout << grads[i] << " " << endl;
    }
    // cout << -log(preds[1])<< endl;


}