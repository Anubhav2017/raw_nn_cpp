#include <vector>
// #include "read.h"
#include "nn.cpp"

using namespace std;

int main(){

    vector<vector <float> > x_train;

    vector<int> y_train; 

    // read_input(x_train, y_train);
    x_train.push_back(vector<float>(0));

    for(int i=0;i<10;i++){
        x_train[0].push_back(i*0.1);
    }
    
    
    
    Neural_Network nn(4);

    vector<float> y(3);
    y[0]=0.33;
    y[1]=-0.66;
    y[2]=0.33;

    vector<float> resetx(4);
    resetx[0]=0;
    resetx[1]=0;
    resetx[2]=0;
    resetx[3]=0;
    nn.set_grads_activations(0,resetx);

    
    // for(int i=0;i<18;i++){
    //     y[i]=0.1;
    // }
    
    // fill(y.begin(),y.end(),0.1);
    
    nn.add_conv(1,1,4,1,2,1);

    nn.fwprop(x_train[0]);
    // mse_derivative(nn.activations[nn.nlayers],nn.grads_activations[nn.nlayers],y);
    nn.set_grads_activations(1,y);
    
    // nn.get_activations(0);
    // nn.get_weights(0);
    nn.backprop();
    // nn.update_weights(0.01);

    

    // for(int i=0;i< nn.grads_weights[0].size();i++){
    //     cout << nn.grads_weights[0][i] << ' ';
    // }
    // cout << '\n';

    // for(int i=0;i< nn.grads_bias[0].size();i++){
    //     cout << nn.grads_bias[0][i] << ' ';
    // }
    // cout << '\n';


    // for(int i=0;i<nn.weights[0].size();i++){
    //     cout << nn.weights[0][i]<< " ";    
    // }
    // cout << '\n';
    for(int i=0;i<nn.activations[0].size();i++){
        cout << nn.grads_activations[0][i]<< " ";
        
    }
    cout << '\n';
    // for(int i=0;i<nn.activations[1].size();i++){
    //     cout << nn.grads_activations[1][i]<< " ";
        
    // }
    cout << '\n';

    // cout << nn.weights
    // nn.add_relu(2346);
    // nn.add_fcc(2346,10);
    
//    for (int i=0;i<784;i++){
//        cout<< x_train[1][i];
//    }
    // cout<<'\n';
    
    // nn.train(x_train,y_train,0.6);


}
