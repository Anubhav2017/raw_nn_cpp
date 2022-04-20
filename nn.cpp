#include "layers.cpp"
#include <vector>
#include <iostream>
#include <String>

using namespace std;

class Neural_Network {

    vector<string> layers;
    int nlayers;

    vector<vector<float>> weights;
    vector<vector<float>> grads_weights;

    vector<vector<float>> biases;
    vector<vector<float>> grads_bias;

    <vector<vector<float>>> activations;
    vector<vector<float>> grads_activations;

    vector<vector<int>> shapes;


    void add_fcc(int xdim, int ydim) {

        layers.push_back("fcc");
        nlayers++;

        weights.push_back(vector<float>(xdim * ydim));
        grads_weights.push_back(vector<float>(ydim * xdim));

        biases.push_back(vector<float>(ydim));
        grads_bias.push_back(vector<float>(ydim));

        activations.push_back(vector<float>(ydim));
        grads_activations.push_back(vector<float>(ydim));

        shapes.push_back(vector<int>(2));
        shapes[nlayers - 1][0] = xdim;
        shapes[nlayers - 1][1] = ydim;
    }

    void add_conv(int F, int C, int H, int W, int FH, int FW) {
        layers.push_back("conv");
        nlayers++;

        weights.push_back(vector<float>(F * C * FH * FW));
        grads_weights.push_back(vector<float>(F * C * FH * FW));

        biases.push_back(vector<float>(F));
        grads_biases.push_back(vector<float>(F));

        activations.push_back(vector<float>(F * (F-FH+1) * (W-FW+1));
        grads_activations.push_back(vector<float>(F * (F-FH+1) * (W-FW+1));

        shapes.push_back(vector<int>(6));
        shapes[nlayers - 1][0] = F;
        shapes[nlayers - 1][1] = C;
        shapes[nlayers - 1][2] = H;
        shapes[nlayers - 1][3] = W;
        shapes[nlayers - 1][4] = FH;
        shapes[nlayers - 1][5] = FW;

    }

    void add_relu(int dim) {
        layers.push_back("relu");
        nlayers++;

        weights.push_back(vector<float>(0));
        grads_weights.push_back(vector<float>(0));

        biases.push_back(vector<float>(0));
        grads_biases.push_back(vector<float>(0));

        activations.push_back(vector<float>(dim));
        grads_activations.push_back(vector<float>(dim));

        shapes.push_back(vector<int>(1));
        shapes[nlayers - 1][0] = dim;
    }

    void fwprop(vector<float> &x){

        if layers[0] == "fcc" {
            forward_fcc(x, weights[0], activations[0], biases[0], shapes[0][0], shapes[0][1]);
        }
        else if layers[0] == "conv" {
            forward_conv(x, weights[0], activations[0], biases[0], shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3], shapes[0][4], shapes[0][5]);
        }
      



        for(int i=1;i<nlayers;i++){
            if(layers[i]=="fcc"){
                forward_fcc(activations[i-1], weights[i], activations[i], biases[i], shapes[i][0], shapes[i][1]);
            }
            else if(layers[i]=="conv"){
                forward_conv(activations[i-1], weights[i], activations[i], biases[i], shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][4], shapes[i][5]);
            }
            else if(layers[i]=="relu"){
                forward_relu(activations[i-1], activations[i], shapes[i][0]);
            }
        }
    }

    void backprop(){
            
        for(int i=nlayers-1;i>0;i--){
            if(layers[i]=="fcc"){
                backward_fcc(activations[i-1], weights[i], grads_activations[i-1], grads_activations[i], grads_weights[i],grads_biases[i], shapes[i][0], shapes[i][1]);
            }
            else if(layers[i]=="conv"){
                backward_conv(activations[i-1], weights[i],activations[i], grads_activations[i-1], grads_weights[i], grads_biases[i], shapes[i][0], grads_activations[i], shapes[i][0],shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][4], shapes[i][5]);
            }
            else if(layers[i]=="relu"){
                backward_relu(activations[i-1], grads_activations[i-1], grads_activations[i], shapes[i][0]);
            }
        }
    }

    void train(vector<vector<float>> &x, vector<float> &y){
        

        N=x.size();
        float loss=0;

        for (int i = 0; i < x.size(); i++) {
            fwprop(x[i]);
            loss+=cross_entropy_derivative(activations[nlayers-1],grads_activations[nlayers-1],y[i],N);
            backprop();
        }
        loss=loss/N;

        print("loss: ",loss);

    }




};