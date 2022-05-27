#include "layers.h"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

class Neural_Network {
    
public:

    vector<string> layers;
    int nlayers;

    vector<vector<float> > weights;
    vector<vector<float> > grads_weights;

    vector<vector<float> > biases;
    vector<vector<float> > grads_bias;

    vector<vector<float> > activations;
    vector<vector<float> > grads_activations;

    

    vector<vector<int> > shapes;
    
    Neural_Network(int xdim){
        activations.push_back(vector<float>(xdim));
        grads_activations.push_back(vector<float>(xdim));
        nlayers=0;
    }

    void add_fcc(int xdim, int ydim) {

        layers.push_back("fcc");
        nlayers++;

        vector<float> weight_vec(xdim*ydim);
        fill(weight_vec.begin(),weight_vec.end(),0.01);
        
        weights.push_back(weight_vec);
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
        
        vector<float> weight_vec(F * C * FH * FW);
        fill(weight_vec.begin(),weight_vec.end(),0.01);
        
        weights.push_back(weight_vec);
        grads_weights.push_back(vector<float>(F * C * FH * FW));

        vector<float> bias_vec(F);
        fill(bias_vec.begin(),bias_vec.end(),0.01);
        biases.push_back(bias_vec);

        grads_bias.push_back(vector<float>(F));

        activations.push_back(vector<float>(F * (H-FH+1) * (W-FW+1)));
        grads_activations.push_back(vector<float>(F * (H-FH+1) * (W-FW+1)));
        // cout << (F * (F-FH+1) * (W-FW+1))<< '\n';
        // cout << "F=" << F<< " F-FH+1="<< F-FH+1 << " W-FW+1="<< W-FW+1 << '\n'; 
        // cout << activations[nlayers - 1].size() << '\n';
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
        grads_bias.push_back(vector<float>(0));

        activations.push_back(vector<float>(dim));
        grads_activations.push_back(vector<float>(dim));

        shapes.push_back(vector<int>(1));
        shapes[nlayers - 1][0] = dim;
    }

    void fwprop(vector<float> &x){

        for(int i=0;i<x.size();i++){
            activations[0][i] = x[i];
        }

        // if (layers[0] == "fcc") {
        //     forward_fcc(activations[0], weights[0], activations[1], biases[0], shapes[0][0], shapes[0][1]);
        // }
        // else if (layers[0] == "conv") {
            
        //     forward_conv(activations[0], weights[0], activations[1], biases[0], shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3], shapes[0][4], shapes[0][5]);
        // }
        
//        for(int i=0;i<weights[0].size();i++){
//            cout << weights[0][i]<<' ';
//        }
//        cout <<'\n';

        for(int i=0;i<nlayers;i++){
            if(layers[i]=="fcc"){
                forward_fcc(activations[i], weights[i], activations[i+1], biases[i], shapes[i][0], shapes[i][1]);
            }
            else if(layers[i]=="conv"){
                // cout << "yof"<< '\n';
                forward_conv(activations[i], weights[i], activations[i+1], biases[i], shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][4], shapes[i][5]);
            }
            else if(layers[i]=="relu"){
                forward_relu(activations[i], activations[i+1], shapes[i][0]);
            }
        }
    }

    // void set_gradient(int i,vector<float> y){
    //     for(int j=0;j<y.size();j++){
    //         grads_activations[i][j] = y[j];
    //     }
    // }


    void backprop(){
        // cout << layers[0]<< '\n';
            
        for(int i=nlayers-1;i>=0;i--){
            // cout << i<< '\n';
            if(layers[i] == "fcc"){

                backward_fcc(activations[i], weights[i], grads_activations[i], grads_activations[i+1], grads_weights[i],grads_bias[i], shapes[i][0], shapes[i][1]);
            }
            else if(layers[i] == ("conv")){
                backward_conv(activations[i], weights[i],activations[i+1], grads_activations[i], grads_weights[i], grads_bias[i],grads_activations[i+1],  shapes[i][0],shapes[i][1], shapes[i][2], shapes[i][3], shapes[i][4], shapes[i][5]);
            }
            else if(layers[i]=="relu"){
                backward_relu(activations[i], grads_activations[i], grads_activations[i+1], shapes[i][0]);
            }
        }
    }

    void get_activations(int i){
        cout << "activations["<<i<<"] size = "<<activations[i].size()<<'\n';
        for(int j=0;j<activations[i].size();j++){
            cout << activations[i][j]<<' ';
        }
        cout <<'\n';
    }

    void set_grads_activations(int i, vector<float> g){
        // cout << g.size()<<'\n';

        for(int j=0;j<g.size();j++){
            grads_activations[i][j] = g[j];
            // cout << grads_activations[i][j]<<' ';
        }
        // grads_activations[i] = g;
    }

    void get_weights(int i){
        for(int j=0;j<weights[i].size();j++){
            cout << weights[i][j]<<' ';
        }
        cout <<'\n';
    }
    
    
    void update_weights(float lr){
        for(int i=0;i< nlayers;i++){
            
            for(int j=0;j<weights[i].size();j++){
                
                weights[i][j] -= grads_weights[i][j]*lr;
            }
            
            for (int j=0;j<biases[i].size();j++){
                biases[i][j] -= grads_bias[i][j]*lr;
            }
            
        }
    }

    float find_accuracy(vector<vector<float> > &x, vector<int> &y){

        int correct=0;

        for(int iter=0;iter<x.size();iter++){
            fwprop(x[iter]);
            float max=activations[nlayers-1][0];
            int maxitem=0;

            for(int i=1;i<activations[nlayers-1].size();i++){
                if (activations[nlayers-1][i] > max){
                    max=activations[nlayers-1][i];
                    maxitem=i;
                }
            }

            if (maxitem == y[iter]){
                correct +=1;

            }
        }

        float accuracy= (float)correct/(float)x.size();
        return accuracy;
    }

    void train(vector<vector<float> > &x, vector<int> &y, float lr, int batch_size, int epochs){
        

        // long int N=x.size();
                
        for(int epoch=0;epoch < epochs;epoch++){
            float loss=0;
            int iter=0;


            for (int i = 0; i < x.size(); i++) {
                fwprop(x[i]);
                loss+=cross_entropy_derivative(activations[nlayers],grads_activations[nlayers],y[i],batch_size);
                backprop();
                if(iter%batch_size==0){
                    update_weights(lr);
                    loss=loss/batch_size;
                    cout << "loss = "<<loss<<'\n';
                    loss=0;

                }
                iter++;


                // mse_derivative(activations[nlayers-1],grads_activations[nlayers-1],grads_activations[nlayers]);
                ///----------------------------------------------------------------------------------------------------------------
//                if( i==10){
//                    for(int l=0;l<nlayers;l++){
//                        cout<< "layer "<<l<<" activations= ";
//                        for(int k=0;k<activations[l].size();k++){
//                            cout<< activations[l][k]<<' ';
//                        }
//                            cout<< '\n';
//
//                    }
//
//                    cout<< '\n';
//
//                }
//                if( i==10){
//                    for(int l=0;l<nlayers;l++){
//                        cout<< "layer "<<l<<" weights= ";
//                        for(int k=0;k<weights[l].size();k++){
//                            cout<< weights[l][k]<<' ';
//                        }
//                            cout<< '\n';
//
//                    }
//                for(int k=0;k<10;k++){
//                    cout<<activations[nlayers-2][i]<<' ';
//                }
//                    cout<< '\n';
//
//                }
                ////-------------------------------------------------------------------------------------------------------------
                // backprop();
                // update_weights(lr);
            //    if((i % 64) ==0){
            //        update_weights(lr);
            //     //    cout << "current loss =" << loss/i<<'\n';
            //    }
    //            cout << "current loss = "<<curr_loss<<'\n';
            }
            // loss=loss/N;
            
            // float accuracy=find_accuracy(x,y);

            // cout<<"epoch:"<<epoch<<" , "<< "loss: "<<loss << " accuracy:" << accuracy << '\n';
        }

    }




};
