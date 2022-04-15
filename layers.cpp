#include<vector>

void forward_fcc(vector<float> &x, vector<float,float> &w, vector<float> &y, vector<float> &b, int xdim, int ydim){

    for(int i=0; i< ydim;i++){
        y[i]= b[i];

        for (int j=0; j<xdim;j++){
            y[i]+= w[i][j]*x[j];
        }
    }

}

void backward_fcc(vector<float> &x, vector<float,float> &w, vector<float> &b, vector<float> &dx, vector<float> &dy,int xdim,int ydim,float lr){
    //compute gradient of activations
    
    vector<float> db(ydim);
    vector<float,float> dw(ydim,xdim);
    
    for(int i=0;i<xdim;i++){
        dx[i]=0;
    }
    
    for(int j=0;j<xdim;j++){
        for(int i=0;i<ydim;i++){
            dx[j] += dy[i] * w[i][j];
        }
        
    }
    //compute gradient of weights
    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            dw[i][j] = dy[i]*x[j];
        }
    }

    //compute gradient of biases
    for (int i=0;i<ydim;i++){
        db[i] = dy[i];
    }
    
    for(int i=0;i<ydim;i++){
        for(int j=0;j<xdim;j++){
            
            w[i][j]-=lr*dw[i][j];
            
        }
        
        b[i] -= lr*db[i];
    }
}


void flatten_forward(vectors)

void conv_forward(vector<float,float,float> &x, vector<float,float,float,float> &w, vector<float,float,float,float> &y, vector<float> &b, int xdim, int ydim){

    

}