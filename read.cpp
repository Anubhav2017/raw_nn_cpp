#include <iostream>
#include <bits/stdc++.h>
#include <fstream>
#include <string>

using namespace std;

size_t split(const std::string &txt, std::vector<int> &strs, char ch)
{
    size_t pos = txt.find( ch );
    size_t initialPos = 0;
    strs.clear();

    // Decompose statement
    while( pos != std::string::npos ) {
        const char* s=txt.substr( initialPos, pos - initialPos ).c_str();
        int x;
        sscanf(s,"%d",&x); 
        strs.push_back(x);
        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    const char* s=txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ).c_str();
    int x;
    sscanf(s,"%d",&x); 
    strs.push_back(x);

    return strs.size();
}

int main(){
    fstream xfile;
    xfile.open("x_train.txt",ios::in); //open a file to perform read operation using file object
//    if (xfile.is_open()){ //checking whether the file is open
//       string tp;
//       vector<int> v;
//       while(getline(xfile, tp)){ //read data from file object and put it into string.
//          split(tp, v,' ');
//          for(int i=0;i<v.size();i++){
//              cout << v[i] <<" ";
//          }
         

//       }
//       cout<<'\n';
//       xfile.close(); //close the file object.
//    }


   fstream yfile;
    yfile.open("y_train.txt",ios::in); //open a file to perform read operation using file object
   if (yfile.is_open()){ //checking whether the file is open
      string tp;
      vector<int> v;
      while(getline(yfile, tp)){ //read data from file object and put it into string.
         const char* s = tp.c_str();
         int x;
        sscanf(s,"%d",&x); 
        v.push_back(x);
         
         }
         for(int i=0;i<v.size();i++){
             cout << v[i] <<" ";
         

      }
      cout<<'\n';
      yfile.close(); //close the file object.
   }
}