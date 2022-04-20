#include "npy.hpp"
#include <vector>

using namespace std;

int test_load(void) {
  vector<unsigned long> shape;
  bool fortran_order;
  vector<int> data;

  vector<const char*> allpaths {
    "x_train.npy",
    "x_test.npy",
    "y_train.npy",
    "y_test.npy",
  };

  for (auto path : allpaths) {
    shape.clear();
    data.clear();
    npy::LoadArrayFromNumpy(path, shape, fortran_order, data);

    // cout << "shape: ";
    // for (size_t i = 0; i<shape.size(); i++)
    //   cout << shape[i] << ", ";
    // cout << endl;
    // cout << "fortran order: " << (fortran_order ? "+" : "-");
    // cout << endl;
    // cout << "data: ";
    // for (size_t i = 0; i<data.size(); i++)
    //   cout << data[i] << ", ";
    // cout << endl << endl;
  }

  return 0;
}

int main(){

    test_load();

}