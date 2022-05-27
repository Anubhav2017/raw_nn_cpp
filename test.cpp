#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;
int main()
{
    std::fstream in("x_train.txt");
    std::string line;
    std::vector<std::vector<float> > v;
    int i = 0;

    while (std::getline(in, line))
    {
        float value;
        std::stringstream ss(line);

        v.push_back(std::vector<float>());

        while (ss >> value)
        {
            v[i].push_back(value);
        }
        ++i;
    }
}