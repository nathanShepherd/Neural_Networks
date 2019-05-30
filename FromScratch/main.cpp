#include <iostream>
#include <fstream>
#include <vector>
#include "nn.h"

using namespace std;

int read_input(vector<vector<double>>& data, vector<int>& labels);

int main() {
  vector<vector<double>> data;
  vector<int> labels;
  int num_classes = read_input(data, labels);

  int layers = 3;  
  NeuralNetwork net(layers, data[0].size(), num_classes);

  train(net, data, labels);
  net.print_weights();

  /* Debug
  vector<vector<double>> lhs = {{1 , 2, 3 ,4}};
  vector<vector<double>> rhs = {{2, 2, 2, 2},
  vector<vector<double>> prod;
  net.dot(lhs, rhs, prod);

  for (double vect: prod[0]) {
    cout << vect <<  " ";
  }
  cout << endl; // debug
  */
/*
  for (int l: labels) {
    cout << l << " ";
  }
  cout << endl;
*/
  return 0;
}

int read_input(vector<vector<double>>& data, vector<int>& labels) {
  ifstream inf;
  inf.open("input.txt");
  int header;
  int input_size;
 
  inf >> header;// DataSize
  data.reserve(header);
  labels.reserve(header);

  inf >> input_size;// InputSize
  for (int i = 0; i < header; ++i) {
    vector<double> datum;
    datum.reserve(header);
    data.push_back(datum);
  }

  int num_classes;
  inf >> num_classes;// NumClasses  

  double temp;
  for (vector<double>& vect: data) {
    for (int i = 0; i < input_size; ++i) {
      inf >> temp;
      //cout << temp << endl; //debug
      vect.push_back(temp);
    }
    inf >> header;// header is label
    labels.push_back(header);
  }
  return num_classes;
}
