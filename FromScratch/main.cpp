#include <iostream>
#include <fstream>
#include <vector>
#include "nn.h"

using namespace std;

int read_input(vector<vector<double>>& data, vector<vector<int>>& labels);

int main() {
  vector<vector<double>> data;
  vector<vector<int>> labels;
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

int read_input(vector<vector<double>>& data, vector<vector<int>>& labels) {
  ifstream inf;
  inf.open("input.txt");
  int header;
  int data_size;
  int input_size;
 
  inf >> data_size;
  data.reserve(data_size);
  labels.reserve(data_size);

  inf >> input_size;// InputSize
  for (int i = 0; i < data_size; ++i) {
    vector<double> datum;
    datum.reserve(header);
    data.push_back(datum);
  }

  int num_classes;
  inf >> num_classes;// NumClasses  
  for (int i = 0; i < data_size; ++i) {
    vector<double> datum;
    datum.reserve(num_classes);
    data.push_back(datum);
  }

  double temp;
  for (int j = 0; j < data_size; ++j) {
    for (int i = 0; i < input_size; ++i) {
      inf >> temp;
      //cout << temp << endl; //debug
      data[j].push_back(temp);
    }
    //inf >> header;// getting '\t'
    for (int i = 0; i < num_classes; ++i) {
      inf >> header;
      labels[j].push_back(temp);
    }
    //labels.push_back(header);
  }
  return num_classes;
}
