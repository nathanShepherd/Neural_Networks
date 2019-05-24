#include <iostream>
#include <fstream>
#include <vector>
#include "nn.h"

using namespace std;

void read_input(vector<vector<double>>& data, vector<int>& labels);

int main() {
  vector<vector<double>> data;
  vector<int> labels;
  read_input(data, labels);
  train(data, labels); 
/*
  for (int l: labels) {
    cout << l << " ";
  }
  cout << endl;
*/
  return 0;
}

void read_input(vector<vector<double>>& data, vector<int>& labels) {
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

  inf >> header;// NumClasses  

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
}
