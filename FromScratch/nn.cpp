#include "nn.h"
#include "math.h"
#include <numeric>
#include <cstdlib>
#include <iostream>
#include <algorithm>
/*
 * @ https://medium.freecodecamp.org/building-a-3-layer-neural-network-from-scratch-99239c4af5d3
 *
 # This is the forward propagation function
 #   a0 = input_data
def forward_prop(weights, biases, a0):
    
    # Load parameters from model
    W1, W2, W3 = weights
    b1, b2, b3 = biases

    # Do the first Linear step 
    z1 = a0.dot(W1) + b1
    
    # Put it through the first activation function
    a1 = np.tanh(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Put through second activation function
    a2 = np.tanh(z2)
    
    #Third linear step
    z3 = a2.dot(W3) + b3
    
    #For the Third linear activation function we use the softmax function
    a3 = softmax(z3)
    
    #Store all results in these values
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'a3':a3,'z3':z3}
    return cache

# This is the backward propagation function
def backward_prop(weights, biases, cache,y):

  # Load parameters from model
    W1, W2, W3 = weights
    b1, b2, b3 = biases
    
    # Load forward propagation results
    a0,a1, a2,a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
    
    # Get number of samples
    m = y.shape[0]
    
    # Calculate loss derivative with respect to output
    dz3 = a3 - y 

    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*(a2.T).dot(dz3) #dW2 = 1/m*(a1.T).dot(dz2) 
    
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    
    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T) ,tanh_derivative(a2))
    
    # Calculate loss derivative with respect to first layer weights
    dW2 = 1/m*np.dot(a1.T, dz2)
    
    # Calculate loss derivative with respect to first layer bias
    db2 = 1/m*np.sum(dz2, axis=0)
    
    dz1 = np.multiply(dz2.dot(W2.T),tanh_derivative(a1))
    
    dW1 = 1/m*np.dot(a0.T,dz1)
    
    db1 = 1/m*np.sum(dz1,axis=0)
    
    # Store gradients
    grads = {'dW3':dW3, 'db3':db3, 'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads 

def update_parameters(model,grads,learning_rate):\n",
    # Update parameters\n",
    W1 -= learning_rate * grads['dW1']\n",
    b1 -= learning_rate * grads['db1']\n",
    W2 -= learning_rate * grads['dW2']\n",
    b2 -= learning_rate * grads['db2']\n",
    W3 -= learning_rate * grads['dW3']\n",
    b3 -= learning_rate * grads['db3']\n", 

*/

double rand_gen() {
  return rand() % 1000 - 500;
}
void NeuralNetwork::gen_matrix(bool gen_bias, int length, int depth) {
  vector<vector<double>> layer;
  layer.reserve(length);

  for (int i = 0; i < length; i++) {
    vector<double> inv(depth);
    generate(inv.begin(), inv.end(), rand_gen);
    layer.push_back(inv);
  }
  if (gen_bias) {
    biases.push_back(layer[0]);
  }
  else {
    weights.push_back(layer);
  }
}

NeuralNetwork::NeuralNetwork(int layers, int input_size, int output_size) {
  // layers includes input and output layers
  // ie. 3 layers has one hidden layer

  // input layer
  bool gen_bias = true;
  bool gen_weight = false;

  gen_matrix(gen_weight, input_size, 
             (input_size * layers) / output_size);

  gen_matrix(gen_bias, 1, (input_size * layers) / output_size);
  layers -= 1;

  // hidden layers
  while (layers > 1) {
    gen_matrix(gen_weight, weights.back().size(),
               (weights.back().size() * layers) / output_size);
  
    gen_matrix(gen_bias, 1, (biases.back().size() * layers) / output_size);
    layers -= 1;
  } 

  // output layer
  gen_matrix(gen_weight, weights.back().size(),output_size);
  gen_matrix(gen_bias, 1, output_size);

}

void NeuralNetwork::dot(vector<vector<double>>& lhs,
                        vector<vector<double>>& rhs,
                        vector<vector<double>>& product) {
  product.clear();
  for (size_t i = 0; i < lhs.size(); ++i) {
    vector<double> prod;
    //prod.reserve(rhs.size());
    for (size_t j = 0; j < rhs.size(); ++j) {
      /*
      double summ = 0;
      for (size_t k = 0; k < rhs[0].size(); ++k) {
        //cout << i << " " << j << " " << k << endl; // debug
        //cout << "::: " << lhs[i][k] << " " << rhs[j][k] << endl;//debug
        summ += lhs[i][k] * rhs[j][k];
      }
      prod.push_back(summ);
      */
      prod.push_back(inner_product(lhs[i].begin(), lhs[i].end(),
                                   rhs[j].begin(), 0));
    }
    //if multiple rows on lhs, pushback new vector to product
    product.push_back(prod);
  }
}

void NeuralNetwork::mat_sum_activate(vector<vector<double>>& lhs,
                                     vector<double>& rhs) {
  // Take the sum of lhs and rhs then apply activation to lhs
  for (size_t row = 0; row < lhs.size(); ++row) {
    for (size_t col = 0; col < rhs.size(); ++col) {
      lhs[row][col] += rhs[col];
      lhs[row][col] = tanh(lhs[row][col]);
    }
  }
}

void NeuralNetwork::softmax(vector<vector<double>>& grads) {
  // Softmax
  // Assumes grads is 1D
  double sum;
  vector<double> exp_v;
  exp_v.reserve(grads[0].size());
  for (double dW: grads[0]) {
    exp_v.push_back(exp(dW));
    sum += exp_v.back();
  }
  for (size_t i = 0; i < grads[0].size(); ++i) {
    grads[0][i] = exp_v[i] / sum;
  }
}


void NeuralNetwork::print_weights() {
  for (vector<vector<double>> w: weights) {
    for (vector<double> vect: w) {
      for (double d: vect) {
        cout << d << " ";
      }
      cout << endl;
    }
    cout << "********************\n";
  }
}
void NeuralNetwork::forward(const vector<vector<double>>& input) {
  cout << "Forwards\n";

  w_grads.clear();
  //b_grads.clear(); not needed
  w_grads.push_back(input);
  for (size_t i = 0; i < weights.size() - 1; ++i) {
    vector<vector<double>> grads;
    dot(w_grads.back(), weights[i], grads);
    mat_sum_activate(grads, biases[i]);
    w_grads.push_back(grads);
  }

  vector<vector<double>> grads;
  dot(w_grads.back(), weights.back(), grads);

  mat_sum_activate(grads, biases.back());

  softmax(grads);
  w_grads.push_back(grads);
/*
  cout << "Output: ";// debug
  for (vector<double> v: grads) {
    for (double n: v) {
      cout << n << " i ";
    }
    cout << endl;
  }
  */
}





void NeuralNetwork::backprop() {
}





int train(NeuralNetwork& net, 
          vector<vector<double>>& data, vector<int>& labels) {

  net.forward(data);
  // max_element(v.begin(), v.end()) - v.begin();
  return 1;
}
