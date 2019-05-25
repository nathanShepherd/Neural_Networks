#include <vector>

using namespace std;

class NeuralNetwork {
public:
  vector<vector<vector<double>>> weights;
  vector<vector<double>> biases;
  vector<vector<vector<double>>> w_grads;
  vector<vector<double>> b_grads;

  double rand_num();
  void gen_matrix(bool gen_bias, int length, int depth);

  NeuralNetwork(int layers, 
                int input_size,
                int output_size);

  void forward();

  void backprop();
};

int train(NeuralNetwork net, 
          vector<vector<double>>& data, vector<int>& labels);
