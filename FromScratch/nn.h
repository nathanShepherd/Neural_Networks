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

  void dot(vector<vector<double>>& lhs, 
           vector<vector<double>>& rhs,
           vector<vector<double>>& product);

  void mat_sum_activate(vector<vector<double>>& lhs,
                        vector<double>& rhs);

  void softmax(vector<vector<double>>& grads);

  void print_weights();

  void forward(const vector<vector<double>>& input);

  void backprop(const vector<vector<double>>& input, 
                const vector<vector<int>>& labels);
};

int train(NeuralNetwork& net, 
          vector<vector<double>>& data, vector<vector<int>>& labels);
