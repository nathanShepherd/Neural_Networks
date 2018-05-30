# Started from Demo by Siraj Raval
# Make_a_neural_network / demo.py
# https://github.com/llSourcell/Make_a_neural_network/blob/master/demo.py

from numpy import exp, array, random, dot

class NeuralNetwork():
  def __init__(self):
    random.seed(1)
    
    self.weights = 2 * random.random((3, 1)) - 1

  def sigmoid(self, x, deriv=False):
    if not deriv:
      return 1 / (1 + exp(-x))
    return x * (1 - x )
  
  def feed_forward(self, ins):
    return self.sigmoid( dot( ins, self.weights ) )
 
  def back_prop(self, x, y, error):
    delta = dot( x.T, error * self.sigmoid( y, deriv=True))
    self.weights += delta

  def train(self, train_x, train_y, epochs):
    for epoch in range(epochs):
      pred = self.feed_forward(train_x)
    
      error = train_y - pred

      self.back_prop( train_x, pred, error )

if __name__ == "__main__":
  nn = NeuralNetwork()

  print("Random starting weights")
  print(nn.weights)
 
  

  X = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])    
  Y = array([[0, 1, 1, 0]]).T

  nn.train(X, Y, 10000)

  print("Weights after training")
  print(nn.weights)

  print(nn.feed_forward(array([1, 0, 0])))
