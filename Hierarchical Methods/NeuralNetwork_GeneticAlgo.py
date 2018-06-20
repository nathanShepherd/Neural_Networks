# # # # # # # # 
# Developed by Nathan Shepherd

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def make_data(amount, x_length, num_classes=2,split_ratio=0.8):
  # Generate samples
  x, y = make_blobs(n_samples = amount, 
                    centers   = num_classes,
                    n_features= x_length )
  targets = y

  # Encode Y as one-hot vectors
  _y = [[] for i in range(len(y))]
  for i in range(len(y)):
    encoding = [0 for _ in range(num_classes)]
    encoding[y[i]] = 1
    _y[i] = encoding
  y = np.array(_y)

  # Train / Test split
  test_x = x[int( len(x) * split_ratio ):]
  test_y = y[int( len(y) * split_ratio ):]
  targets= targets[int(len(y)*split_ratio):]

  return x, y, test_x, test_y, targets

class NeuralNetwork:
    def __init__(self):
        self.img_pixels = IMG_PIXELS

        self.X = self.define_input()
    
    def define_input(self):
        return tf.placeholder(tf.float32, shape=([None, self.img_pixels]))

    def perceptron(self, x, w, b):
        return tf.matmul(x, w) + b

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

'''
Darwinian Natural Selection
--> Variation: There is a variety of traits and or a means of mutation
--> Selection: Threshold of fitness for specific traits
--> Heredity: Children recieve parent's genetic information

Genetic Algorithm
1.) Create a random population of N genetic objects
     ex.) Predict "cat" from (tar, aat, ase, etc.)
2.) Calculate Fitness for all N elements
3.) Selection of M genetic objects
     ex.) Assign probability of selecting each object relative to Fitness
     ex.) Probability of selection for cat >> tar == .33, aas == .66, ase == .01
4.) Reproduction via some means
     ex.) tar + aas >> t|ar + a|as >> (probablilistic mutation) >> tas (del ase)
'''

class Animus(NeuralNetwork):
    def __init__(self): 
        super().__init__()

class GeneticNeuralNetwork:
    def __init__(self):
        pass

INPUT_SIZE = 10000
DATA_SIZE  = 1000
NUM_CLASSES= 200

BATCH_SIZE = int(DATA_SIZE/10)
if __name__ == "__main__":
    train_x, train_y, test_x, test_y, targets = make_data(DATA_SIZE, 
                                               INPUT_SIZE,
                                               NUM_CLASSES)
