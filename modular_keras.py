# Quickly deploy a keras NN to any given specifications

import numpy as np
from sklearn.datasets.samples_generator import make_blobs

from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten


def define_model(in_dims, out_dims, hidden_vect, LR=0.1):
  #TODO: add regularization
  model = Sequential()
  model.add(Dense(hidden_vect[0], input_shape=(1,in_dims,)))
  model.add(Activation('linear'))
  model.add(Flatten())

  for layer in hidden_vect[1:]:
    model.add(Dense(layer))
    model.add(Activation('linear'))

  adam = Adam(lr=LR)
  model.compile(loss='mse', optimizer=adam, 
                metrics=['accuracy'])
  print(model.summary())

  return model

def make_data(amount, x_length, num_classes=2,split_ratio=0.8):
  # Generate samples
  x, y = make_blobs(n_samples = amount, 
                    centers   = num_classes,
                    n_features= x_length )

  # Encode Y as one-hot vectors
  for i in range(len(y)):
    encoding = np.zeros(num_classes)
    y[i] = encoding[y[i]]

  # Train / Test split
  test_x = x[int(len(x) * split_ratio):]
  test_y = y[int(len(y) * split_ratio):]

  return x, y, test_x, test_y
  
INPUT_SIZE = 4
DATA_SIZE  = 100
NUM_CLASSES= 2

if __name__ == "__main__":
  train_x, train_y, test_x, test_y = make_data(DATA_SIZE, INPUT_SIZE,
                                               NUM_CLASSES)
  hidden = [10, 5]

  model = define_model(INPUT_SIZE, NUM_CLASSES, hidden)
