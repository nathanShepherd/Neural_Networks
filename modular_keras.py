# Quickly deploy a keras NN to any given specifications

import numpy as np
from sklearn.datasets.samples_generator import make_blobs

from keras.models import Model
from keras.layers import Dense, Input

def define_model(in_dims, out_dims, hidden_vect, LR=0.1):
  # Logistic regression
  inputs = Input(shape = (in_dims,))
  x = Dense(hidden_vect[0], activation='sigmoid')(inputs)

  for layer in hidden_vect[1:]:
    x = Dense(layer, activation='relu')(x)

  y = Dense(out_dims, activation = 'softmax')(x)
  model = Model(inputs=inputs, outputs=y)

  model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
  return model

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
  
INPUT_SIZE = 10000
DATA_SIZE  = 1000
NUM_CLASSES= 200

BATCH_SIZE = int(DATA_SIZE/10)

if __name__ == "__main__":
  #TODO: Generate data from different modules
  # @ https://bit.ly/2Kkk9wn

  train_x, train_y, test_x, test_y, targets = make_data(DATA_SIZE, 
                                               INPUT_SIZE,
                                               NUM_CLASSES)
  hidden = [10]

  model = define_model(INPUT_SIZE, NUM_CLASSES, hidden)

  #print(train_x)
  #print(train_y)

  #train_x = np.expand_dims(train_x, axis= 1)
  #train_y = np.stack(train_y, axis= 0)
  model.fit(train_x, train_y, epochs=50, batch_size=BATCH_SIZE, verbose=2)
  #Verbosity modes: 0=silent, 1=progress bar, 2=one line per epoch.
  
  #print(test_x) 
  predictions = np.array([np.argmax(t_x) for t_x in model.predict(test_x)])
 
  print len(targets), targets[:10], '. . .'
  print len(predictions), predictions[:10], '. . .'

  acc = 0.0
  for i in range(len(predictions)):
    if predictions[i] == targets[i]:
      acc += 1
  print 'Accuracy:', acc/len(targets)




#?
