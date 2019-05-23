 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
plt.switch_backend("TkAgg")

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

def viz_data(x, y):
  colors = ('r', 'b', 'g', 'o','v','k')
  for i in range(len(x)):
    _y = np.argmax(y[i])
    plt.scatter(x[i][0], x[i][1], c=colors[_y])
  plt.title('Generated Data')
  plt.show()

INPUT_SIZE = 2
DATA_SIZE  = 10
NUM_CLASSES= 2


if __name__ == "__main__":
  #TODO: Generate data from different modules
  # @ https://bit.ly/2Kkk9wn

  train_x, train_y, test_x, test_y, targets = make_data(DATA_SIZE, 
                                               INPUT_SIZE,
                                               NUM_CLASSES)
  for i in range(DATA_SIZE):
    print(train_x[i], "\t", np.argmax(train_y[i]))

  viz_data(train_x, train_y)

  with open("input.txt", "w") as f:
    f.write(str(DATA_SIZE) + " ")
    f.write(str(INPUT_SIZE) + " ")
    f.write(str(NUM_CLASSES) + '\n')
    for idx in range(DATA_SIZE):
      for xi in range(INPUT_SIZE):
        f.write(str(train_x[idx][xi]) + " ")
      for yi in range(NUM_CLASSES):
        if train_y[idx][yi] == 1:
          f.write(str(yi) + '\n')
          break;

#?
