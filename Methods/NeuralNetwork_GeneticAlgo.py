# # # # # # # # 
# Developed by Nathan Shepherd

import os
import time
import random
import numpy as np
#import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.datasets.samples_generator import make_blobs

def make_data(amount, x_length, num_classes=2,split_ratio=0.8):
  print('Generating data for classification')
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
  test_x  = x[-int(np.ceil( len(x) * (1 - split_ratio) )):]
  test_y  = y[-int(np.ceil( len(y) * (1 - split_ratio) )):]
  targets = targets[  :int( len(y) *  split_ratio)]
  x = x[:int( len(x) * split_ratio )]
  y = y[:int( len(y) * split_ratio )]
  
  print('Data generation complete!')
  return x, y, test_x, test_y, targets

def Define_Model(in_dims, out_dims, hidden_vect, LR=0.1):
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

# Genetic Algo HyperParameters
# --> Discrete in order to simplify search space
LATENT_DEPTH = [16, 32, 64, 128, 256, 512, 1024]
HIDDEN_LAYERS = [2, 3, 5, 7, 9, 18, 36, 64, 128]
LEARNING_RATES = [0.5, 0.25, 0.2, 0.1, 0.05, 0.001]
FITNESS_BIAS = {'train_time': 0.4,
                 'accuracy' : 0.4,
                 'stability': 0.2,}
                
# Mutation is on the order of existing value for param
# if depth == 16, then mutation(depth) \
#    is at most (depth +/- depth * MAX_MUTATION)
MAX_MUTATION = 0.5

# Mutate child if random.random() <= MUTATION_RATE
MUTATION_RATE = 0.6

def prob_dist(param):
  probs = np.zeros((len(param)))
  # Probability of selection is
  # greatest at left end of vector
  for idx in range(len(param)):
      probs[idx] = 1 + len(param) - idx
  probs = [probs[i]/sum(probs) for i in range(len(param))]
  return probs

  
class Animus():
    def __init__(self, data_shape, genetic_material=None, mutate=True):
        self.x_len, self.y_len = data_shape

        if genetic_material is None:
            self.batch_size = BATCH_SIZE
            
            depth = np.random.choice(HIDDEN_LAYERS, p=prob_dist(HIDDEN_LAYERS))
            # TODO: Endow hidden selection process with geometric properties
            #       In example: input --> 128 --> 64 --> 32 --> ouput
            
            self.hidden = [np.random.choice(LATENT_DEPTH,
                           p=prob_dist(LATENT_DEPTH)) for layer in range(depth)]
            
            self.learn_rate = np.random.choice(LEARNING_RATES,
                                               p=prob_dist(LEARNING_RATES))
            self.model = Define_Model(self.x_len, self.y_len,
                                      self.hidden, LR=self.learn_rate)
        else:
            parent = genetic_material
            if random.random() <= MUTATION_RATE and mutate is True:
                self.hidden = self.mutation('hidden', parent.hidden)
                self.learn_rate = self.mutation('scalar', parent.learn_rate)
                self.batch_size = self.mutation('scalar', parent.batch_size)
                self.model = self.mutation('model', parent)
            else:
                self.hidden = parent.hidden
                self.learn_rate = parent.learn_rate
                self.batch_size = parent.batch_size
                self.model = self.mutation('model', parent)

    def calc_fitness(self, data, epochs):
        train_x, train_y, test_x, test_y, targets = data

        start = time.time()
        self.history = self.model.fit(train_x, train_y, epochs=epochs,
                       batch_size=self.batch_size, verbose=0).history

        training_dur = time.time() - start

        #self.acc = history['acc']
        #self.loss = history['loss']
        predictions = np.array([np.argmax(t_x) for t_x in self.model.predict(test_x)])

        
      
        acc = 0.0
        for i in range(len(predictions)):
            if predictions[i] == targets[i]:
                acc += 1

        self.acc = (acc/len(targets)) * FITNESS_BIAS['accuracy']
        
        #TODO: Add metrics for training time and stability

        self.dur = training_dur * FITNESS_BIAS['train_time']
        self.fitness = self.acc - self.dur

    def mutation(self, config, switch):
        # Mutate the learning rate
        if config is 'scalar':
            sign = random.randint(0, 1);
            if not sign: sign = -1
            return switch + sign * int(switch * random.random() * MAX_MUTATION)

        # Mutate the depth of each hidden layer
        elif config is 'hidden':
            for i in range(len(switch)):
                sign = random.randint(0, 1);
                if not sign: sign = -1
                switch[i] += sign * int(switch[i] * random.random() * MAX_MUTATION)
            return switch

        # Compile the child model
        elif config is 'model':
            # TODO: Transfer model weights from parent to child for unmutated layers
            #       In example: self.model.weights = switch.model.get_weights
            model = Define_Model(self.x_len, self.y_len,
                                 self.hidden, LR=self.learn_rate)
            return model
          
        else:
            print('Mutation of the configuration key',
                  '\'%s\' is not supported'% config)
            raise TypeError
          


class GeneticNNClassifier:
    def __init__(self, zipp, data_shape, pop_size=20):
        self.zipp = zipp
        self.pop_size = pop_size
        self.data_shape = data_shape

        self.init_population()

    def train(self, epochs, child_lifetime=2):
        print("Training Neural Network population",
              "with lifetime %d per child"%child_lifetime)
        for epoch in range(epochs):
            self.Epoch(child_lifetime)
            print('Epoch',epoch+1,'of',epochs,
                  '| Greatest accuracy:',
                  self.population[0].acc)

    def init_population(self):
        self.population = []
        for child in range(self.pop_size):
            dna = Animus(self.data_shape)
            self.population.append(dna)

        print("Initializing population")
        self.calc_fitness(1)

    def calc_fitness(self, epochs):
        for i, dna in enumerate(self.population):
            if (i % 1) == 0:
                print(i*100/self.pop_size)
            dna.calc_fitness(self.zipp, epochs)

    def display_pop(self):
        # Only metric supported is accuracy
        for dna in clf.population:
            print(dna.acc)

    def Epoch(self, lifetime):
      ###########################################
        # key --> dna.acc
        self.population.sort(key=lambda dna: dna.acc, reverse=True)

        div = int(self.pop_size/3)        

        # Keep top 33% of networks
        winners = self.population[:div]
        
        # Chance to mutate middle 50%
        mutants = []
        for dna in self.population[div : int(div*1.5)]:
            parent = random.choice(winners)
            dna = Animus(self.data_shape, genetic_material= parent)
            mutants.append(dna)

        # Recombine population
        self.population = winners + mutants
        
        # Replace bottom 25% of networks
        while len(self.population) < self.pop_size:
            dna = Animus(self.data_shape)
            self.population.append(dna)

        
        self.calc_fitness(lifetime)

# Data Parameters
INPUT_SIZE = 10000
DATA_SIZE  = 100
NUM_CLASSES= 10
BATCH_SIZE = int(DATA_SIZE/10)

'''
TODO:
  Impliment an unsupervised preprocessing step
  Add additional logging statistics and visualizations
    graph the loss and acc for each dna sample in the population per epoch
    viz the size and shape of an average network over time
'''
if __name__ == "__main__":
    zipped = make_data(DATA_SIZE, INPUT_SIZE, NUM_CLASSES)
    train_x, train_y, test_x, test_y, targets = zipped

    data_shape = (len(train_x[0]), len(train_y[0]))

    clf = GeneticNNClassifier(zipped, data_shape, 10)
    clf.train(100, 5)

    
    # Define a genetic body like this:
    #     a = Animus(data_shape, None)

    # Reproduction: pass the parent material into the initializer
    #     a = Animus(data_shape, a)

    # Fitness
    #     a.calc_fitness(zipped, 2)
    #     print(a.acc)

    



















