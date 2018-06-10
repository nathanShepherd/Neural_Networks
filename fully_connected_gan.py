# Started from How to autoencode your Pokémon
# niazangels / vae-pokedex / pokedex.ipynb
# @ https://github.com/niazangels/vae-pokedex/blob/master/pokedex.ipynb
# Developed by Nathan Shepherd

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize
import IPython.display as ipyd
import tensorflow as tf
from PIL import Image

from keras import optimizers
from keras.models import Model
from keras.models import Sequential
from keras.layers import *

from skimage.util import view_as_blocks
from skimage.transform import resize
from skimage import data

from libs import utils
plt.style.use('ggplot')

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


class NeuralNetwork:
    def __init__(self):
        self.img_pixels = IMG_PIXELS
    
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

    def load_dataset(self, data_dir):
        img_data = []
        print('Loading dataset')
        for i, filename in enumerate(os.listdir(data_dir)):
          if i % (50) == 0:
              print(i*100/(len(os.listdir(data_dir))))
              
          if len(filename.split('.')) > 1:
              #img = Image.open(data_dir + filename)
              img = plt.imread(data_dir + filename)
              img_data.append(img)
          

        self.img_data = img_data
        print('Loaded dataset successfully')

        
    def viz_dataset(self, num, as_grid=True, images=None):
        fig = plt.figure(figsize = (5, 5))
        
        if images is None:
            images = self.img_data[:min(len(self.img_data), num)]
        else:
            images = images[:min(len(images), num)]
        
        if as_grid:
            cols = int(len(images)/3); rows = int(len(images)/2)
            for idx in range(1, len(images)):
                fig.add_subplot(rows, cols, idx).axis('off')
                plt.imshow(images[idx])
            plt.show()
            
        else: [[plt.imshow(image), plt.show()] for image in images]


class FCGAN(NeuralNetwork):
    def __init__(self):        
        hidden = [IMG_PIXELS**3, IMG_PIXELS**2,
                  1000, 100, 100]
        self.discriminator = self.build_model( [self.Discriminator(hidden)],
                                               loss='binary_crossentropy')
        
        hidden = [NOISE_DIM*5, NOISE_DIM*10,
                  IMG_PIXELS**3 * NUM_CHANNELS,
                  IMG_PIXELS**2 * NUM_CHANNELS]
        self.G = self.Generator(hidden, IMG_SHAPE)
        self.generator = self.build_model([ self.G ],
                                          loss='mean_squared_error')

        self.adversarial = self.build_model([self.G, self.D],
                                            loss='mean_squared_error')

    def viz_img_gen(self, num_images):
        noise = np.random.uniform(-1.0, 1.0, size=(num_images, NOISE_DIM))
        img_fakes = self.generator.predict(noise)
        self.viz_dataset(num_images, images=img_fakes)

    def train(self, epochs=10, batch_size=7, save_interval=0):
        # Random uniform selection distribution for each batch
        rand_uni = [1/batch_size for i in range(batch_size)]
        
        for ep in range(epochs):
            img_train = [random.choice(self.img_data) for dim in range(batch_size)]

            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, NOISE_DIM))
            
            img_fakes = self.generator.predict(noise)
            
            # Label training data as real (1) or fake (0)
            self.discriminator.trainable = True
            
            y = [0.9 for dim in range(batch_size)]
            d_loss = self.discriminator.train_on_batch(np.array(img_train), y)

            y = [0 for dim in range(batch_size)]
            d_loss += self.discriminator.train_on_batch(img_fakes, y)
            

            # Train Adversarial Model (including generator)
            # TODO: Add freezing in self.D layers when training adversarial
            # ALSO: Why is the adversarial model training on just noise?
            #       Should there be a generation component for this step?
            self.discriminator.trainable = False
            
            y = [0.9 for dim in range(batch_size)]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
            a_loss = self.adversarial.train_on_batch(noise, y)

            print("Epoch", ep, "of", epochs,
                  "Discr: [ loss: %f, acc: %f]"% (d_loss[0],d_loss[1]),
                  "Adv: [loss: %f, acc: %f]"% (a_loss[0], a_loss[1]))
                  
        print("Completed Training!")

    def build_model(self, modules, loss=None,
                        LR= 0.1, clipvalue= 255.0, decay= 3e-8):
        model = Sequential()
        for mod in modules:
            model.add(mod)

        #TODO: Try other optimization
        optimizer = optimizers.RMSprop(lr=LR, clipvalue=clipvalue, decay= decay)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    def Generator(self, hidden, output_dim, dropout=0.4):
        G = Sequential()
        
        G.add(Dense(hidden[0], input_dim=NOISE_DIM))        
        G.add(BatchNormalization(momentum = 0.9))
        G.add(Activation('sigmoid'))
        G.add(Dropout(dropout))

        for hidden_units in hidden[1:]:
            G.add(Dense(hidden_units))
            G.add(Activation('relu'))
            G.add(Dropout(dropout))
            
        G.add(Reshape(output_dim))
        print(G.summary())

        self.G = G
        return G
        
        
    def Discriminator(self, hidden, dropout=0.4):
        D = Sequential()
        input_shape = (IMG_PIXELS,
                       IMG_PIXELS,
                       NUM_CHANNELS)

        D.add(Dense(hidden[0], input_shape=input_shape))
        D.add(Activation('relu'))
        D.add(Dropout(dropout))

        for hidden_units in hidden[1:]:
            D.add(Dense(hidden_units))
            D.add(Activation('relu'))
            D.add(Dropout(dropout))

        D.add(Flatten())
        D.add(Dense(1))# Output: Real or Fake
        D.add(Activation('sigmoid'))
        print(D.summary())

        self.D = D
        return self.D

NOISE_DIM = 100
IMG_PIXELS = 28
NUM_CHANNELS = 3
IMG_SHAPE = (IMG_PIXELS,
             IMG_PIXELS,
             NUM_CHANNELS)

'''
TODO:
    Convert inputs into binary (black and white) images
    Use convolution in discriminator
    Use style transfer to add color and sharpen black and white output
'''
if __name__ == "__main__":
    gan = FCGAN()
    gan.load_dataset('images/pokemon/sample_of_10/28/')
    #gan.viz_dataset(10)
    gan.train(5)

    gan.viz_img_gen(10)
                                
    
    
