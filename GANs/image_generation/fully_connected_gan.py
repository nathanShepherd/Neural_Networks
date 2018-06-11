##Source codes cited:
# niazangels / vae-pokedex / pokedex.ipynb
##@ https://github.com/niazangels/vae-pokedex/blob/master/pokedex.ipynb
# Zackory / Keras-MNIST-GAN / mnist_gan.py
##@ https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_gan.py
# # # # # # # # 
# Developed by Nathan Shepherd

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import *
from keras import initializers
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU

from skimage import data
from skimage.transform import resize
from skimage.util import view_as_blocks

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
        hidden = [IMG_PIXELS**2 * NUM_CHANNELS,
                  IMG_PIXELS**2, 1000, 100,
                  100]
        self.discriminator = self.build_model( [self.Discriminator(hidden)],
                                               loss='binary_crossentropy')
        
        hidden = [NOISE_DIM*5, NOISE_DIM*10,
                  IMG_PIXELS**3 * NUM_CHANNELS,
                  IMG_PIXELS**2 * NUM_CHANNELS]
        
        self.generator = self.build_model([ self.Generator(hidden) ],
                                          loss='binary_crossentropy')

        self.gan = self.adversarial(0.0001)

    def adversarial(self, LR):
        self.discriminator.trainable = False
        ganInput = Input(shape=(NOISE_DIM,))
        
        x = self.generator(ganInput)
        ganOutput = self.discriminator(x)

        opt = Adam(lr=LR, beta_1=0.5)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
        self.gan = gan
        return gan
        
        
    def viz_img_gen(self, num_images):
        noise = np.random.uniform(-1.0, 1.0, size=(num_images, NOISE_DIM))
        img_fakes = self.generator.predict(noise).reshape((num_images,)+IMG_SHAPE)
        self.viz_dataset(num_images, images=img_fakes)

    def train(self, epochs=10, batch_size=7, save_interval=0):
        
        for ep in range(epochs):
            img_train = [random.choice(self.img_data) for dim in range(batch_size)]

            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, NOISE_DIM))
            
            img_fakes = self.generator.predict(noise)
            
            
            self.discriminator.trainable = True

            X = np.concatenate([img_train, img_fakes])
            # Images are real (0) or fake (1)
            y = np.zeros(2* batch_size);
            y[:batch_size] = 0.9
            
            d_loss = self.discriminator.train_on_batch(X, y)

            # Train Adversarial Model (including generator)
            # TODO: Add freezing in self.D layers when training adversarial
            # ALSO: Why is the adversarial model training on just noise?
            #       Should there be a generation component for this step?
            self.discriminator.trainable = False
            
            y = [0.9 for dim in range(batch_size)]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
            a_loss = self.gan.train_on_batch(noise, y)

            print("Epoch", ep, "of", epochs,
                  "Discr: [ loss: %f, acc: %f]"% (d_loss[0],d_loss[1]),
                  "Adv: [loss: %f, acc:]"% (a_loss[0],))
                  
        print("Completed Training!")

    def build_model(self, modules, loss=None,
                        LR= 0.0001, clipvalue= 255.0):
        model = Sequential()
        for mod in modules:
            model.add(mod)

        #TODO: Try other optimization
        optimizer = Adam(lr=LR, beta_1=0.5)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    def Generator(self, hidden, dropout=0.4):
        G = Sequential()
        
        G.add(Dense(hidden[0], input_dim=NOISE_DIM,
                    kernel_initializer=initializers.RandomNormal(stddev=0.02)))    
        G.add(BatchNormalization(momentum = 0.9))
        G.add(LeakyReLU(alpha=0.2))
        G.add(Dropout(dropout))

        for hidden_units in hidden[1:]:
            G.add(Dense(hidden_units))
            G.add(LeakyReLU(alpha=0.2))
            G.add(Dropout(dropout))
            
        G.add(Reshape(IMG_SHAPE))
        print(G.summary())

        self.G = G
        return G
        
        
    def Discriminator(self, hidden, dropout=0.4):
        D = Sequential()
        input_shape = IMG_SHAPE

        D.add(Dense(hidden[0], input_shape=input_shape,
                    kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        D.add(LeakyReLU(alpha=0.2))
        D.add(Dropout(dropout))

        for hidden_units in hidden[1:]:
            D.add(Dense(hidden_units))
            D.add(LeakyReLU(alpha=0.2))
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
    gan.load_dataset('./../../images/pokemon/orig/28/')
    #gan.viz_dataset(10)
    gan.train(500)

    gan.viz_img_gen(10)
                                
    
    
