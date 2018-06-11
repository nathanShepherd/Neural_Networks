# Source code from "GAN by Example" Blog by Atienza
# @ https://bit.ly/2HNHHbJ
# Written by Nathan Shepherd

import os
import numpy as np
from PIL import Image

from keras import optimizers
from keras.models import Sequential
from keras.layers import *

class DCGAN:
    def __init__(self):
        num_filters = 64
        hidden = [num_filters* 1, num_filters* 2,
                  num_filters* 4, num_filters* 8]
        self.discriminator = self.build_model( [self.Discriminator(hidden)] )

        hidden = [num_filters* 2, num_filters* 1,
                  int(num_filters/ 2), int(num_filters/ 4)]
        self.G = self.Generator(hidden)# Generator

        self.adversarial = self.build_model([self.G, self.D])

    def train(self, epochs=10, batch_size=2, save_interval=0):
        # Random uniform selection distribution for each batch
        rand_uni = [1/batch_size for i in range(batch_size)]
        
        for i in range(epochs):
            img_train = np.random.choice(self.img_data, batch_size, p=rand_uni)

            noise = np.random.uniform(-1.0, 1.0, size=(batch_size, 100))
            img_fakes = self.generator.predict(noise)

            # Train Discriminator
            x = np.concatenate((img_train, img_fakes))
            y = np.ones((2*batch_size, 1)); y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # Train Adversarial Model (including generator)
            # TODO: Add freezing in self.D layers when training adversarial
            # ALSO: Why is the adversarial model training on just noise?
            #       Should there be a generation component for this step?
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)


    def build_model(self, modules, loss='binary_crossentropy',
                        LR= 0.05, clipvalue= 1.0, decay= 3e-8):
        model = Sequential()
        for mod in modules:
            model.add(mod)

        #TODO: Try ADAM as optimization
        optimizer = optimizers.RMSprop(lr=LR, clipvalue=clipvalue, decay= 33-8)
        
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
        return model

    def Discriminator(self, hidden, dropout=0.4, kernel_size=5):        
        D = Sequential()
        input_shape = (IMG_PIXELS,
                       IMG_PIXELS,
                       NUM_CHANNELS)
        
        D.add(Conv2D(hidden[0], kernel_size,
                     strides=2, input_shape= input_shape,
                     padding='same', activation= LeakyReLU(alpha=0.2)))
        
        D.add(Dropout(dropout))

        for num_filters in hidden[1:]:
            D.add(Conv2D(num_filters, kernel_size, strides= 2,
                         padding='same',activation= LeakyReLU(alpha= 0.2)))
            
            D.add(Dropout(dropout))

        D.add(Flatten())
        D.add(Dense(1))# Output: Real or Fake
        D.add(Activation('sigmoid'))
        print(D.summary())

        self.D = D
        return self.D

    def Generator(self, hidden, input_dim=100, dropout=0.4, kernel_size=5):
        G = Sequential()

        G.add(Dense(hidden[0], input_dim=input_dim))
        G.add(BatchNormalization(momentum = 0.9))
        G.add(Activation('relu'))

        #print(G.output_shape)
        G.add(Reshape((1, 1, G.output_shape[1])))
        G.add(Dropout(dropout))

        for i, num_filters in enumerate(hidden[1:]):
            if i != len(hidden) - 1: G.add(UpSampling2D(size=((hidden[i]))))
            G.add(Conv2DTranspose(num_filters, kernel_size, padding='same'))
            G.add(BatchNormalization(momentum=0.9))
            G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        G.add(Conv2DTranspose(1, kernel_size, padding='same'))
        G.add(Activation('sigmoid'))
        print(G.summary())
        return G
       

    def load_dataset(self, data_dir):
        img_data = []
        print('Loading dataset')
        for i, filename in enumerate(os.listdir(data_dir)):
          if i % (50) == 0:
              print(i*100/(len(os.listdir(data_dir))))

          img = Image.open(data_dir + filename)
          img_data.append(img)

        self.img_data = img_data
        print('Loaded dataset successfully')

IMG_PIXELS   = 28
NUM_CHANNELS = 2# 3

if __name__ == "__main__":
    gan = DCGAN()
    gan.load_dataset('./../../Neural_Networks/images/pokemon/sample_of_10/28/')
    gan.train()
