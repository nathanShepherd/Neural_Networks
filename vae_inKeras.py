# Buildong AutoEncoders in Keras: Variational AutoEncoder
# Source: https://blog.keras.io/building-autoencoders-in-keras.html
# # # # # #
# Developed by Nathan Shepherd

import os
import numpy as np
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Dense, Input, Lambda
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Base_Class():
    def __init__(self):
        pass

    def binarize(self, img, threshold=200):
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i][j] /= threshold
        return img

    def load_dataset(self, data_dir, mnist=False):
        self.img_data = []
        print('Loading dataset')
        if mnist:
            for i, sub_dir in enumerate(os.listdir(data_dir)):
                print(i*100.0/len(os.listdir(data_dir)))
                sub_dir += '/'
                
                for k, filename in enumerate(os.listdir(data_dir + sub_dir)):
                    if k % (700) == 0:
                      print('-->',k*100.0/(len(os.listdir(data_dir + sub_dir))))
                    img = plt.imread(data_dir + sub_dir + filename)

                    #img = self.binarize(img)
                  
                    self.img_data.append(img)

            print('Loaded dataset successfully'); return 0
            
        for i, filename in enumerate(os.listdir(data_dir)):
          if i % (50) == 0:
              print(i*100/(len(os.listdir(data_dir))))
              
          if len(filename.split('.')) > 1:
              #img = Image.open(data_dir + filename)
              img = plt.imread(data_dir + filename)

              img = self.binarize(img)
              
              self.img_data.append(img)
          
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

class VarAutoEnc(Base_Class):
    def __init__(self):
        super().__init__()

        x, z_mean, z_log_sigma = self.Encoder([256])
        x_decoded_mean, dec_mu, decoder_input = self.Decoder([256], z_mean, z_log_sigma)

        encoder = Model(x, z_mean)

        generator = Model(decoder_input, dec_mu)

        self.vae = Model(x, x_decoded_mean)
        self.vae.compile(optimizer='rmsprop', loss=self.vae_loss)
        

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean( \
            1 + self.z_log_sigma - K.square(self.z_mean) - K.exp(self.z_log_sigma), axis=-1)
        return xent_loss + kl_loss        

    def Encoder(self, hidden, non_lin='relu'):
        x = Input(batch_shape=(BATCH_SIZE, IMG_PIXELS))
        h = Dense(hidden[0], activation=non_lin)(x)

        for num_dims in hidden[1:]:
            h = Dense(num_dims, activation=non_lin)(h)

        z_mean =  Dense(NOISE_DIM)(h)
        z_log_sigma = Dense(NOISE_DIM)(h)
        self.z_log_sigma = z_log_sigma
        self.z_mean = z_mean
        return x, z_mean, z_log_sigma

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal_variable(shape=(BATCH_SIZE, NOISE_DIM),
                                           mean= 0.0, scale= 1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def Decoder(self, hidden, z_mean, z_log_sigma, non_lin='relu'):
        z = Lambda(self.sampling, output_shape=(NOISE_DIM,))([z_mean, z_log_sigma])

        decoder_h = Dense(hidden[0], activation=non_lin)
        for num_dims in hidden[1:]:
            decoder_h = Dense(num_dims, activation=non_lin)(decoder_h)
            
        decoder_mean = Dense(IMG_PIXELS, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        decoder_input = Input(shape=(NOISE_DIM,))
        dec_h = decoder_h(decoder_input)
        dec_mu = decoder_mean(dec_h)
        
        return x_decoded_mean, dec_mu, decoder_input
        


NOISE_DIM = 100
BATCH_SIZE = 3
IMG_PIXELS = 28*28
IMG_SHAPE = (1, IMG_PIXELS)
             #IMG_PIXELS,)
             #NUM_CHANNELS)

if __name__ == "__main__":
    vae = VarAutoEnc()
    vae.load_dataset('images/pokemon/sample_of_10/monochrome/28/')
    #vae.load_dataset('images/pokemon/orig/monochrome/28/')
    #vae.load_dataset('images/mnist/mnist_png/training/', mnist=True)
    vae.viz_dataset(10)

























    
        
