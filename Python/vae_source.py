##
# Variational-Autoencoder source code:
# GitHub / bluebelmont / Variational-Autoencoder
# @ https://bit.ly/2Kh4FwG
##
# Related work: Optimizing the Latent Space of Generative Networks
# @ https://arxiv.org/pdf/1707.05776.pdf
##
# # # # # # # # 
# Developed by Nathan Shepherd

import os
import time
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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

    def binarize(self, img, threshold=200):
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] > threshold:
                    img[i][j] = 1
                else:
                    img[i][j] = 0
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

class VarAutoEnc(NeuralNetwork):
    def __init__(self): 
        super().__init__()
        
        self.Encoder([500, 1000])
        self.Decoder([1000, 500])
        self.Init_Optimizer()

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        self.sess.run(init)
        

    def Init_Optimizer(self):
        # We want to maximize this lower bound,
        #   but because tensorflow doesn't have a 'maximizing' optimizer,
        #   we minimize the negative lower bound.
        # Add epsilon to log to prevent numerica overflow.
        self.log_likelihood = tf.reduce_sum(self.X*\
                                tf.log(self.reconstruction + 1e-9)+(1 - self.X)*\
                                       tf.log(1 - self.reconstruction + 1e-9), reduction_indices=1)
        self.KL_term = -.5*tf.reduce_sum(1 + 2*self.logstd \
                                      - tf.pow(self.mu,2) \
                                      - tf.exp(2*self.logstd), reduction_indices=1)
        self.variational_lower_bound = tf.reduce_mean(self.log_likelihood - self.KL_term)
        self.optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(-self.variational_lower_bound)
        
    def Decoder(self, hidden):
        W_dec = self.weight_variable([hidden[0], hidden[-1]], 'W_dec')
        b_dec = self.bias_variable([hidden[-1]], 'b_dec')
        h_dec = tf.nn.tanh(self.perceptron(self.z, W_dec, b_dec))

        W_reconstruct = self.weight_variable([hidden[-1], self.img_pixels], 'W_reconstruct')
        b_reconstruct = self.bias_variable([self.img_pixels], 'b_reconstruct')
        self.reconstruction = tf.nn.sigmoid(self.perceptron(h_dec, W_reconstruct, b_reconstruct))
        

    def Encoder(self, hidden):
        # TODO: Pixel to convolutions
        # Pixel to Hidden space
        W_enc = self.weight_variable([self.img_pixels, hidden[0]], 'W_enc')
        b_enc = self.bias_variable([hidden[0]], 'b_enc')
        h_enc = tf.nn.tanh(self.perceptron(self.X, W_enc, b_enc))

        # Hidden to the mean of a latent space
        W_mu = self.weight_variable([hidden[0], hidden[-1]], 'W_mu')
        b_mu = self.bias_variable([hidden[-1]], 'b_mu')
        self.mu = self.perceptron(h_enc, W_mu, b_mu)

        # Hidden to the log(stddev) of a latent space
        W_logstd = self.weight_variable([hidden[0], hidden[-1]], 'W_logstd')
        b_logstd = self.bias_variable([hidden[-1]], 'b_logstd')
        self.logstd = self.perceptron(h_enc, W_logstd, b_logstd)

        # reparameterization trick
        noise = tf.random_normal([1, hidden[-1]])
        self.z = self.mu + tf.multiply(noise, tf.exp(.5*self.logstd))

        

    def train(self, epochs, batch_size=100, log_interval=100):
        iteration_array = [i*log_interval for i in range(int(epochs/log_interval))]
        variational_lower_bound_array = []
        log_likelihood_array = []
        KL_term_array = []

        prev_vlb = 0
        for i in range(epochs):
            img_train = [random.choice(self.img_data) for dim in range(batch_size)]
            x_batch = [img.flatten() for img in img_train]
            self.sess.run(self.optimizer, feed_dict={self.X: x_batch})
            
            if (i % log_interval) == 0:
                vlb_eval = self.variational_lower_bound.eval(feed_dict={self.X: x_batch})
                deriv_vlb = prev_vlb - vlb_eval; prev_vlb = vlb_eval
                print("Iteration: %d | %d"% (i, epochs),
                      "Loss: %d"    % int(vlb_eval),
                      "Delta: %d"   % int(deriv_vlb))
                variational_lower_bound_array.append(vlb_eval)
                log_likelihood_array.append(np.mean(self.log_likelihood.eval(feed_dict={self.X: x_batch})))
                KL_term_array.append(np.mean(self.KL_term.eval(feed_dict={self.X: x_batch})))

        plt.figure()
        plt.plot(iteration_array, variational_lower_bound_array)
        plt.plot(iteration_array, KL_term_array)
        plt.plot(iteration_array, log_likelihood_array)
        plt.legend(['Variational Lower Bound',
                    'KL divergence',
                    'Log Likelihood'])#, bbox_to_anchor=(1.05, 1), loc=2)
        plt.title('Loss per iteration')
        plt.show()
        
        self.viz_reconstruction(10)

    def viz_reconstruction(self, num):
        plt.figure()
        images = []
        for i in range(num):
            x = np.zeros(IMG_SHAPE)
            x[0] = random.choice(self.img_data).flatten()
            gen_img = self.reconstruction.eval(feed_dict={self.X: x})
            gen_img = (np.reshape(gen_img, (28,28)))
            images.append(gen_img)
        
        self.viz_dataset(num, images=images)

        
        
NOISE_DIM = 100
IMG_PIXELS = 28*28
NUM_CHANNELS = 1 #3
IMG_SHAPE = (1, IMG_PIXELS)
             #IMG_PIXELS,)
             #NUM_CHANNELS)

# TODO: Add the ability to save/load trained models
# TODO: Train for 10 ** 6 epochs on GPU or in the cloud
if __name__ == "__main__":
    vae = VarAutoEnc()
    vae.load_dataset('images/pokemon/orig/monochrome/28/')
    #vae.load_dataset('images/mnist/mnist_png/training/', mnist=True)
    vae.viz_dataset(10)
    vae.train(10 ** 4)#10 ** 5

