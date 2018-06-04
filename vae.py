# Started from Tutorial by Siraj Raval
# How to Generate Images with Tensorflow (LIVE)
# @ https://youtu.be/iz-TZOEKXzA
# Developed by Nathan Shepherd

import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)


class NeuralNetwork:
    def __init__(self):
        pass
    
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

          img = Image.open(data_dir + filename)
          img_data.append(img)

        self.img_data = img_data
        print('Loaded dataset successfully')

class VAE(NeuralNetwork):
    def __init__(self, latent_dim, hidden):
        self.latent_dim = latent_dim
        self.img_pixels = IMG_PIXELS
        self.hidden = hidden

        # Build the model: Variational AutoEncoder
        self.X = self.define_input()
        self.define_encoder()
        self.define_decoder()
        self.compile_model()

    def train(self, epochs, batch_size=10):
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        sess.run(init)
        
        variational_lower_bound_array = []
        log_likelihood_array = []
        KL_term_array = []
        
        for i in range(0, epochs - batch_size, batch_size):
            x_batch = np.zeros((batch_size, IMG_PIXELS))
            
            for idx, pic in enumerate(self.img_data[i: i + batch_size]):
                x_batch[idx] = self.PIL2array(pic)

            sess.run(self.optimizer, feed_dict={self.X: x_batch})

            if i % 10 == 0:
                vlb_eval = self.variational_lower_bound.eval(feed_dict={self.X: x_batch})
                print("Iteration: {}, Loss: {}".format(i, vlb_eval), end="")
                variational_lower_bound_array.append(vlb_eval)
                #log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
                #KL_term_array.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))     

    def compile_model(self):
        #Information is lost because it goes from a smaller to a larger dimensionality. 
        #How much information is lost? We measure this using the reconstruction log-likelihood 
        #This measure tells us how effectively the decoder has learned to reconstruct
        #an input image x given its latent representation z.
        log_likelihood = tf.reduce_sum(self.X * tf.log(self.reconstruction + 1e-9)\
                                       + (1 - self.X)*tf.log(1 - self.reconstruction + 1e-9),
                                       reduction_indices = 1)

        KL_div = -.5 * tf.reduce_sum(1 + 2*self.logstd\
                                            - tf.pow(self.mu, 2)\
                                            - tf.exp(2 * self.logstd),
                                            reduction_indices = 1)

        # This allows us to use stochastic gradient descent with respect to the variational parameters
        self.variational_lower_bound = tf.reduce_mean(log_likelihood - KL_div)
        self.optimizer = tf.train.AdadeltaOptimizer().minimize(-self.variational_lower_bound)
        

    def generate(self, seed):
        if seed == None: pass

    def define_encoder(self):
        # First layer is a simple feed forward network
        w_enc = self.weight_variable([self.img_pixels, self.latent_dim], 'W_encoder')
        b_enc = self.bias_variable([self.latent_dim], 'B_encoder')
        h_enc = tf.nn.tanh(self.perceptron(self.X, w_enc, b_enc))

        # Mean
        w_mu = self.weight_variable([self.latent_dim, self.hidden[0]], 'W_mu')
        b_mu = self.bias_variable([self.hidden[0]], 'B_mu')
        self.mu = self.perceptron(h_enc, w_mu, b_mu)

        # Standard Deviation
        w_logstd = self.weight_variable([self.latent_dim, self.hidden[0]], 'W_logstd')
        b_logstd = self.bias_variable([self.hidden[0]], 'B_logstd')
        self.logstd = self.perceptron(h_enc, w_logstd, b_logstd)

        noise = tf.random_normal([10, self.hidden[-1]])
        
        # Z is the output parameter of encoder
        self.Z = self.mu + tf.matmul(tf.transpose(noise), tf.exp(.5*self.logstd))

    def define_decoder(self):
        w_dec = self.weight_variable([self.hidden[-1], self.latent_dim], 'W_decoder')
        b_dec = self.bias_variable([self.latent_dim], 'V_decoder')
        h_dec = tf.nn.tanh(self.perceptron( self.Z, # Output from encoder
                                            w_dec, b_dec))

        w_reconstruct = self.weight_variable([self.latent_dim, self.img_pixels], 'W_reconstruct')
        b_reconstruct = self.bias_variable([self.img_pixels], 'B_reconstruct')
        self.reconstruction = tf.nn.sigmoid(self.perceptron(h_dec, w_reconstruct, b_reconstruct))

    def PIL2array(self, img):
        return np.array(img.getdata(), np.uint8).reshape(1, -1)

    def array2PIL(self, vector):
        channels = 3
        dims = (IMG_PIXELS / channels) / 2
        return vector.reshape(dims,dims,3)
        
        

        
IMG_PIXELS = 28*28 * 3


if __name__ == "__main__":
    vae = VAE(500, [20])
    vae.load_dataset('images/pokemon/sample_of_10/28/')
    vae.train(100)
    
    
