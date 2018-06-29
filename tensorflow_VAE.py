## References
# Convolutional Neural Networks Tutorial in TensorFlow
# @ http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
##
# GitHub / bluebelmont / Variational-Autoencoder
# @ https://bit.ly/2Kh4FwG
##
# Related work: Optimizing the Latent Space of Generative Networks
# @ https://arxiv.org/pdf/1707.05776.pdf
# # # # # # # # 
# Developed by Nathan Shepherd

import os
import time
import random
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class NeuralNetwork:
    def __init__(self):
        pass

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.03)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def load_mnist(self, data_dir, compressed=False):
        self.img_data = []
        print('Loading dataset')
        if not compressed: #MNIST
            for i, sub_dir in enumerate(os.listdir(data_dir)):
                # sub_dir == [i for i in range(10)]
                print(i*100.0/len(os.listdir(data_dir)))
                sub_dir += '/'

                digit_data = []
                if sub_dir[:-1] != 'mnist_compressed':
                    for k, filename in enumerate(os.listdir(data_dir + sub_dir)):
                        if k % (700) == 0:
                            print('-->',k*100.0/(len(os.listdir(data_dir + sub_dir))))
                        img = plt.imread(data_dir + sub_dir + filename)
                        digit_data.append(img)

                    filename = data_dir +'mnist_compressed/' +sub_dir[:-1] +".pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(digit_data, f)
                    
            print('Loaded dataset successfully'); return 0

        if compressed is True:
            self.img_data = []
            data_dir = data_dir +'/mnist_compressed/'
            for k, filename in enumerate(os.listdir(data_dir)):
                print(k*100/len(os.listdir(data_dir)))
                with open(data_dir + filename, 'rb') as f:
                    digit_data = pickle.load(f)

                for img in digit_data:
                    # One-Hot encode digit class
                    y_str = filename.split('.')[0]
                    y = np.zeros((10)); y[int(y_str)] = 1
                    self.img_data.append({'x':img,'y':y})
                    
            print('Loaded dataset successfully'); return 0

    def viz_dataset(self, num, as_grid=True, images=None):
        fig = plt.figure(figsize = (5, 5))
        
        if images is None:
            images = [random.choice(self.img_data)['x'] \
                      for i in range(min(len(self.img_data), num))]
        else:
            images = images[:min(len(images), num)]
        
        if as_grid:
            rows = int(len(images)/2)
            cols = int(len(images)/4)
            for idx in range(1, len(images)):
                fig.add_subplot(rows, cols, idx).axis('off')
                plt.imshow(images[idx])
            plt.show()

class VarAutoEnc(NeuralNetwork):
    def __init__(self): 
        self.img_pixels = IMG_PIXELS
        self.Encoder()
        self.Decoder([256, 512])
        self.Init_Optimizer()

        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.sess.run(init)

        # For logging
        self.variational_lower_bound_array = []
        self.log_likelihood_array = []
        self.KL_term_array = []

    def Init_Optimizer(self):
        # We want to maximize this lower bound,
        #   but because tensorflow doesn't have a 'maximizing' optimizer,
        #   we minimize the negative lower bound.
        # Add epsilon to log to prevent numerical overflow.
        self.log_likelihood = tf.reduce_sum(self.X*\
                                tf.log(self.reconstruction + 1e-9)+(1 - self.X)*\
                                       tf.log(1 - self.reconstruction + 1e-9), reduction_indices=1)
        self.KL_term = -.5*tf.reduce_sum(1 + 2*self.logstd \
                                      - tf.pow(self.mu,2) \
                                      - tf.exp(2*self.logstd), reduction_indices=1)
        self.variational_lower_bound = tf.reduce_mean(self.log_likelihood - self.KL_term)
        self.optimizer = tf.train.AdadeltaOptimizer(0.01).minimize(-self.variational_lower_bound)

    def Decoder(self, hidden):
        W_dec = self.weight_variable([NOISE_DIM, hidden[0]], 'W_dec')
        b_dec = self.bias_variable([hidden[0]], 'b_dec')
        h_dec = tf.nn.relu(tf.matmul(self.z, W_dec) + b_dec)

        # More layers stabalize training at the cost of compute time
        W_dec2 = self.weight_variable([hidden[0], hidden[1]], 'W_dec2')
        b_dec2 = self.bias_variable([hidden[1]], 'b_dec2')
        h_dec2 = tf.nn.relu(tf.matmul(h_dec, W_dec2) + b_dec2)

        W_reconstruct = self.weight_variable([hidden[1], self.img_pixels], 'W_reconstruct')
        b_reconstruct = self.bias_variable([self.img_pixels], 'b_reconstruct')
        self.reconstruction = tf.nn.sigmoid(tf.matmul(h_dec2, W_reconstruct) + b_reconstruct)

    def Encoder(self):
        # Inputs: Monochrome Pixels
        self.X = tf.placeholder(tf.float32, shape=([None, IMG_PIXELS]))
        self.x_shaped = tf.reshape(self.X, [-1, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

        # Pixel to Convolutional filters
        layer1 = self.Conv_Layer(self.x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
        layer2 = self.Conv_Layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
        flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

        # Convolution filters to hidden space
        wd1 = self.weight_variable([7 * 7 * 64, 1000], name='wd1')
        bd1 = self.bias_variable([1000],  name='bd1')
        dense_layer1 = tf.matmul(flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)

        wd2 = self.weight_variable([1000, 512], name='wd2')
        bd2 = self.bias_variable([512], name='bd2')
        h_enc = tf.matmul(dense_layer1, wd2) + bd2
        h_enc = tf.nn.relu(h_enc)

        # Hidden to the mean of a latent space
        W_mu = self.weight_variable([512, NOISE_DIM], 'W_mu')
        b_mu = self.bias_variable([NOISE_DIM], 'b_mu')
        self.mu = tf.matmul(h_enc, W_mu) + b_mu

        # Hidden to the log(stddev) of a latent space
        W_logstd = self.weight_variable([512, NOISE_DIM], 'W_logstd')
        b_logstd = self.bias_variable([NOISE_DIM], 'b_logstd')
        self.logstd = tf.matmul(h_enc, W_logstd) + b_logstd

        # Reparameterization trick
        noise = tf.random_normal([1, NOISE_DIM])
        self.z = self.mu + tf.multiply(noise, tf.exp(.5*self.logstd))
        
        

    def Conv_Layer(self, input_data, num_input_channels,
                         num_filters, filter_shape, pool_shape, name):
        
        conv_filt_shape = [filter_shape[0], filter_shape[1],
                           num_input_channels, num_filters]
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape,
                                                  stddev=0.03),name=name+'_W')
        bias = self.bias_variable([num_filters], name=name+'_b')

        out_layer = tf.nn.conv2d(input_data, weights,
                                 [1, 1, 1, 1], padding='SAME')

        out_layer += bias; out_layer = tf.nn.relu(out_layer)

        # max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize,
                                   strides=strides, padding='SAME')
        return out_layer

    def train(self, epochs, batch_size=100, log_interval=10):        
        prev_vlb = 0
        
        for i in range(epochs):
            img_train = [np.array(random.choice(self.img_data)['x']) \
                                                  for dim in range(batch_size)]
            x_batch = [img.flatten() for img in img_train]
            self.sess.run(self.optimizer, feed_dict={self.X: x_batch})
            
            if (i % log_interval) == 0:
                vlb_eval = self.variational_lower_bound.eval(feed_dict={self.X: x_batch})
                deriv_vlb = prev_vlb - vlb_eval; prev_vlb = vlb_eval
                print("Iteration: %d | %d"% (i, epochs),
                      "Loss: %d"    % int(vlb_eval),
                      "Delta: %d"   % (deriv_vlb))
                self.variational_lower_bound_array.append(vlb_eval)
                self.log_likelihood_array.append(np.mean(self.log_likelihood.eval(feed_dict={self.X: x_batch})))
                self.KL_term_array.append(np.mean(self.KL_term.eval(feed_dict={self.X: x_batch})))

        plt.figure()
        plt.plot(self.variational_lower_bound_array)
        plt.plot(self.KL_term_array)
        plt.plot(self.log_likelihood_array)
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
            x = np.zeros((1, IMG_PIXELS))
            x[0] = random.choice(self.img_data)['x'].flatten()
            gen_img = self.reconstruction.eval(feed_dict={self.X: x})
            gen_img = (np.reshape(gen_img, (28,28)))
            images.append(gen_img)
        
        self.viz_dataset(num, images=images)

    def viz_generation(self, mu, sigma):# Ported for Keras
        z_sample = np.random.normal(mu, sigma, (1, NOISE_DIM))
        x_decoded = self.generator.predict(z_sample)
        img = x_decoded[0].reshape(IMG_HEIGHT, IMG_HEIGHT)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.show()

    def save(self, in_str):
        self.saver.save(self.sess, in_str)

    def load(self, graph):
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(graph + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        all_vars = tf.get_collection('vars')
        for v in all_vars:
            v_ = self.sess.run(v)
            print(v_)

    

NOISE_DIM = 100
IMG_PIXELS = 28*28
NUM_CHANNELS = 1 #3
IMG_SHAPE = (None, 28, 28)
LEARNING_RATE = 0.001
BATCH_SIZE = 128

'''
TODO:
    Make an epoch run over the entire dataset once
    Support functions that add and save (using pkl?)
'''
if __name__ == "__main__":
    vae = VarAutoEnc()
    #vae.load_dataset('images/pokemon/orig/monochrome/28/')
    vae.load_mnist('images/mnist/mnist_png/training/', compressed=True)
    vae.train(10)#1000 is entire dataset if BATCH_SIZE is ~64
    
    



    
