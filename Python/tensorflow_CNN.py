## References
# Convolutional Neural Networks Tutorial in TensorFlow
# @ http://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
##
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

class ConvNet(NeuralNetwork):
    def __init__(self): 
        self.img_pixels = IMG_PIXELS

        # Inputs
        self.X = tf.placeholder(tf.float32, shape=([None, IMG_PIXELS]))
        self.x_shaped = tf.reshape(self.X, [-1, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])

        # Convolutional Layers
        layer1 = self.Conv_Layer(self.x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
        layer2 = self.Conv_Layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')
        self.flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])

        # Fully Connected Layers for softmax classification
        self.y_, self.dense_layer2 = self.Fully_Connected([7 * 7 * 64, 1000, 10])

        # Cost function
        self.cross_entropy = tf.reduce_mean( \
                tf.nn.softmax_cross_entropy_with_logits( \
                        logits=self.dense_layer2, labels=self.y))
        # Optimization
        self.optimiser = tf.train.AdamOptimizer( \
                    learning_rate=LEARNING_RATE).minimize(self.cross_entropy)

        # Accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def Fully_Connected(self, hidden):
        wd1 = tf.Variable(tf.truncated_normal([hidden[0], hidden[1]], stddev=0.03), name='wd1')
        bd1 = tf.Variable(tf.truncated_normal([hidden[1]], stddev=0.01), name='bd1')
        dense_layer1 = tf.matmul(self.flattened, wd1) + bd1
        dense_layer1 = tf.nn.relu(dense_layer1)

        wd2 = tf.Variable(tf.truncated_normal([hidden[1], hidden[-1]], stddev=0.03), name='wd2')
        bd2 = tf.Variable(tf.truncated_normal([hidden[-1]], stddev=0.01), name='bd2')
        dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
        y_ = tf.nn.softmax(dense_layer2)
        return y_, dense_layer2    

    def Conv_Layer(self, input_data, num_input_channels,
                         num_filters, filter_shape, pool_shape, name):
        
        conv_filt_shape = [filter_shape[0], filter_shape[1],
                           num_input_channels, num_filters]
        weights = tf.Variable(tf.truncated_normal(conv_filt_shape,
                                                  stddev=0.03),name=name+'_W')
        bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')

        out_layer = tf.nn.conv2d(input_data, weights,
                                 [1, 1, 1, 1], padding='SAME')

        out_layer += bias; out_layer = tf.nn.relu(out_layer)

        # max pooling
        ksize = [1, pool_shape[0], pool_shape[1], 1]
        strides = [1, 2, 2, 1]
        out_layer = tf.nn.max_pool(out_layer, ksize=ksize,
                                   strides=strides, padding='SAME')
        return out_layer

    def train(self, epochs):
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            total_batch = int(len(self.img_data) / BATCH_SIZE)

            for epoch in range(epochs):
                avg_cost = 0

                for i in range(total_batch):
                    batch = random.sample(self.img_data, BATCH_SIZE)
                    train_x = [np.array(img['x']).flatten() for img in batch]
                    train_y = [img['y'] for img in batch]

                    _, c = sess.run([self.optimiser, self.cross_entropy], 
                            feed_dict={self.X: train_x, self.y: train_y})

                    avg_cost += c / total_batch

                    # Testing
                    batch = random.sample(self.img_data, BATCH_SIZE)
                    test_x = [np.array(img['x']).flatten() for img in batch]
                    test_y = [img['y'] for img in batch]
                    test_acc = sess.run(self.accuracy, feed_dict={self.X: test_x,
                                                                  self.y: test_y})
                    print("Epoch:", (epoch + 1),
                          "%Complete:", int(i*100/total_batch),
                          "cost =", "{:.3f}".format(avg_cost),
                          "test accuracy: {:.3f}".format(test_acc))
    

NOISE_DIM = 100
IMG_PIXELS = 28*28
NUM_CHANNELS = 1 #3
IMG_SHAPE = (None, 28, 28)
LEARNING_RATE = 0.0001
BATCH_SIZE = 64

'''
TODO:
    Integrate CNN with VarAutoEncoder as the encoder network
    Define functions for prediction and visualizations
    Save/load trained models
'''
if __name__ == "__main__":
    cnn = ConvNet()
    #vae.load_dataset('images/pokemon/orig/monochrome/28/')
    cnn.load_mnist('images/mnist/mnist_png/training/', compressed=True)
    cnn.train(1)
    



    
