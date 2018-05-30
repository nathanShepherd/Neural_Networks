#%#%#%#%#%#% Neural Network #%#%#%#%#%#%
import tensorflow as tf
from math import floor

class NeuralComputer:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        print('init: len(x) = {}, len(y) = {}'.format(len(self.x), len(self.y)))
        
        self.in_dim = len(x[0])
        self.out_dim = len(y[0])
        
    def Perceptron(self, tensor):
        #with tf.name_scope('softmax_linear'):
            
        V0 = tf.Variable(tf.truncated_normal([self.in_dim, 1000]))
        b0 = tf.Variable(tf.truncated_normal([1000]))
        l0 = tf.sigmoid(tf.matmul(tensor, V0) + b0)

        V1 = tf.Variable(tf.truncated_normal([1000, 750]))
        b1 = tf.Variable(tf.truncated_normal([750]))
        l1 = tf.sigmoid(tf.matmul(l0, V1) + b1)

        V2 = tf.Variable(tf.truncated_normal([750, 600]))
        b2 = tf.Variable(tf.truncated_normal([600]))
        l2 = tf.sigmoid(tf.matmul(l1, V2) + b2)

        V3 = tf.Variable(tf.truncated_normal([600, 500]))
        b3 = tf.Variable(tf.truncated_normal([500]))
        l3 = tf.sigmoid(tf.matmul(l2, V3) + b3)

        V4 = tf.Variable(tf.truncated_normal([500, 300]))
        b4 = tf.Variable(tf.truncated_normal([300]))
        l4 = tf.sigmoid(tf.matmul(l3, V4) + b4)
        
        V5 = tf.Variable(tf.truncated_normal([300, 100]))
        b5 = tf.Variable(tf.truncated_normal([100]))
        l5 = tf.sigmoid(tf.matmul(l4, V5) + b5)

        V6 = tf.Variable(tf.truncated_normal([100, 25]))
        b6 = tf.Variable(tf.truncated_normal([25]))
        l6 = tf.sigmoid(tf.matmul(l5, V6) + b6)
        
        weights = tf.Variable( tf.zeros([25, self.out_dim]),name='weights')
        biases = tf.Variable(tf.zeros([self.out_dim]),name='biases')

        logits = tf.nn.softmax(tf.matmul(l6, weights) + biases)
        
        return logits, weights, biases

    def init_placeholders(self, n_classes, batch_size):
        #init Tensors: fed into the model during training
        x = tf.placeholder(tf.float32, shape=(None, self.in_dim))
        y_ = tf.placeholder(tf.float32, shape=(batch_size, n_classes))

        #Neural Network Model
        y, W, b = self.Perceptron(x)

        return y, W, b, x, y_

    def train(self, test_x, in_str, batch_size=1000, training_epochs=10,learning_rate=.5,display_step=1):
        print('train: len(x) = {}, len(y) = {}'.format(len(self.x), len(self.y)))
        print('len(test_x):',len(test_x))
        #batch_size = len(test_x)
        test_size = batch_size* floor(len(self.x)/batch_size)

        #to verify accuracy on novel data
        acc_x = self.x[test_size - batch_size*2:]
        acc_y = self.y[test_size - batch_size*2:]
        print("acc_x:",len(acc_x), ' acc_y:',len(acc_y))
        
        print("len_train",int(test_size - batch_size*2))
        self.x = self.x[:test_size - batch_size*2]
        self.y = self.y[:test_size - batch_size*2]
        
        # Train W, b such that they are good predictors of y
        self.out_y, W, b, self.in_x, y_ = self.init_placeholders(self.out_dim, batch_size)

        # Cost function: Mean squared error
        cost = tf.reduce_sum(tf.pow(y_ - self.out_y, 2))/(batch_size)

        # Gradient descent: minimize cost via Adam Delta Optimizer (SGD)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate,rho=.99,epsilon=3e-08).minimize(cost)

        # Initialize variables and tensorflow session
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init)

        start_time = time.time()
        print_time = True
        for i in range(training_epochs):
            j=0
            while j < len(self.x):
                start = j
                end = j + batch_size
                
                self.sess.run([optimizer, cost], feed_dict={self.in_x: self.x[start:end],
                                                            y_: self.y[start:end]})
                j += batch_size
            # Display logs for epoch in display_step
            if (i) % display_step == 0:
                if print_time:
                    print_time = False
                    elapsed_time = time.time() - start_time
                    print('Predicted duration of this session:',(elapsed_time*training_epochs//60) + 1,'minute(s)')
                cc = self.sess.run(cost, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})
                print("Training step: {} || cost= {}".format(i,cc))
                        
        print("\nOptimization Finished!\n")
        training_cost = self.sess.run(cost, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})
        print("Training cost=",training_cost,"\nW=", self.sess.run(W)[:1],"\nb=",self.sess.run(b),'\n')
        correct_prediction = tf.equal(tf.argmax(self.out_y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy for predictions of {}'.format(in_str),
                self.sess.run(accuracy, feed_dict={self.in_x: acc_x[:batch_size], y_:acc_y[:batch_size]})*100,'%')
        
        #str(self.sess.run(accuracy, feed_dict={self.in_x: self.x[:batch_size], y_:self.y[:batch_size]})*100//1) + ' %'

    def save(self, in_str):
        self.saver.save(self.sess, in_str)

    def load(self, graph):
        #out_y, W, b, in_x, y_ = self.init_placeholders(self.out_dim, batch_size)
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(graph + '.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./'))
        
    def predict(self, test_x):
        predictions = []
        for matrix in test_x:
            predictions.append(self.sess.run(self.out_y, feed_dict={self.in_x:matrix}))

        self.sess.close()
        return predictions

    def max_of_predictions(self, predictions):
        out_arr = []
        for pred in predictions:
            #print('\n========')
            _max = [0, 0]# [index, value]
            for matrix in pred:
                for i, vect in enumerate(matrix):
                    if _max[1] <  vect:
                        _max[1] = vect
                        _max[0] = i
                    #print('{}::{}'.format(i,vect))
                #print(':MAX:', _max[0], _max[1])
            out_arr.append(_max[0])

        #indecies of max values in one-hot arrays
        return out_arr

#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%v#%#
