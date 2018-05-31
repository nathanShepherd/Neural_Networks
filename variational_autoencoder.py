# Started from Tutorial by Siraj Raval
# How to Generate Images with Tensorflow (LIVE)
# @ https://youtu.be/iz-TZOEKXzA
# Developed by Nathan Shepherd

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

