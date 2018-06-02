# Started from Tutorial by Siraj Raval
# How to Generate Images with Tensorflow (LIVE)
# @ https://youtu.be/iz-TZOEKXzA
# Developed by Nathan Shepherd

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)




class VAE:
    def __init__(self):
        pass

    def load_dataset(self, data_dir):
        print('Loading dataset'); img_data = []
        for i, filename in enumerate(os.listdir(data_dir)):
          if (i % (len(os.listdir(data_dir)) / 10)) == 0:
              print(i*100/(len(os.listdir(data_dir))))

          img = Image.open(data_dir + filename)
          img_data.append(img)

        self.img_data = img_data




if __name__ == "__main__":
    vae = VAE()
    vae.load_dataset('images/pokemon/28/')
    
    
