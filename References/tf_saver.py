#
#
#

import tensorflow as tf 

'''
#%#%#%#%#% Save to file
W = tf.Variable([[1,2,3],[1,2,3]], dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]], dtype=tf.float32, name='bias')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
  sess.run(init)
  save_path = saver.save(sess, 'saved_model/save.ckpt')
  print('Saved graph to', save_path)
'''

#%#%#%#%#% Restore Variables
W = tf.Variable(tf.zeros([2,3]), dtype=tf.float32, name='weights')
b = tf.Variable(tf.zeros([1,3]), dtype=tf.float32, name='bias')

#pass init step

saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "saved_model/save.ckpt")
  print('weights:', sess.run(W))
  print('bias:', sess.run(b))

