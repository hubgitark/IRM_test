# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:30:12 2022

@author: Ark

IRM implem to keras based on original paper IRM 
Implementation

"""

import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

tf.random.set_seed(10)

#https://stackoverflow.com/questions/65566794/how-to-compute-the-penalty-for-invariant-risk-minimization-in-tensorflow?newreg=083d2f4127fd436db849bf8823f149fb
# this should be mathematical equal to separately calc and add two penalties together
# details on scribble
def compute_penalty(y_true, y_pred):
    dummy_w = tf.constant(1.0)
    y_true1 = y_true[0::2]
    y_pred1 = y_pred[0::2]
    
    y_true2 = y_true[1::2]
    y_pred2 = y_pred[1::2]
    
    mse = tf.keras.losses.MeanSquaredError()
    
    with tf.GradientTape() as tape:
        tape.watch(dummy_w)
        losses1 = mse(y_true1, y_pred1*dummy_w)
    grad1 = tape.gradient(losses1, [dummy_w])[0]
    
    with tf.GradientTape() as tape:
        tape.watch(dummy_w)
        losses2 = mse(y_true2, y_pred2*dummy_w)
    grad2 = tape.gradient(losses2, [dummy_w])[0]
    
    return tf.math.reduce_sum(grad1*grad2)


def loss_IRM(y_true, y_pred):
    y_true1 = y_true[:10000]
    y_pred1 = y_pred[:10000]
    
    y_true2 = y_true[10000:]
    y_pred2 = y_pred[10000:]
    
    mse = tf.keras.losses.MeanSquaredError()
    
    error1 = mse(y_true1, y_pred1)
    penalty1 = compute_penalty(y_true1, y_pred1)
    
    error2 = mse(y_true2, y_pred2)
    penalty2 = compute_penalty(y_true2, y_pred2)
    
    error = error1+error2
    penalty = penalty1+penalty2
    
    return 1e-5 * error + penalty
 
def example_1 (n=10000, d=2, env=1):
    x = tf.random.normal(shape=(n,d))*env
    y = x + tf.random.normal(shape=(n,d))*env
    z = y + tf.random.normal(shape=(n,d))
    return tf.concat([x, z], 1), tf.math.reduce_sum(y,1, keepdims=True)


environments = [example_1(env=0.1),example_1(env=1.0)]


###///------------------------- Model Training -------------------------\\\### 
input_shape = 4
#------model components------\
model = models.Sequential()
model.add(layers.InputLayer(input_shape=input_shape))
model.add(layers.Dense(1,use_bias=False))
model.build(input_shape)
model.summary()
#custom
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=loss_IRM)
model.set_weights([np.array([[1],[1],[1],[1]])])


print('\n---------IRM from paper on Keras---------')
print('tf.random.set_seed(10)')
historyLst = []
for iteration in range(5000):
    for i in range(len(environments)):
        x_e, y_e = environments[i]
        if i == 0:
            train_x = x_e
            train_y = y_e
        else:
            train_x = tf.concat([train_x, x_e], 0)
            train_y = tf.concat([train_y, y_e], 0)
        
    history = model.fit(train_x, train_y, epochs=1, batch_size = len(x_e),verbose=0)  
    historyLst.append(history)
    #model.predict(x_e)
    
    if iteration % 1000 == 0:
        weights = model.get_weights()
        print('weights of x0,x1,z0,z1\n',weights)


#------training visualizing, loss------\
hist_hist_loss_train = []
hist_hist_loss_test = []
for history in historyLst:
    hist_hist_loss_train.extend(history.history['loss'])
        
plt.plot(hist_hist_loss_train, label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0, 0.01])
plt.legend(loc='lower right')
plt.show()
plt.clf()
    


'''
#tester
model.set_weights([np.array([[1],[1],[1],[1]])])
#model.set_weights([np.array([[1],[1],[0],[0]])]) # perfect weight tester

y_true = y_e
y_pred = model.predict(x_e)
loss_IRM(y_true, y_pred)

'''





