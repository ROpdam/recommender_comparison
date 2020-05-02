import tensorflow as tf
import numpy as np
from time import time

class TimingCallback(tf.keras.callbacks.Callback):
    """
    Log training time callback to be used in tf.keras.model.fit
    """
  def __init__(self):
    self.logs=[]
  def on_epoch_begin(self,epoch, logs={}):
    self.starttime=time()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(time()-self.starttime)


######################################## NOT NEEDED ##########################################
# class Batch_Generator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, X, y, batch_size=1, shuffle=True):
#         'Initialization'
#         self.X = X
#         self.y = y
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.on_epoch_end()
#
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.y)#int(np.floor(len(self.y)/self.batch_size))
#
#     def __getitem__(self, index):
#         return self.__data_generation(index)
#
#     def on_epoch_end(self):
#         'Shuffles indexes after each epoch'
#         self.indexes = np.arange(len(self.y))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
#
#     def __data_generation(self, index):
#         return np.stack(self.X[index]), np.stack(self.y[index])