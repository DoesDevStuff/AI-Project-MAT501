"""
Created on Sat Dec 11   02:38:11 2021

@author: Charlie
"""

from keras.callbacks import TensorBoard
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = tf.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        self._write_logs(stats, step)
