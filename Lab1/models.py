import math
from typing import Union, Collection

import tensorflow as tf

TFData = Union[tf.Tensor, tf.Variable, float] # This data type can work as any of these

class GMModel:
    def __init__(self, K):
        self.K = K
        # Wrapping parameters in tf.Variable tells TensorFlow to keep track of gradients
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property # Means that it can be used like a non-function: instance.variables
    def variables(self) -> Collection[TFData]: # A convenient bundle of parameters meant for the optimizer's tweaking
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)

        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

    @tf.function # Takes a regular Python function and compiles it into a high-performance TF graph
    def loss(self, data: TFData):
        raise NotImplementedError

    def p_xz(self, x: TFData, k: int) -> TFData:
        raise NotImplementedError

    def p_x(self, x: TFData) -> TFData:
        raise NotImplementedError
