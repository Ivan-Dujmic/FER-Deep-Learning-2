import math
from typing import Union, Collection

import tensorflow as tf
import tensorflow.math as tfm

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
        log_pi_norm = tf.nn.log_softmax(self.logpi)

        log_probs = []

        for k in range(self.K):
            var_k = tfm.exp(self.logvar[k])
            L_xz_k = -0.5 * (tfm.log(2 * math.pi) + self.logvar[k] + tfm.square(data - self.mean[k]) / var_k)

            if len(L_xz_k.shape) > 1: # In case data has multiple features
                L_xz_k = tf.reduce_sum(L_xz_k, axis=-1)

            log_probs.append(log_pi_norm[k] + L_xz_k)

        log_probs = tf.stack(log_probs, axis=1)
        log_likelihood = tf.reduce_logsumexp(log_probs, axis=1)
        return -log_likelihood

    @staticmethod
    def normal_pdf(x: TFData, mu: TFData, sigma2: TFData) -> TFData:
        return tfm.exp(-0.5 * (x - mu) ** 2 / sigma2) / tfm.sqrt(2 * math.pi * sigma2)

    def p_xz(self, x: TFData, k: int) -> TFData:
        return self.normal_pdf(x, self.mean[k], tfm.exp(self.logvar[k]))

    def p_xz_w(self, x: TFData, k: int) -> TFData:
        p_xz_k = self.p_xz(x, k)
        pi_k = tf.nn.softmax(self.logpi)[k]
        return pi_k * p_xz_k

    def p_x(self, x: TFData) -> TFData:
        pi = tf.nn.softmax(self.logpi)
        components = [pi[k] * self.p_xz(x, k) for k in range(self.K)]
        return tf.add_n(components)
