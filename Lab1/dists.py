from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np

NumT = Union[np.ndarray, float] # Custom type that can be either np.ndarray or float

@dataclass # Auto-generates an __init__ method that accepts the fields defined inside the class (just pi here)
class MixtureDist(ABC): # Abstract Base Class
    pi: NumT
    K: int = field(init=False) # Will be calculated automatically because it depends on other inputs, not passed by the user

    def __post_init__(self): # Counts how many mixing weights were provided and sets K right after init
        self.K = len(self.pi)

    @abstractmethod
    def sample(self, L: int) -> NumT: # How to draw L random samples from the mixture distribution
        raise NotImplementedError

    @abstractmethod # How to compute the probability or density of a data point x given a specific component k
    def p_xz(self, x: NumT, k: int) -> NumT:
        raise NotImplementedError

    def p_xz_w(self, x: NumT, k: int) -> NumT: # Weighted version of the same thing
        pi_k = self.pi[k] if hasattr(self.pi, "__getitem__") else self.pi
        return pi_k * self.p_xz(x, k)

    def p_x(self, x: NumT) -> NumT: # Returns the weighted sum of heights of all the curves at point x
        return np.sum([self.pi[k] * self.p_xz(x, k) for k in range(self.K)], axis=0)

@dataclass
class GMDist(MixtureDist):
    mu: NumT
    sigma2: NumT

    @classmethod
    def random(cls, K: int) -> GMDist:
        mu = np.random.normal(size=K).astype(np.float32) # Pick a random mean through a N(0, 1)
        sigma2 = np.exp(np.random.normal(size=K) - 1).astype(np.float32) # Pick a random (strictly positive) variance
        pi = np.random.dirichlet(5 * np.ones(K)).astype(np.float32) # The 5 parameter smooths it out nicely so that one class doesn't hog up all the probability

        return cls(pi, mu, sigma2)

    def sample(self, L: int) -> NumT:
        choices = np.random.choice(np.arange(self.K), size=L, p=self.pi) # Choose z - L times
        sample = np.random.normal(self.mu[choices], np.sqrt(self.sigma2[choices])).astype(np.float32) # Choose an x per z distribution

        return sample

    @staticmethod
    def normal_pdf(x: NumT, mu: NumT, sigma2: NumT) -> NumT: # Returns the height of the bell curve at x
        return np.exp(-0.5 * (x - mu) ** 2 / sigma2) / np.sqrt(2 * np.pi * sigma2)

    def p_xz(self, x: NumT, k: int) -> NumT: # Uses normal_pdf to return the height of the bell curve at x for a specific component
        return self.normal_pdf(x, self.mu[k], self.sigma2[k])

@dataclass
class UMDist(MixtureDist):
    a: NumT
    b: NumT

    @classmethod
    def random(cls, K: int) -> UMDist:
        raise NotImplementedError

    def sample(self, L: int) -> NumT:
        raise NotImplementedError

    @staticmethod
    def uniform_pdf(x: NumT, a: NumT, b: NumT) -> NumT:
        raise NotImplementedError

    def p_xz(self, x: NumT, k: int) -> NumT:
        raise NotImplementedError
