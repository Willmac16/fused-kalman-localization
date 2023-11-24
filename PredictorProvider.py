
from abc import ABC, abstractmethod
import numpy as np

import SigmaGenerator

class PredictorProvider(ABC):
  @abstractmethod
  def process_noise(self):
    pass

  @abstractmethod
  def point_prop(self, state, dt):
    pass

  @abstractmethod
  def covariance_predictor(self, state, covariance, dt):
    pass


class LinearPredictorProvider(PredictorProvider):
  @abstractmethod
  def linear_prop_mat(self, state, dt):
    # Implement linear propagation logic here
    pass

  # Implement non-linear point_prop if desired
  def point_prop(self, state, dt):
    return self.linear_prop_mat(state, dt) @ state

  def covariance_predictor(self, state, covariance, dt):
    return self.point_prop(state, covariance, dt), self.linear_prop_mat(state, dt) @ covariance @ self.linear_prop_mat(state, dt).T + self.process_noise()

class UnscentedPredictorProvider(PredictorProvider):
  def __init__(self, num_states, alpha=1e-3, beta=2, kappa=1):
    self.num_states = num_states

    self.sigma_generator = SigmaGenerator.SigmaGenerator(num_states, alpha, beta, kappa)

  @abstractmethod
  def point_prop(self, state, dt):
    # Implement unscented point propagation
    pass

  def covariance_predictor(self, state, covariance, dt):
    sigma_points, mean_weights, cov_weights = self.sigma_generator.sigma_points(state, covariance)

    x_prime = self.point_prop(sigma_points, dt)

    mean = mean_weights.T @ x_prime

    dev = x_prime - mean
    covariance = cov_weights.T @ dev @ dev.T + self.process_noise()

    return mean, covariance
