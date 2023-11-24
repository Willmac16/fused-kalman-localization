from abc import ABC, abstractmethod
import numpy as np

import SigmaGenerator

# Each corrector provider handles a single, atomic sensor update
class CorrectorProvider(ABC):
  @abstractmethod
  def measurement_noise(self):
    pass

  @abstractmethod
  def obs_pred(self, state):
    pass

  @abstractmethod
  def correct(self, state, covariance, measurement):
    pass

class LinearCorrectorProvider(CorrectorProvider):
  @abstractmethod
  def linear_obs_mat(self):
    pass

  def obs_pred(self, state):
    return self.linear_obs_mat() @ state

  def correct(self, state, covariance, measurement):
    obs_mat = self.linear_obs_mat()

    kalman_gain = covariance @ obs_mat.T @ np.linalg.inv(obs_mat @ covariance @ obs_mat.T + self.measurement_noise())

    posterior_x = state + kalman_gain @ (measurement - self.obs_pred(state))
    posterior_covariance = (np.eye(len(state)) - kalman_gain @ obs_mat) @ covariance

    return posterior_x, posterior_covariance


class UnscentedPredictorProvider(CorrectorProvider):
  def __init__(self, num_states, alpha=1e-3, beta=2, kappa=1):
    self.num_states = num_states

    self.sigma_generator = SigmaGenerator.SigmaGenerator(num_states, alpha, beta, kappa)

  def correct(self, state, covariance, measurement):
    sigma_points, mean_weights, cov_weights = self.sigma_generator.sigma_points(state, covariance)

    pred_measurements = self.obs_pred(sigma_points)

    mean_measure = mean_weights.T @ pred_measurements
    dev = pred_measurements - mean_measure
    cov_measure = cov_weights.T @ dev @ dev.T + self.measurement_noise()

    cross_cov = cov_weights.T @ (sigma_points - state) @ dev.T

    kalman_gain = cross_cov @ np.linalg.inv(cov_measure)

    posterior_x = state + kalman_gain @ (measurement - mean_measure)
    posterior_cov = covariance - kalman_gain @ cov_measure @ kalman_gain.T

    return posterior_x, posterior_cov
