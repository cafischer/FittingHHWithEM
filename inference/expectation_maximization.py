from time import time
from random import Random
import numpy as np
from model import *


def expectation_maximization(data, model, keys_params, lower_bounds_params, upper_bounds_params, lower_bounds_states, upper_bounds_states,
                             output_counter, n_particles=None, smoothing_lag=None, seed=time()):

    # random number generator
    random_generator = Random()
    random_generator.seed(seed)

    # adaptation parameters
    a = 0.01
    b = 0.01
    c = 0.01

    lower_bounds = lower_bounds_params + lower_bounds_states
    upper_bounds = upper_bounds_params + upper_bounds_states

    # number of particles
    if n_particles is None:
        n_particles = (1 + len(lower_bounds_params) + len(lower_bounds_states)) * 100

    # sample initial states
    Z0 = [map(lambda x: x * random_generator.random(), lower_bounds + (upper_bounds - lower_bounds)) for i in range(n_particles)]

    # initial Q
    param_cov0 = np.diag(upper_bounds - lower_bounds) * 1e-2  # TODO: why 1e-2???

    # execute filter
    if smoothing_lag is None or smoothing_lag <= 1:
        Zavg, Q = sequential_monte_carlo_filter(update_states_, update_weights_, data, model, keys_params, Z0, param_cov0, output_counter,
                                                random_generator)
    else:
        Zavg, Q = sequential_monte_carlo_filter_with_smoothing_lag(update_states_, update_weights_, smoothing_lag, data, model, keys_params,
                                                                   Z0, param_cov0, output_counter, random_generator)

    return Zavg, Q


def sequential_monte_carlo_filter(update_states, update_weights, data, model, keys_params, Z0, Q0, output_counter, random_generator):

    n_timesteps = np.shape(data)[1]
    n_params, n_particles = np.shape(Z0)

    # allocate memory
    Zavg = np.zeros(n_params, n_timesteps)

    # initialize
    Z = Z0
    w = update_weights(timestep=0, data=data, Z=Z, w_old=np.ones(n_particles) / n_particles)
    param_cov = Q0

    Zavg[:, 0] = Z * w

    # main loop
    for timestep in range(1, n_timesteps):
        # re-sample if necessary
        effective_n_particles = 1 / sum(w * w)
        if effective_n_particles < n_particles / 2:
            Z, w = resample_weights(Z, w, random_generator)

        # propagate particles
        Z, param_cov = update_states(timestep, data, Z, w, param_cov, random_generator)
        w = update_weights(timestep, data, Z, w)

        # normalize weights
        w = w / sum(w)
        if any(np.isnan(w)):  # check for NaN and act accordingly
            w = 1 / n_particles

        # compute statistics
        Zavg[:, timestep] = Z * w

        # log progress
        if timestep % output_counter == 0:
            print 'Step '+str(timestep)+' of '+str(n_timesteps)
    return Zavg, param_cov


def sequential_monte_carlo_filter_with_smoothing_lag():
    pass


def update_states_(timestep, data, model, keys_params, Z, w, old_param_cov, lower_bounds, upper_bounds, a, b, c, idxs, random_generator):
    # read data
    t1 = data(1, timestep - 1)
    t2 = data(1, timestep)
    Iinj = data(2, timestep - 1)

    # update scale factor
    Z[1,:] = map(lambda x: x * np.exp(c * random_generator.gauss(0, 1)), Z[1, :])
    Z[1, Z[1, :] < lower_bounds[1]] = lower_bounds[1]
    Z[1, Z[1, :] > upper_bounds[1]] = upper_bounds[1]

    # update parameters and states
    if idxs is not None:
        Z[idxs, :], old_param_cov = update_params(Z[idxs, :], Z[1, :], w, old_param_cov, lower_bounds(idxs), upper_bounds(idxs),
                                                  a, b, random_generator)

    for i in range(n_particles):
        # update parameters in the model
        for keys, value in zip(keys_params, values):
            model.update_attr(keys, value)
        v, a_gates, b_gates = update_voltage_and_gates(model, v, a_gates, b_gates, i_inj, dt, std_noise_intrinsic, random_generator)

    return Z, old_param_cov


def update_weights_(time_step, data, Z, w_old):
    v_observed = data(3, time_step)
    std_noise_observed = Z[3, :]
    v_inferred = Z[36, :]

    # update weights
    w = w_old * (np.exp(-0.5 * ((v_inferred - v_observed) / std_noise_observed) ** 2)
                 / (np.sqrt(2.0 * np.pi) * std_noise_observed))
    return w


def resample_weights(Z, w, random_generator):
    idxs = get_weight_indices(w, random_generator)
    Z = Z[:, idxs]  # update Z
    w = 1 / len(w) * np.ones(len(w))  # TODO: why now start again from uniform?!
    return Z, w


def get_weight_indices(w, random_generator):
    """
    Sample len(w) new weights from weights according to the cumulative sum of the weights.
    :param w: weight vector
    :param random_generator: Random number generator.
    :return: Indices of sampled weights.
    """
    cumsum_w = np.cumsum(w)
    idxs = np.zeros(len(w))
    for i in range(len(w)):
        idx = 0
        r = random_generator.random()
        while cumsum_w[idx] < r:
            idx += 1
        idxs[i] = idx
    return idxs


def update_params(params, scale_factor, w, old_param_cov, lower_bounds_params, upper_bounds_params,
                  a, b, random_generator):
    # compute mean and covariance
    params_mean = params * w
    residuals = params - params_mean
    params_cov = np.dot(residuals * w, residuals)

    # compute new mean and covariance
    new_param_mean = (1 - a) * params + a * params_mean
    new_param_cov = (1 - b) * old_param_cov + b * params_cov

    # update parameters
    randns = np.reshape([random_generator.gauss(0, 1) for i in range(np.size(scale_factor))], np.shape(scale_factor))
    params = new_param_mean + np.linalg.cholesky(new_param_cov) * randns * scale_factor
    params[params < lower_bounds_params] = lower_bounds_params
    params[params > upper_bounds_params] = upper_bounds_params
    return params, new_param_cov
