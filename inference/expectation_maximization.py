from time import time
from random import Random
import numpy as np
from model import *
import matplotlib.pyplot as pl
import pickle


def get_random_matrix(random_generator, kind, m, n, args=None):
    if args is None:
        args = []
    mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            mat[i, j] = getattr(random_generator, kind)(*args)
    return mat



class ExpectationMaximization:

    def __init__(self, data, model, keys_params, lower_bounds_noise, upper_bounds_noise, lower_bounds_params, upper_bounds_params,
                             lower_bounds_states, upper_bounds_states, adaptation_params, n_particles=None, smoothing_lag=None,
                             seed=time(), output_counter=1000):

        self.data = data
        self.model = model
        self.keys_params = keys_params

        # bounds
        self.lower_bounds = np.array(lower_bounds_noise + lower_bounds_params + lower_bounds_states)
        self.upper_bounds = np.array(upper_bounds_noise + upper_bounds_params + upper_bounds_states)

        # adaptation parameters
        self.a, self.b, self.c = adaptation_params

        # number of particles
        if n_particles is None:
            n_particles = (1 + len(lower_bounds_params) + len(lower_bounds_states)) * 100
        self.n_particles = n_particles

        # smoothing lag
        self.smoothing_lag = smoothing_lag

        # random number generator
        self.random_generator = Random()
        self.random_generator.seed(seed)

        # output counter
        self.output_counter = output_counter

        # to get indices for Z easier
        n_noise = 3
        n_params = len(keys_params)
        n_gates = len(self.model.ionchannels)
        self.to_idx = {'scale_factor': 0, 'std_noise_intrinsic': 1,'std_noise_observed': 2,
                       'params': range(n_noise, n_noise+n_params), 'v': n_noise+n_params,
                       'a_gates': range(n_noise+n_params+1, n_noise+n_params+1+n_gates),
                       'b_gates': range(n_noise+n_params+1+n_gates, n_noise+n_params+1+2*n_gates)}

    def run(self):
        # figure
        pl.ion()
        f, self.ax = pl.subplots()

        # sample initial states
        Z0 = np.tile(np.array([self.lower_bounds]).T, (1, self.n_particles)) \
                     + np.tile(np.array([self.upper_bounds - self.lower_bounds]).T, (1, self.n_particles)) \
                     * get_random_matrix(self.random_generator, 'random', len(self.lower_bounds), self.n_particles) #TODO: all at upper bounds

        # initial covariance matrix
        idx_params = self.to_idx['params'] + [self.to_idx['std_noise_intrinsic']] + [self.to_idx['std_noise_observed']]
        param_cov0 = np.diag(self.upper_bounds[idx_params] - self.lower_bounds[idx_params]) * 1e-2

        if self.smoothing_lag is None or self.smoothing_lag <= 1:
            Zavg = self.sequential_monte_carlo_filter(Z0, param_cov0)
        else:
            Zavg = self.sequential_monte_carlo_filter_with_smoothing_lag(Z0, param_cov0)
        return Zavg

    def sequential_monte_carlo_filter(self, Z0, param_cov0):

        n_timesteps = np.shape(self.data)[1]
        len_z = np.shape(Z0)[0]

        # allocate memory
        Zavg = np.zeros((len_z, n_timesteps))

        # initialize
        Z = Z0
        w = self.update_weights(timestep=0, Z=Z, w_old=np.ones(self.n_particles) / self.n_particles)
        param_cov = param_cov0
        Zavg[:, 0] = np.dot(Z, w)

        # main loop
        for timestep in range(1, n_timesteps):
            # re-sample if necessary
            effective_n_particles = 1 / sum(w * w)
            if effective_n_particles < self.n_particles / 2:
                Z, w = self.resample_weights(Z, w)

            # propagate particles
            Z, param_cov = self.update_states(timestep, Z, w, param_cov)
            w = self.update_weights(timestep, Z, w)

            # normalize weights
            w = w / sum(w)
            if any(np.isnan(w)):  # check for NaN and act accordingly
                w = np.ones(self.n_particles) / self.n_particles

            # compute statistics
            Zavg[:, timestep] = np.dot(Z, w)

            # log progress
            if timestep % self.output_counter == 0:
                print 'Step '+str(timestep)+' of '+str(n_timesteps)
                x = np.squeeze(np.asarray(self.data[0, :timestep]))
                y = np.squeeze(np.asarray(self.data[2, :timestep]))
                self.ax.plot(x, y, 'k', label='v observed')
                self.ax.plot(x, Zavg[self.to_idx['v'], :timestep], 'r', label='v inferred')
                pl.draw()
                pl.pause(0.0001)
        pl.show(block=True)
        return Zavg


    def sequential_monte_carlo_filter_with_smoothing_lag(self, Z0, param_cov0):

        n_timesteps = np.shape(self.data)[1]
        len_z = np.shape(Z0)[0]

        # allocate memory
        Zavg = np.zeros((len_z, n_timesteps))
        Z = np.zeros((len_z, self.n_particles, self.smoothing_lag))
        w = np.zeros((self.n_particles, self.smoothing_lag))

        # initialize
        Z[:, :, 0] = Z0
        w[:, 0] = self.update_weights(timestep=0, Z=Z[:, :, 0], w_old=np.ones((self.n_particles)) / self.n_particles)
        param_cov = param_cov0

        # filter first L data points
        for timestep in range(1, self.smoothing_lag):
            # read auxiliary variables
            Z_ = Z[:, :, timestep - 1]
            w_ = w[:, timestep - 1]

            # re-sample if necessary
            effective_n_particles = 1 / sum(w_ * w_)
            if effective_n_particles < self.n_particles / 2:
                Z_, w_ = self.resample_weights(Z_, w_)

            # propagate particles
            [Z_, param_cov] = self.update_states(timestep, Z_, w_, param_cov)
            w_ = self.update_weights(timestep, Z_, w_)

            # normalize weights
            w_ = w_ / sum(w_)
            if any(np.isnan(w_)):  # check for NaN and act accordingly
                w_ = np.ones(self.n_particles) / self.n_particles

            # update Z and w
            Z[:, :, timestep] = Z_
            w[:, timestep] = w_

        # main loop
        for timestep in range(self.smoothing_lag, n_timesteps):
            # compute statistics
            Zavg[:, timestep - self.smoothing_lag] = np.dot(Z[:, :, 0], w[:, 0])

            # re-sample, if necessary
            effective_n_particles = 1 / sum(w[:, self.smoothing_lag-1] * w[:, self.smoothing_lag-1])
            if effective_n_particles < self.n_particles / 2:
                Z, w = self.resample_weights_with_smoothing_lag(Z, w)  # resample whole trajectories

            # read auxiliary variables
            Z_ = Z[:, :, self.smoothing_lag-1]
            w_ = w[:, self.smoothing_lag-1]

            # propagate particles
            Z_, param_cov = self.update_states(timestep, Z_, w_, param_cov)
            w_ = self.update_weights(timestep, Z_, w_)

            # normalise weights
            w_ = w_ / sum(w_)
            if any(np.isnan(w_)):  # check for NaN and act accordingly
                w_ = np.ones(self.n_particles) / self.n_particles

            # update storage
            Z[:, :, :self.smoothing_lag-1] = Z[:, :, 1:self.smoothing_lag]
            Z[:, :, self.smoothing_lag-1] = Z_
            w[:, :self.smoothing_lag-1] = w[:, 1:self.smoothing_lag]
            w[:, self.smoothing_lag-1] = w_

            # log progress
            if timestep % self.output_counter == 0:
                print 'Step ' + str(timestep) + ' of ' + str(n_timesteps)
                x = np.squeeze(np.asarray(self.data[0, :timestep - self.smoothing_lag]))
                y = np.squeeze(np.asarray(self.data[2, :timestep - self.smoothing_lag]))
                self.ax.plot(x, y, 'k', label='v observed')
                self.ax.plot(x, Zavg[self.to_idx['v'], :timestep - self.smoothing_lag], 'r', label='v inferred')
                pl.draw()
                pl.pause(0.0001)
        pl.show(block=True)
        return Zavg

    def update_weights(self, timestep, Z, w_old):
        v_observed = self.data[2, timestep]
        std_noise_observed = Z[self.to_idx['std_noise_observed'], :]
        v_inferred = Z[self.to_idx['v'], :]

        # update weights
        w = w_old * (np.exp(-0.5 * ((v_inferred - v_observed) / std_noise_observed) ** 2)
                     / (np.sqrt(2.0 * np.pi) * std_noise_observed))
        return w

    def update_states(self, timestep, Z, w, old_param_cov):

        # update scale factor
        Z[self.to_idx['scale_factor'], :] = map(lambda x: x * np.exp(self.c * self.random_generator.gauss(0, 1)),
                                                Z[0, :])
        Z = self.reduce_to_bounds(Z, self.to_idx['scale_factor'])

        # update parameters and states
        Z, old_param_cov = self.update_params(Z, w, old_param_cov)

        for i in range(self.n_particles):
            # update parameter in the model
            values = Z[self.to_idx['params'], i]
            for keys, value in zip(self.keys_params, values):
                self.model.update_attr(keys, value)
            v = Z[self.to_idx['v'], i]
            a_gates = Z[self.to_idx['a_gates'], i]
            b_gates = Z[self.to_idx['b_gates'], i]
            i_inj = self.data[1, timestep]
            dt = self.data[0, timestep] - self.data[0, timestep-1]
            std_noise_intrinsic = Z[self.to_idx['std_noise_intrinsic'], i]
            v, a_gates, b_gates = update_voltage_and_gates(self.model, v, a_gates, b_gates, i_inj, dt,
                                                           std_noise_intrinsic, self.random_generator)
            self.reduce_to_bounds(Z, self.to_idx['a_gates']+self.to_idx['b_gates'])
            Z[self.to_idx['v'], i] = v
            Z[self.to_idx['a_gates'], i] = a_gates
            Z[self.to_idx['b_gates'], i] = b_gates

        return Z, old_param_cov

    def update_params(self, Z, w, old_param_cov):
        idx_params = self.to_idx['params']+[self.to_idx['std_noise_intrinsic']]+[self.to_idx['std_noise_observed']]
        params = Z[idx_params, :]
        scale_factor = Z[self.to_idx['scale_factor'], :]

        # compute mean and covariance
        params_mean = np.dot(params, w)
        residuals = (params.T - params_mean).T
        params_cov = np.dot(residuals * w, residuals.T)

        # compute new mean and covariance
        new_param_mean = ((1 - self.a) * params.T + self.a * params_mean).T
        new_param_cov = (1 - self.b) * old_param_cov + self.b * params_cov

        # update parameters
        randns = get_random_matrix(self.random_generator, 'gauss', len(params_mean), self.n_particles, [0, 1])
        params = new_param_mean + np.dot(np.linalg.cholesky(new_param_cov), randns * scale_factor)
        Z[idx_params, :] = params
        Z = self.reduce_to_bounds(Z, idx_params)
        return Z, new_param_cov

    def resample_weights(self, Z, w):
        idxs = self.get_weight_indices(w)
        Z = Z[:, idxs]  # update Z
        w = np.ones(len(w)) / len(w)   # TODO: why now start again from uniform?!
        return Z, w

    def resample_weights_with_smoothing_lag(self, Z, w):
        idxs = self.get_weight_indices(w[:, -1])
        Z = Z[:, idxs, :]
        w[:, :] = 1 / np.shape(w)[0]
        return Z, w

    def get_weight_indices(self, w):
        cumsum_w = np.cumsum(w)
        idxs = np.zeros(len(w), dtype=int)
        for i in range(len(w)):
            idx = 0
            r = self.random_generator.random()
            while cumsum_w[idx] < r:
                idx += 1
            idxs[i] = idx
        return idxs

    def reduce_to_bounds(self, Z, idx):
        Z_idx = Z[idx, :]
        lower = (Z_idx.T < self.lower_bounds[idx]).T
        upper = (Z_idx.T > self.upper_bounds[idx]).T
        if lower.any():
            Z_idx[lower] = (lower.T * self.lower_bounds[idx]).T[lower]
        if upper.any():
            Z_idx[upper] = (upper.T * self.upper_bounds[idx]).T[upper]
        Z[idx, :] = Z_idx
        return Z

    def save(self, save_dir):
        with open(save_dir, 'w') as f:
            pickle.dump(self, f)
