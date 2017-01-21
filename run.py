import pandas as pd
import matplotlib.pyplot as pl
from model import *
from inference.expectation_maximization import *


def get_lowerbound_upperbound_keys(variables):
    lower_bound = np.zeros(len(variables))
    upper_bound = np.zeros(len(variables))
    variable_keys = list()
    for i, var in enumerate(variables):
        lower_bound[i] = var[0]
        upper_bound[i] = var[1]
        variable_keys.append(var[2])
    return list(lower_bound), list(upper_bound), variable_keys


# parameter
seed = 1
adaptation_params = [0.0, 0.0, 0.0] # c = 0.01
n_particles = 900
smoothing_lag = 0
output_counter = 100

# read data
#exp_data = pd.read_csv('./data/2015_08_26b/rampIV/3.0(nA).csv')
exp_data = pd.read_csv('./data/toymodel/data2.csv')
i_inj = exp_data.i.values
t = exp_data.t.values
v_observed = exp_data.v.values

# subsampling if wanted
sampling_rate = 0.1
dt = t[1]
step = int(sampling_rate / dt)
data = np.matrix([t[::step], i_inj[::step], v_observed[::step]])

# build model
cm = 1
length = 100
diam = 50
equilibrium_potentials = {'eleak': -54.4, 'ena': 55, 'ek': -77}
leak_channel = IonChannel(g_max=0.0003, equilibrium_potential='eleak', power_gates=[0, 0])
na_channel = IonChannel(g_max=0.12, equilibrium_potential='ena', power_gates=[3, 1],
                        vh=[-39.6051, -62.1596], vs=[9.4690, -7.0678],
                        tau_min=[0.0093, 0.4012], tau_max=[1.0262, 16.0834], tau_delta=[0.4464, 0.3719])
k_channel = IonChannel(g_max=0.036, equilibrium_potential='ek', power_gates=[4, 0],
                       vh=[-51.4643, None], vs=[16.4318, None],
                       tau_min=[0.5235, None], tau_max=[8.9236, None], tau_delta=[0.7980, None])
ionchannels = [leak_channel, na_channel, k_channel]
model = Cell(cm, length, diam, ionchannels, equilibrium_potentials)


# define variables to fit
"""
variables = [
    [0, 1.5, ['ionchannels', '0', 'g_max']],
    [0, 1.5, ['ionchannels', '1', 'g_max']],
    [0, 1.5, ['ionchannels', '2', 'g_max']],
    [-100, 0, ['equilibrium_potentials', 'eleak']],
    [0, 100, ['equilibrium_potentials', 'ena']],
    [-100, 0, ['equilibrium_potentials', 'ek']],
    [-70, -30, ['ionchannels', '1', 'vh', '0']],
    [-70, -30, ['ionchannels', '1', 'vh', '1']],
    [-70, -30, ['ionchannels', '2', 'vh', '0']],
    [5, 25, ['ionchannels', '1', 'vs', '0']],
    [-25, -5, ['ionchannels', '1', 'vs', '1']],
    [5, 25, ['ionchannels', '2', 'vs', '0']],
    [0.008, 1, ['ionchannels', '1', 'tau_min', '0']],
    [0.01, 1, ['ionchannels', '1', 'tau_min', '1']],
    [0.01, 1, ['ionchannels', '2', 'tau_min', '0']],
    [0.01, 20, ['ionchannels', '1', 'tau_max', '0']],
    [0.01, 20, ['ionchannels', '1', 'tau_max', '1']],
    [0.01, 20, ['ionchannels', '2', 'tau_max', '0']],
    [0, 1, ['ionchannels', '1', 'tau_delta', '0']],
    [0, 1, ['ionchannels', '1', 'tau_delta', '1']],
    [0, 1, ['ionchannels', '2', 'tau_delta', '0']],
]"""
variables = [
    [0, 1.5, ['ionchannels', '1', 'g_max']],
    [0, 1.5, ['ionchannels', '2', 'g_max']],
]
lower_bounds_params, upper_bounds_params, keys_params = get_lowerbound_upperbound_keys(variables)

# define bounds of initial v and gates
n_gates = 2 * len(model.ionchannels)
lower_bounds_states = [v_observed[0]] + [1, 0, 0, 1, 0, 1]  # put unused gates to 1, 1
upper_bounds_states = [v_observed[0]] + [1, 1, 1, 1, 1, 1]
lower_bounds_noise = [1, 0, 0.0001]  # 0.15
upper_bounds_noise = [1, 10, 10]  # 10

EM = ExpectationMaximization(data, model, keys_params,
                             lower_bounds_noise, upper_bounds_noise,
                             lower_bounds_params, upper_bounds_params,
                             lower_bounds_states, upper_bounds_states,
                             adaptation_params, n_particles, smoothing_lag,
                             seed, output_counter)
EM.save('./results/toymodel/EM')
Zavg = EM.run()

Zavg = Zavg[:, -1]
Zavg_save = np.asarray(Zavg)
np.savetxt("./results/toymodel/Zavg.csv", Zavg_save, delimiter=",")

# update cell with last params in Yavg
values = Zavg[EM.to_idx['params']]
for keys, value in zip(EM.keys_params, values):
    EM.model.update_attr(keys, value)

# simulate
v_fit, v, a_gates, b_gates = simulate(EM.model, t, i_inj, v0=v_observed[0], p_gates0=None,
                                           std_noise_observed=Zavg[EM.to_idx['std_noise_observed']],
                                           std_noise_intrinsic=Zavg[EM.to_idx['std_noise_intrinsic']],
                                           random_generator=EM.random_generator)

pl.figure()
pl.plot(t, v_observed, 'k', label='V observed')
pl.plot(t, v_fit, 'r', label='V fit')
pl.legend()
pl.show(block=True)