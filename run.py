import pandas as pd
import matplotlib.pyplot as pl
from model import *

# parameter
seed = 1.0
n_particles = 4000
smoothing_lag = 100
output_counter = 1000

# read data
exp_data = pd.read_csv('./data/2015_08_26b/rampIV/3.0(nA).csv')
i_inj = exp_data.i.values
t = exp_data.t.values
v_observed = exp_data.v.values

# subsampling if wanted
sampling_rate = 0.01
dt = t[1]
step = sampling_rate / dt
data = [t[::step], i_inj[::step], v_observed[::step]]

# build model
cm = 1
length = 100
diam = 50
leak_channel = IonChannel(g_max=0.01, equilibrium_potential=-60, n_gates=0, power_gates=[0, 0], vh, vs, tau_min, tau_max, tau_delta)
na_channel = IonChannel(g_max=1.2, equilibrium_potential=60, n_gates=2, power_gates=[3, 1], vh, vs, tau_min, tau_max, tau_delta)
k_channel = IonChannel(g_max=0.36, equilibrium_potential=-60, n_gates=2, power_gates=[3, 1], vh, vs, tau_min, tau_max, tau_delta)
nap_channel = IonChannel(g_max=0.05, equilibrium_potential=60, n_gates=2, power_gates=[3, 1], vh, vs, tau_min, tau_max, tau_delta)
ionchannels = [leak_channel, na_channel, k_channel, nap_channel]
model = Cell(cm, length, diam, ionchannels)

variables = [
    [0, 1.5, ['ionchannels', '0', 'g_max']],
    [0, 1.5, ['ionchannels', '1', 'g_max']],
    [0, 1.5, ['ionchannels', '2', 'g_max']],
    [-90, -50, ['equilibrium_potentials', 'eleak']],
    [40, 100, ['equilibrium_potentials', 'ena']],
    [-90, -50, ['equilibrium_potentials', 'ek']],
    [0, 1.5, ['ionchannels', '2', 'g_max']],
    [0, 1.5, ['ionchannels', '3', 'g_max']],
    [0, 1.5, ['ionchannels', '0', 'g_max']],
    [0, 1.5, ['ionchannels', '1', 'g_max']],
    [0, 1.5, ['ionchannels', '2', 'g_max']],
    [0, 1.5, ['ionchannels', '3', 'g_max']],
]

[Zavg, Q] = filter_mdl_stellate(data, n_particles, smoothing_lag, output_counter)

Zavg.to_csv('./results/2015_08_26b/Yavg.csv', Zavg)
Q.to_csv('./results/2015_08_26b/Q.csv', Q)

# Zavg = csvread('./data/Yavg.csv')

v_inferred, X = simulate_mdl_stellate_with_new_params(t, Iinj, Zavg, seed)

pl.figure()
pl.plot(t, v_observed, 'k', label='V observed')
pl.plot(t, v_inferred, 'r', label='V inferred')
pl.legend()
pl.show()