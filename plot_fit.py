import numpy as np
import pickle
from model import *
import matplotlib.pyplot as pl
import pandas as pd

# set seed
seed = 1
random_generator = np.random.RandomState(seed)

Zavg = np.loadtxt("./results/toymodel/Zavg.csv", delimiter=",")
with open("./results/toymodel/EM", 'r') as f:
    EM = pickle.load(f)
exp_data = pd.read_csv('./data/toymodel/data_sx.csv')
i_inj = exp_data.i.values
t = exp_data.t.values
v_observed = exp_data.v.values

# update cell with last params in Yavg
values = Zavg[EM.to_idx['params']]
values = [0.12, 0.036]
for keys, value in zip(EM.keys_params, values):
    EM.model.update_attr(keys, value)
std_noise_observed = 0  # Zavg[EM.to_idx['std_noise_observed']]
std_noise_intrinsic = 1  # Zavg[EM.to_idx['std_noise_intrinsic']]
v0 = v_observed[0]
random_generator = random_generator #EM.random_generator

# simulate
p_gates0 = [[1, 1], [0, 1], [1, 1]]
v_fit, v, a_gates, b_gates = simulate(EM.model, t, i_inj, v0=v0, p_gates0=p_gates0,
                                          std_noise_observed=std_noise_observed,
                                          std_noise_intrinsic=std_noise_intrinsic,
                                          random_generator=random_generator)

pl.figure()
pl.plot(t, v_observed, 'k', label='V observed')
pl.plot(t, v_fit, 'r', label='V fit')
pl.legend()
pl.show()

pl.figure()
pl.plot(t, a_gates[1, :], 'b', label='m')
pl.plot(t, b_gates[1, :], 'g', label='h')
pl.plot(t, a_gates[2, :], 'r', label='n')
pl.legend()
pl.show()