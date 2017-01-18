import numpy as np
from copy import copy


class Cell:
    def __init__(self, cm, length, diam, ionchannels, equilibrium_potentials):
        self.cm = cm  # (uF/cm2)
        self.length = length  # (um)
        self.diam = diam  # (um)
        self.cell_area = self.length * self.diam * np.pi * 1e-8  # (cm2)
        self.ionchannels = ionchannels  # list of IonChannels
        self.equilibrium_potentials = equilibrium_potentials  # dict (mV)

    def derivative_v(self, i_ion, i_inj):
        i_ion = copy.copy(i_ion) * self.cell_area  # (mA)
        cm = self.cm * self.cell_area * 1e-3  # (mF)
        i_inj *= 1e-6  # (mA)
        return (-1 * np.sum(i_ion, 0) + i_inj) / cm  # (mV/ms)

    def update_attr(self, keys, value):
        attr = reduce(lambda k: getattr(self, k), keys[:-1])
        setattr(attr[keys[-1]], value)


class IonChannel:
    def __init__(self, g_max, equilibrium_potential, n_gates, power_gates, vh, vs, tau_min, tau_max, tau_delta):
        self.g_max = g_max  # (S/cm2)
        self.equilibrium_potential = equilibrium_potential  # name: str
        self.n_gates = n_gates
        self.power_gates = np.array(power_gates)
        self.vh = vh
        self.vs = vs
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_delta = tau_delta

    def compute_current(self, vs, p_gates, equilibrium_potentials):
        if self.n_gates == 0:
            return self.g_max * (vs - equilibrium_potentials[self.equilibrium_potential])
        else:
            return self.g_max * np.prod(p_gates**self.power_gates) * (vs - equilibrium_potentials[self.equilibrium_potential])  # (mA/cm2)

    def init_gates(self, v0, p_gates0=None):
        if p_gates0 is not None:
            return p_gates0
        else:
            return self.inf_gates(v0)

    def derivative_gates(self, vs, p_gate):
        inf_gates = self.inf_gates(vs)
        tau_gates = self.tau_gates(vs, inf_gates)
        return [(inf_gates[0] - p_gate[0]) / tau_gates[0],
                (inf_gates[1] - p_gate[1]) / tau_gates[1]]

    def inf_gates(self, v):
        return [1 / (1 + np.exp((self.vh[0] - v) / self.vs[0])),
                1 / (1 + np.exp((self.vh[0] - v) / self.vs[0]))]

    def tau_gates(self, v, inf_gates):
        return [self.tau_min[0] + (self.tau_max[0] - self.tau_min[0]) * inf_gates[0]
                * np.exp(self.tau_delta[0] * (self.vh[0] - v) / self.vs[0]),
                self.tau_min[1] + (self.tau_max[1] - self.tau_min[1]) * inf_gates[1]
                * np.exp(self.tau_delta[1] * (self.vh[1] - v) / self.vs[1])
                ]


def simulate_cell(cell, t, i_inj, v0, p_gates0=None, std_noise_observed=1, std_noise_intrinsic=1,
                  random_generator=None):

    # allocate memory
    v = np.zeros(len(t))
    v_observed = np.zeros(len(t))
    a_gates = np.zeros((len(cell.ionchannels), len(t)))
    b_gates = np.zeros((len(cell.ionchannels), len(t)))

    # initial conditions
    v[0] = v0
    a_gates[:, 0] = [cell.ionchannels[i].init_gates(v0, p_gates0)[0] for i in range(len(cell.ionchannels))]
    b_gates[:, 0] = [cell.ionchannels[i].init_gates(v0, p_gates0)[1] for i in range(len(cell.ionchannels))]

    # solve differential equation
    dt = np.diff(t)
    for ts in range(1, len(t)):
        v[ts], a_gates[:, ts], b_gates[:, ts] = update_voltage_and_gates(cell, v[ts],
                                                                       a_gates[:, ts - 1], b_gates[:, ts - 1],
                                                                         i_inj[ts - 1], dt[ts - 1],
                                                                         std_noise_intrinsic, random_generator)
        v_observed[ts] = v[ts] + std_noise_observed * random_generator.gauss(0, 1)

    return v_observed, v, a_gates, b_gates


def update_voltage_and_gates(cell, v, a_gates, b_gates, i_inj, dt, std_noise_intrinsic, random_generator):

    # compute ionic current
    i_ion = np.array([cell.ionchannels[i].compute_current(v, [a_gates[i], b_gates[i]])
                      for i in range(len(cell.ionchannels))])

    # compute derivatives
    dvdt = cell.derivative_v(i_ion, i_inj)
    dgatesdt = [cell.ionchannels[i].derivative_gates(v, [a_gates[i], b_gates[i]])
            for i in range(len(cell.ionchannels))]

    da_gatesdt, db_gatesdt = map(list, zip(*dgatesdt))  # transpose list to divide between a_gates and b_gates

    # update states
    v = v + dvdt * dt + std_noise_intrinsic * np.sqrt(dt) * random_generator.gauss(0, 1)
    a_gates = a_gates + da_gatesdt * dt
    b_gates = b_gates + db_gatesdt * dt

    return v, a_gates, b_gates