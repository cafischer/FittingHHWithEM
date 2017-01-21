import numpy as np
import copy


class Cell:
    def __init__(self, cm, length, diam, ionchannels, equilibrium_potentials):
        self.cm = cm  # (uF/cm2)
        self.length = length  # (um)
        self.diam = diam  # (um)
        self.cell_area = self.length * self.diam * np.pi * 1e-8  # (cm2)
        self.ionchannels = ionchannels  # list of IonChannels
        self.equilibrium_potentials = equilibrium_potentials  # dict (mV)

    def compute_dvdt(self, i_ion, i_inj):
        i_ion = copy.copy(i_ion) #* self.cell_area  # (mA)
        cm = self.cm * 1e-3 #* self.cell_area # (mF)
        i_inj *= 1e-3 #1e-6  # (mA) TODO
        return (-1 * np.sum(i_ion, 0) + i_inj) / cm  # (mV/ms)

    def update_attr(self, keys, value):
        keys = [self] + keys
        attr_carrier = reduce(lambda o, k: self.get_attr(o, k), keys[:-1])
        self.set_attr(attr_carrier, keys[-1], value)

    def get_attr(self, x, key):
        if key.isdigit():
            return x[int(key)]
        else:
            return getattr(x, key)

    def set_attr(self, x, key, value):
        if isinstance(x, dict):
            x[key] = value
        elif key.isdigit():
            x[int(key)] = value
        else:
            setattr(x, key, value)


class IonChannel:
    def __init__(self, g_max, equilibrium_potential, power_gates,
                 vh=None, vs=None, tau_min=None, tau_max=None, tau_delta=None):
        self.g_max = g_max  # (S/cm2)
        self.equilibrium_potential = equilibrium_potential  # name: str
        self.power_gates = np.array(power_gates)
        self.vh = vh
        self.vs = vs
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_delta = tau_delta

    def compute_current(self, v, p_gates, equilibrium_potentials):
        return self.g_max * np.prod(p_gates**self.power_gates)\
                * (v - equilibrium_potentials[self.equilibrium_potential])  # (mA/cm2)

    def init_gates(self, v0, p_gates0=None):
        p_gates = np.ones(2)
        for i in [0, 1]:
            if self.power_gates[i] == 0:
                p_gates[i] = 1   # 1 is neutral element (gate totally open)
            elif p_gates0 is not None:
                p_gates[i] = p_gates0[i]
            else:
                p_gates[0] = self.inf_gates(v0, i)
        return p_gates

    def compute_dgatesdt(self, vs, p_gate):
        dgatesdt = np.zeros(2)
        for i in [0, 1]:
            if self.power_gates[i] == 0:
                dgatesdt[i] = 0  # 0 neutral element (gate will stay totally open)
            else:
                inf_gate = self.inf_gates(vs, i)
                tau_gate = self.tau_gates(vs, inf_gate, i)
                dgatesdt[i] = (inf_gate - p_gate[i]) / tau_gate
        return dgatesdt

    def inf_gates(self, v, i):
        return 1 / (1 + np.exp((self.vh[i] - v) / self.vs[i]))

    def tau_gates(self, v, inf_gate, i):
        return self.tau_min[i] + (self.tau_max[i] - self.tau_min[i]) * inf_gate \
                                 * np.exp(self.tau_delta[i] * (self.vh[i] - v) / self.vs[i])


def simulate(cell, t, i_inj, v0, p_gates0=None, std_noise_observed=1, std_noise_intrinsic=1,
                  random_generator=None):

    # allocate memory
    v = np.zeros(len(t))
    v_observed = np.zeros(len(t))
    a_gates = np.zeros((len(cell.ionchannels), len(t)))
    b_gates = np.zeros((len(cell.ionchannels), len(t)))

    # initial conditions
    if p_gates0 is None:
        p_gates0 = [None, None, None]
    v[0] = v0
    v_observed[0] = v0
    a_gates[:, 0] = [cell.ionchannels[i].init_gates(v0, p_gates0[i])[0] for i in range(len(cell.ionchannels))]
    b_gates[:, 0] = [cell.ionchannels[i].init_gates(v0, p_gates0[i])[1] for i in range(len(cell.ionchannels))]

    # solve differential equation
    dt = np.diff(t)
    for ts in range(1, len(t)):
        v[ts], a_gates[:, ts], b_gates[:, ts] = update_voltage_and_gates(cell, v[ts - 1],
                                                                         a_gates[:, ts - 1], b_gates[:, ts - 1],
                                                                         i_inj[ts - 1], dt[ts - 1],
                                                                         std_noise_intrinsic, random_generator)
        v_observed[ts] = v[ts] + std_noise_observed * random_generator.randn()

    return v_observed, v, a_gates, b_gates


def update_voltage_and_gates(cell, v, a_gates, b_gates, i_inj, dt, std_noise_intrinsic, random_generator):

    # compute ionic current
    i_ion = np.array([cell.ionchannels[i].compute_current(v, [a_gates[i], b_gates[i]], cell.equilibrium_potentials)
                      for i in range(len(cell.ionchannels))])

    # compute derivatives
    dvdt = cell.compute_dvdt(i_ion, i_inj)
    dgatesdt = [cell.ionchannels[i].compute_dgatesdt(v, [a_gates[i], b_gates[i]])
            for i in range(len(cell.ionchannels))]

    da_gatesdt, db_gatesdt = map(list, zip(*dgatesdt))  # transpose list to divide between a_gates and b_gates

    # update states
    v = v + dvdt * dt + std_noise_intrinsic * np.sqrt(dt) * random_generator.randn()
    a_gates = a_gates + np.array(da_gatesdt) * dt
    b_gates = b_gates + np.array(db_gatesdt) * dt

    return v, a_gates, b_gates