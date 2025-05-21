import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

this_file_dir = os.path.abspath(os.path.dirname(__file__))
rebrotes_dir = os.path.join(this_file_dir, "..", "..")
sys.path.insert(0, rebrotes_dir)

import jax

jax.config.update("jax_enable_x64", True)
print(jax.devices())
import jax.numpy as jnp
import matplotlib.pyplot as plt

from cofilin.forward_model.config import Constants, FMConfig
from cofilin.forward_model.fmodel import FModel
from cofilin.forward_model.fourier import my_ifft
from cofilin.forward_model.bias import manage_params
from cofilin.forward_model.stats import get_pow_spec_1D, get_pow_spec_quadrupole
from cofilin.forward_model.plot_utils import plot_cubes, compare_pow_spec


N, Z_I, Z_F = 256, 99, 0
R = 3.906
L = R * N

cte = Constants(N, L, Z_I, Z_F)

lpt_method = "ALPT"
MND = 1e-2
N_TR = L**3 * MND

det_bias_model = "PowerLaw"
stoch_bias_model = "Poisson"

cweb = "PhiWeb"

fm_cfg = FMConfig(
    N,
    L,
    Z_I,
    Z_F,
    N_TR=N_TR,
    rsd=False, 
    lpt_method=lpt_method
)
fmodel = FModel(fm_cfg)

fm_cfg_rsd_plane = FMConfig(
    N,
    L,
    Z_I,
    Z_F,
    N_TR=N_TR,
    rsd=True,
    rsd_type='Plane',
    lpt_method=lpt_method
)
fmodel_rsd_plane = FModel(fm_cfg_rsd_plane)

fm_cfg_rsd_rad = FMConfig(
    N,
    L,
    Z_I,
    Z_F,
    N_TR=N_TR,
    rsd=True,
    rsd_type='Radial',
    lpt_method=lpt_method
)
fmodel_rsd_rad = FModel(fm_cfg_rsd_rad)



q_data = fmodel.input_arr(jax.random.PRNGKey(1))

delta_lpt = fmodel.delta_lpt(q_data)
delta_lpt_rsd_plane = fmodel_rsd_plane.delta_lpt(q_data)
delta_lpt_rsd_rad = fmodel_rsd_rad.delta_lpt(q_data)

fig, ax = plot_cubes(
    [delta_lpt, delta_lpt_rsd_plane, delta_lpt_rsd_rad],
    cmap=["gnuplot"]*3,
    vlim=[(-1, 10)]*3,
    width=1, axis=2, idx=N//2,
    figsize=7.5
)
fig.savefig('maps.png', bbox_inches='tight')



n_bins = 50
ks, pk = get_pow_spec_1D(delta_lpt, L, n_bins)
ks, pk_rsd_plane = get_pow_spec_1D(delta_lpt_rsd_plane, L, n_bins)
ks, pk_rsd_rad = get_pow_spec_1D(delta_lpt_rsd_rad, L, n_bins)

ks, pk_q = get_pow_spec_quadrupole(delta_lpt, L, n_bins, direction='Radial')
ks, pk_q_rsd_plane = get_pow_spec_quadrupole(delta_lpt_rsd_plane, L, n_bins, direction='PlaneParallel')
ks, pk_q_rsd_rad = get_pow_spec_quadrupole(delta_lpt_rsd_rad, L, n_bins, direction='Radial')

fs, rat = 5, 2
fig, axs = plt.subplots(1, 2, figsize=(fs*rat, fs))

axs[0].plot(ks, pk, c='k', ls='-', alpha=1, label=None)
axs[0].plot(ks, pk_rsd_plane, c='r', ls='--', alpha=1, label=None)
axs[0].plot(ks, pk_rsd_rad, c='b', ls='--', alpha=1, label=None)

axs[1].plot(ks, pk_q, c='k', ls='-', alpha=1, label=None)
axs[1].plot(ks, pk_q_rsd_plane, c='r', ls='--', alpha=1, label=None)
axs[1].plot(ks, pk_q_rsd_rad, c='b', ls='--', alpha=1, label=None)

for ax in axs:
    ax.grid(True)
    
axs[0].set_xscale('log')
axs[0].set_yscale('log')

fig.savefig('pk.png', bbox_inches='tight')
