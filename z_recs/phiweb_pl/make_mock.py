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

from cofilin.forward_model.fourier import my_ifft
from cofilin.forward_model.config import Constants, FMConfig
from cofilin.forward_model.fmodel import FModel
from cofilin.forward_model.plot_utils import plot_cubes

from cofilin.recs.mock_maker import make_mock

SEED_INT_Q = 1
SEED_INT_N_TR = 1

N, Z_I, Z_F = 128, 99, 0.1
R = 3.4 * 2  # * 4
MND = 1e-2
L = R * N
N_TR = 1e-2 * L**3
print(f"L: {L:0.2f} Mpc/h")
print(f"N_TR: {N_TR:0.2e}")

lpt_method = "2LPT"
rsd = True
det_bias_model = "HighPassPowerLaw"
stoch_bias_model = "NegBinomial"
cweb = "PhiWeb"
soft_cweb = True
cweb_sharpness = 10

fm_cfg = FMConfig(
    N,
    L,
    Z_I,
    Z_F,
    N_TR=N_TR,
    lpt_method=lpt_method,
    rsd=rsd,
    det_bias_model=det_bias_model,
    stoch_bias_model=stoch_bias_model,
    cweb=cweb,
    soft_cweb=soft_cweb,
    cweb_sharpness=cweb_sharpness,
)

params = {
    "alpha": jnp.array([1.0, 1.2, 1.05, 1.2]),
    "e_hp": jnp.array([1.2, 1.3, 1.5, 1.2]),
    "rho_hp": jnp.array([1., 0.9, 1.2, 1.1]),
    "beta": jnp.array([14.2, 12.1, 11.2, 10.5]),
}

rsd_str = "RSD_" if rsd else ""
SAVEDIR = f"N{N}_R{R:0.0f}_{cweb}_{det_bias_model}_{stoch_bias_model}_SQ{SEED_INT_Q}"
SAVEDIR = os.path.join(this_file_dir, SAVEDIR)
print(SAVEDIR)

make_mock(fm_cfg, SAVEDIR, SEED_INT_Q, params, SEED_INT_N_TR, saveplot=True)
