import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


this_file_dir = os.path.abspath(os.path.dirname(__file__))
rebrotes_dir = os.path.join(this_file_dir, "..")
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

N, Z_I, Z_F = 64, 99, 0
R = 3.906 * 1  # * 4
MND = 1e-2
L = R * N
N_TR = MND * L**3
print(f"L: {L:0.2f} Mpc/h")
print(f"N_TR: {N_TR:0.2e}")

input_kind = 'CWN'
lpt_method = "ALPT"
rsd = True
rsd_type = "Radial"
det_bias_model = "PowerLaw"
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
    input_kind=input_kind,
    lpt_method=lpt_method,
    rsd=rsd,
    rsd_type=rsd_type,
    det_bias_model=det_bias_model,
    stoch_bias_model=stoch_bias_model,
    cweb=cweb,
    soft_cweb=soft_cweb,
    cweb_sharpness=cweb_sharpness,
)

params = {
    "alpha": jnp.array([1.05, 1.11, 1.23, 1.3]),
    "beta": jnp.array([7.1, 8.7, 8.1, 9.4]),
}

# params = {"alpha": jnp.array([1.])}

cweb_str = 'S' if soft_cweb else 'H'

SAVEDIR = f"N{N}_R{R:0.0f}_{input_kind}_{lpt_method}_{cweb_str}{cweb}_{det_bias_model}_{stoch_bias_model}_SQ{SEED_INT_Q}"
SAVEDIR = os.path.join(this_file_dir, SAVEDIR)
print(SAVEDIR)

make_mock(fm_cfg, SAVEDIR, SEED_INT_Q, params, SEED_INT_N_TR, saveplot=True)
