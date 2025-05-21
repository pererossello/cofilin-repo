import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

this_file_dir = os.path.abspath(os.path.dirname(__file__))
repo_dir = "/home/pere/code/cofilin-repo"
sys.path.append(repo_dir)

from tqdm import tqdm
import numpy as np
import h5py
import jax
print(jax.devices())
import jax.numpy as jnp

import matplotlib.pyplot as plt
from jax.tree_util import Partial as partial

from cofilin.forward_model.config import Constants, FMConfig, load_fm_config
from cofilin.forward_model.fmodel import FModel
from cofilin.forward_model.ics import get_delta_in
from cofilin.forward_model.fourier import my_ifft

recdir = "/home/pere/code/cofilin-repo/z_final_recs/N64_R4_PhiWeb_PowerLaw_NegBinomial_SQ1"

sampledir = "/home/pere/code/cofilin-repo/z_final_recs/N64_R4_PhiWeb_PowerLaw_NegBinomial_SQ1/00CH00/CH01/CH02/CH03"


fm_cfg_path = os.path.join(recdir, "data/fm_config.json")
fm_cfg = load_fm_config(fm_cfg_path)
cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)


sample_path = os.path.join(sampledir, "sample.hdf5")
with h5py.File(sample_path, "r") as f:
    q_rec = f["params"]["q"][:]
    alpha_rec = f["params"]["alpha"][:]
    beta_rec = f["params"]["beta"][:]

N_SAMPLES = q_rec.shape[0]


N_VOXELS = 10000
max_lag = N_SAMPLES//10
max_lag = 5
#max_lag = 5
print('max lag', max_lag)
seed_int = 1
max_relative_k=0.1

ks = jnp.arange(max_lag)
key = jax.random.PRNGKey(seed_int)
voxel_idxs = jax.random.randint(key, (N_VOXELS,), 0, fm_cfg.N**3 - 1)

def autocorr_lag_k(x, ks):
    N = x.shape[1]
    mean_x = jnp.mean(x, axis=1)
    var_x = jnp.var(x, axis=1)
    delta_x = x - mean_x[:, None]
    fft_x = jnp.fft.rfft(delta_x, n=2 * N - 1, axis=1)
    auto_full = jnp.fft.irfft(fft_x * jnp.conj(fft_x), axis=1)
    return auto_full[:, ks] / (var_x[:, None] * (N - ks))

if ks.max() >= N_SAMPLES:
    raise ValueError(f"Not enough samples for lag {ks.max()}!")

# Determine practical maximum lag
practical_max_k = int(min(max_relative_k * N_SAMPLES, ks.max()))
if practical_max_k < ks.max():
    warnings.warn(
        f"Lags above {practical_max_k} may be unreliable. Reducing max lag."
    )

arr = jnp.zeros((N_VOXELS, N_SAMPLES))
for i in range(N_SAMPLES):
    q = q_rec[i]
    q = my_ifft(get_delta_in(q, cte, fm_cfg), cte.INV_L3)
    q_flat = jnp.array(q).ravel()
    q_idxed = q_flat[voxel_idxs]
    arr = arr.at[:, i].set(q_idxed)

c = autocorr_lag_k(arr, ks)  # shape (N_VOXELS, len(ks))

x_max = None
title = ''

ts = 3
fs, r = 7, 2
fig, ax = plt.subplots(1, 1, figsize=(fs * r, fs))

lw = 0.5 * fs
labelsize = 3.25*fs

# for n in range(c.shape[0]):
#     ax.plot(c[n, :], c="k", alpha=0.05)

fact = 10
#fact = 1
x = jnp.arange(max_lag, )
x *= fact

c_mean = c.mean(axis=0)
c_std = c.std(axis=0)
ax.plot(x, c_mean, c="r", alpha=1, lw=lw, label="Mean")
ax.fill_between(x, c_mean-c_std, c_mean+c_std, alpha=0.2, color='r', lw=0)


ax.hlines(0.1, xmin=0, xmax=x.max(), color="k", ls="--", lw=lw*0.75)
#ax.text(3, 0.1*1.01, r'$0.1$', ha='left', va='bottom', fontsize=labelsize)
#ax.hlines(0.0, xmin=0, xmax=x.max(), color="k", lw=lw*0.75)


sw = 0.18*fs
tick_length = sw*6
for spine in ax.spines.values():
    spine.set_linewidth(sw)
ax.tick_params(which='major', direction='inout', width=sw,  length=tick_length)
ax.tick_params(which='minor', direction='inout', width=sw, length=tick_length*0.8)
ax.tick_params(labelsize=labelsize)
ax.grid(
    which="major",
    linewidth=0.5,
    color="k",
    alpha=0.1,
)

ax.set_ylabel("Autocorrelation at lag $m$", fontsize=labelsize)
ax.set_xlabel("$m$", fontsize=labelsize)
ax.legend(fontsize=labelsize)

ax.set_ylim(-0.5, 1.1)
ax.set_xlim(x.min(), x.max())
# if x_max is not None:
#     ax.set_xlim(0, x_max)

if title is not None:
    ax.set_title(title, fontsize=ts * fs)

figpath = os.path.join(this_file_dir, 'figures', 'corr_length.png')
fig.savefig(figpath, bbox_inches='tight')
figpath = os.path.join(this_file_dir, 'figures', 'corr_length.pdf')
fig.savefig(figpath, bbox_inches='tight')
