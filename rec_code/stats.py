import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

repo_dir = "/home/pere/code/cofilin-repo"
sys.path.append(repo_dir)

from tqdm import tqdm
import numpy as np
import h5py
import jax
import json

print(jax.devices())
import jax.numpy as jnp

import matplotlib.pyplot as plt
from jax.tree_util import Partial as partial

from cofilin.forward_model.config import Constants, FMConfig, load_fm_config
from cofilin.forward_model.fmodel import FModel
from cofilin.forward_model.bias import manage_params
from cofilin.forward_model.fourier import my_ifft
from cofilin.forward_model.plot_utils import (
    plot_cubes,
    compare_pow_spec,
    stats_plot_for_article,
)
from cofilin.forward_model.stats import (
    get_pow_spec_1D,
    get_cross_power_spec_1D,
    get_reduced_bispectrum,
)
from cofilin.recs.utils import make_and_check_dir

def get_delta_n_tr(n_tr):
    return (n_tr.shape[0] ** 3) / jnp.sum(n_tr) * n_tr - 1


savefold = os.path.join(repo_dir, f"z_final_results/results/")
make_and_check_dir(savefold)


recdir = "/home/pere/code/cofilin-repo/z_final_recs/N64_R4_PhiWeb_PowerLaw_NegBinomial_SQ1"
sampledir = os.path.join(recdir, "/00CH00/CH01/CH02/CH03")

fm_cfg_path = os.path.join(recdir, "data/fm_config.json")
fm_cfg = load_fm_config(fm_cfg_path)
fmodel = FModel(fm_cfg)
get_n_tr_mean = fmodel.n_tr_mean()

cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)

mock_path = os.path.join(recdir, "data/mock.hdf5")
with h5py.File(mock_path, "r") as f:
    q = jnp.array(f["input"][:])
    din_ref = jnp.array(f["din"][:])
    n_tr_data = jnp.array(f["n_tr_data"][:])
    n_tr_mean = jnp.array(f["n_tr_mean"][:])
dn_tr_data = get_delta_n_tr(n_tr_data)


params_ref_path = os.path.join(recdir, "data/params.json")
with open(params_ref_path, "r") as f:
    params_ref = json.load(f)
params_ref = manage_params(params_ref)


N = cte.N
L = cte.L
R = L / N
sphere_only = False
n_bins = 40
n_thetas = 50
k1, k2 = 0.1, 0.2


get_pw_ = partial(get_pow_spec_1D, L=L, n_bins=n_bins, sphere_only=sphere_only)


def get_pw(arr):
    _, pk = get_pw_(arr)
    return pk


get_pw = jax.jit(get_pw)

thetas = jnp.linspace(0, jnp.pi, n_thetas)
get_bispec = partial(get_reduced_bispectrum, L=L, k1=k1, k2=k2, thetas=thetas)
get_bispec = jax.jit(get_bispec)

get_cross_din_ = partial(
    get_cross_power_spec_1D,
    delta_1=din_ref,
    L=L,
    n_bins=n_bins,
    sphere_only=sphere_only,
)
get_cross_dn_tr_ = partial(
    get_cross_power_spec_1D,
    delta_1=dn_tr_data,
    L=L,
    n_bins=n_bins,
    sphere_only=sphere_only,
)


def get_cross_din(arr):
    _, ck = get_cross_din_(delta_2=arr)
    return ck


def get_cross_dn_tr(arr):
    _, ck = get_cross_dn_tr_(delta_2=arr)
    return ck


get_cross_din = jax.jit(get_cross_din)
get_cross_dn_tr = jax.jit(get_cross_dn_tr)

k_c, din_ref_pk = get_pw_(din_ref)
din_ref_bispec = get_bispec(din_ref)

ntr_ref_pk = get_pw(dn_tr_data)
ntr_ref_bispec = get_bispec(dn_tr_data)

sample_path = os.path.join(sampledir, "sample.hdf5")
with h5py.File(sample_path, "r") as f:
    q_rec = f["params"]["q"][:]
    alpha_rec = f["params"]["alpha"][:]
    beta_rec = f["params"]["beta"][:]

M = q_rec.shape[0]

# DIN
din_pks = jnp.zeros((n_bins, M))
din_pk_rats = jnp.zeros((n_bins, M))

din_bispecs = jnp.zeros((n_thetas, M))
din_bispec_rats = jnp.zeros((n_thetas, M))

din_crosses = jnp.zeros((n_bins, M))

# N_TR
ntr_pks = jnp.zeros((n_bins, M))
ntr_pk_rats = jnp.zeros((n_bins, M))

ntr_bispecs = jnp.zeros((n_thetas, M))
ntr_bispec_rats = jnp.zeros((n_thetas, M))

ntr_crosses = jnp.zeros((n_bins, M))

# Optimal cross
K = 5
sample_crosses = jnp.zeros((n_bins, K))
sks = jax.random.split(jax.random.PRNGKey(1), K)
print("Computing optimal cross")
for i in tqdm(range(K), total=K):
    dev = fmodel.delta_lpt(q)
    cweb_arr = fmodel.cweb(dev)

    params_ref = manage_params(params_ref)
    n_tr_mean = get_n_tr_mean(dev, params_ref, cweb_arr)
    n_tr_sample = fmodel.sample_n_tr(n_tr_mean,sks[i], params=params_ref, cweb=cweb_arr)

    dn_tr_sample = get_delta_n_tr(n_tr_sample)

    ck = get_cross_dn_tr(dn_tr_sample)
    sample_crosses = sample_crosses.at[:, i].set(ck)
optimal_cross = jnp.mean(sample_crosses, axis=1)


datapath = os.path.join(savefold, f"N{N}_R{R:0.0f}.hdf5")
with h5py.File(datapath, "w") as h5_file:
    header = h5_file.create_group("Header")
    header.attrs["N"] = cte.N
    header.attrs["L"] = cte.L
    header.attrs["R"] = cte.R
    header.attrs["Z_I"] = cte.Z_I
    header.attrs["Z_F"] = cte.Z_F

    header.attrs["N_paths"] = M

    header.attrs["n_bins"] = n_bins
    header.attrs["n_thetas"] = n_thetas
    header.attrs["k1"] = k1
    header.attrs["k2"] = k2

    h5_file.create_dataset("k_c", data=k_c, dtype="float32")
    h5_file.create_dataset("thetas", data=thetas, dtype="float32")
    h5_file.create_dataset("din_ref_pk", data=din_ref_pk, dtype="float32")
    h5_file.create_dataset("ntr_ref_pk", data=ntr_ref_pk, dtype="float32")
    h5_file.create_dataset("din_ref_bispec", data=din_ref_bispec, dtype="float32")
    h5_file.create_dataset("ntr_ref_bispec", data=ntr_ref_bispec, dtype="float32")
    h5_file.create_dataset("optimal_cross", data=optimal_cross, dtype="float32")



print("Computing rec stats")
sks = jax.random.split(jax.random.PRNGKey(3), M)
for i in tqdm(range(M), total=M):
    q = q_rec[i]
    alpha = alpha_rec[i]
    beta = beta_rec[i]
    params = {"alpha": alpha, "beta": beta}
    params = manage_params(params)

    din = my_ifft(fmodel.delta_in(q), cte.INV_L3)
    dev = fmodel.delta_lpt(q)
    cweb_arr = fmodel.cweb(dev)

    n_tr_mean = get_n_tr_mean(dev, params_ref, cweb_arr)
    n_tr = fmodel.sample_n_tr(n_tr_mean,sks[i], params=params_ref, cweb=cweb_arr)
    dn_tr = get_delta_n_tr(n_tr)


    # PK
    # din pk
    pk = get_pw(din)
    din_pks = din_pks.at[:, i].set(pk)
    din_pk_rats = din_pk_rats.at[:, i].set(pk / din_ref_pk)

    # n_tr pk
    pk = get_pw(dn_tr)
    ntr_pks = ntr_pks.at[:, i].set(pk)
    ntr_pk_rats = ntr_pk_rats.at[:, i].set(pk / ntr_ref_pk)

    # BISPEC
    # din bispec
    bk = get_bispec(din)
    din_bispecs = din_bispecs.at[:, i].set(bk)
    din_bispec_rats = din_bispec_rats.at[:, i].set(bk/din_ref_bispec)

    # n_tr bispec
    bk = get_bispec(dn_tr)
    ntr_bispecs = ntr_bispecs.at[:, i].set(bk)
    ntr_bispec_rats = ntr_bispec_rats.at[:, i].set(bk/ntr_ref_bispec)

    # CROSS
    # din cross
    ck = get_cross_din(din)
    din_crosses = din_crosses.at[:, i].set(ck)

    # ntr cross
    ck = get_cross_dn_tr(dn_tr)
    ntr_crosses = ntr_crosses.at[:, i].set(ck)

with h5py.File(datapath, "a") as h5_file:
    h5_file.create_dataset("din_pks", data=din_pks, dtype="float32")
    h5_file.create_dataset("din_pk_rats", data=din_pk_rats, dtype="float32")

    h5_file.create_dataset("ntr_pks", data=ntr_pks, dtype="float32")
    h5_file.create_dataset("ntr_pk_rats", data=ntr_pk_rats, dtype="float32")

    h5_file.create_dataset("din_bispecs", data=din_bispecs, dtype="float32")
    h5_file.create_dataset("din_bispec_rats", data=din_bispec_rats, dtype="float32")

    h5_file.create_dataset("ntr_bispecs", data=ntr_bispecs, dtype="float32")
    h5_file.create_dataset("ntr_bispec_rats", data=ntr_bispec_rats, dtype="float32")

    h5_file.create_dataset("din_crosses", data=din_crosses, dtype="float32")
    h5_file.create_dataset("ntr_crosses", data=ntr_crosses, dtype="float32")


propagator = get_cross_din(dn_tr_data)
with h5py.File(datapath, "a") as h5_file:
    h5_file.create_dataset("propagator", data=propagator, dtype="float32")