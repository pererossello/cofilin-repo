import os
import sys
import json
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

repo_dir = "/home/pere/code/cofilin-repo/"

sys.path.insert(0, repo_dir)

import h5py
import jax


jax.config.update("jax_enable_x64", True)
print(jax.devices())
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt

from numpyro.infer import MCMC, NUTS, HMC, init_to_value
from numpyro.infer.util import initialize_model
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro import handlers

from cofilin.forward_model.config import (
    Constants,
    FMConfig,
    load_fm_config,
    serialize_dic,
)
from cofilin.forward_model.fmodel import FModel
from cofilin.forward_model.fourier import my_ifft
from cofilin.forward_model.bias import manage_params
from cofilin.forward_model.plot_utils import plot_cubes, compare_pow_spec
from cofilin.forward_model.utils import apply_wiener_filter

from cofilin.recs.utils import make_and_check_dir
from cofilin.recs.plot_utils import compare_fields

this_file_dir = os.path.abspath(os.path.dirname(__file__))

#lastrecfold = os.path.join(this_file_dir, f"{IDX:02d}CH{CHAIN:02d}")

lastrecfold = os.path.join(this_file_dir, f"00CH00/CH01")

savefold = os.path.join(lastrecfold, f"S02")
make_and_check_dir(savefold)

fm_cfg_path = os.path.join(this_file_dir, "data/fm_config.json")
fm_cfg = load_fm_config(fm_cfg_path)
cte = Constants(fm_cfg.N, fm_cfg.L, fm_cfg.Z_I, fm_cfg.Z_F)
fmodel = FModel(fm_cfg)

mock_path = os.path.join(this_file_dir, "data/mock.hdf5")
with h5py.File(mock_path, "r") as f:
    din_ref = jnp.array(f["din"][:])
    n_tr_data = jnp.array(f["n_tr_data"][:])

lastrec_path = os.path.join(lastrecfold, "sample.hdf5")
with h5py.File(lastrec_path, "r") as f:
    q_in = f["params"]["q"][:]
    alpha_in = f["params"]["alpha"][:]
    beta_in = f["params"]["beta"][:]

init_bias_params_constrained = {
    "alpha": alpha_in,
    "beta":  beta_in,
}
positive_transform = biject_to(constraints.positive)
init_params = {'q': q_in}
for k, v in init_bias_params_constrained.items():
     v_unconstr = positive_transform.inv(v)
     init_params[k] = v_unconstr

cfg_path = os.path.join(lastrecfold, "sample_cfg.json")
with open(cfg_path, "rb") as sample_cfg:
    cfg_loaded = pickle.load(sample_cfg)

init_inv_mass_matrix = cfg_loaded["sampling_cfg"]["inv_mass_matrix"]
init_step_size = cfg_loaded["sampling_cfg"]["step_size"]

model = fmodel.build_model()

max_tree_depth = 10
adapt_mass_matrix = False
dense_mass = False
adapt_step_size = False

num_warmup = 0

num_samples = 1000
thinning = 10

sample_seed = 0
key = jax.random.PRNGKey(sample_seed)
kernel = NUTS(
    model,
    dense_mass=dense_mass,
    adapt_mass_matrix=adapt_mass_matrix,
    max_tree_depth=max_tree_depth,
    adapt_step_size=adapt_step_size,
    init_strategy=init_to_value(values=init_params),
    inverse_mass_matrix=init_inv_mass_matrix,
    step_size=init_step_size,
)

mcmc = MCMC(
    kernel,
    num_warmup=num_warmup,
    num_samples=num_samples,
    thinning=thinning,
    progress_bar=True,
)

mcmc.run(key, data=n_tr_data, init_params=init_params)
posterior = mcmc.get_samples()

mcmc_last = mcmc._last_state
mean_accepted_prob = mcmc_last.mean_accept_prob

cfg = {
    "hmc_cfg": {
        "max_tree_depth": max_tree_depth,
        "adapt_mass_matrix": adapt_mass_matrix,
        "adapt_step_size": adapt_step_size,
        "dense_mass": dense_mass,
    },
    "seed": sample_seed,
    "num_warmup": num_warmup,
    "num_samples": num_samples,
    "thinning": thinning,
    "sampling_cfg": {
        "step_size": init_step_size,
        "mean_accepted_prob": mean_accepted_prob,
        "inv_mass_matrix": init_inv_mass_matrix,
    },
}

cfg_path = os.path.join(savefold, "sample_cfg.json")
with open(cfg_path, "wb") as fp:
    pickle.dump(cfg, fp)

sample_path = os.path.join(savefold, "sample.hdf5")
with h5py.File(sample_path, "w") as f:
    grp = f.create_group("params")
    for name, values in posterior.items():
        grp.create_dataset(name, data=values)
