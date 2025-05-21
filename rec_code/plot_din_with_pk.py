import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

this_file_dir = os.path.abspath(os.path.dirname(__file__))
rebrotes_dir = os.path.join(this_file_dir, "..")
sys.path.insert(0, rebrotes_dir)

from tqdm import tqdm
import numpy as np
import h5py

import matplotlib.pyplot as plt
from cofilin.forward_model.plot_utils import stats_plot_for_article
from cofilin.forward_model.cosmo import get_linear_power_1D, get_linear_power_1D_2

N = 64
R = 4
results_dir = os.path.join(this_file_dir, 'results')
datapath = os.path.join(results_dir, f'N{N}_R{R:0.0f}.hdf5') 
with h5py.File(datapath, "r") as h5_file:

    header = h5_file['Header']
    M1 = header.attrs['N_paths']

    k_c = np.array(h5_file['k_c'][:])
    thetas = np.array(h5_file['thetas'][:])

    din_ref_pk = np.array(h5_file["din_ref_pk"][:])
    din_pks = np.array(h5_file["din_pks"][:])
    din_pk_rats = np.array(h5_file["din_pk_rats"][:])

    ntr_ref_pk = np.array(h5_file["ntr_ref_pk"][:])
    ntr_pks = np.array(h5_file["ntr_pks"][:])
    ntr_pk_rats = np.array(h5_file["ntr_pk_rats"][:])

    din_ref_bispec = np.array(h5_file["din_ref_bispec"][:])
    din_bispecs = np.array(h5_file["din_bispecs"][:])
    din_bispec_rats = np.array(h5_file["din_bispec_rats"][:])

    ntr_ref_bispec = np.array(h5_file["ntr_ref_bispec"][:])
    ntr_bispecs = np.array(h5_file["ntr_bispecs"][:])
    ntr_bispec_rats = np.array(h5_file["ntr_bispec_rats"][:])

    din_crosses = np.array(h5_file["din_crosses"][:])
    ntr_crosses = np.array(h5_file["ntr_crosses"][:])

    propagator = np.array(h5_file["propagator"][:])


# print(M1)
# print(M2)

M_bi = 0

# Power spectra
din_pks_bi = din_pks[:, :M_bi]
din_pks = din_pks[:, M_bi:]

din_pk_rats_bi = din_pk_rats[:, :M_bi]
din_pk_rats = din_pk_rats[:, M_bi:]

ntr_pks_bi = ntr_pks[:, :M_bi]
ntr_pks = ntr_pks[:, M_bi:]

ntr_pk_rats_bi = ntr_pk_rats[:, :M_bi]
ntr_pk_rats = ntr_pk_rats[:, M_bi:]

# Bispectra

din_bispecs_bi = din_bispecs[:, :M_bi]
din_bispecs = din_bispecs[:, M_bi:]

din_bispec_rats_bi = din_bispec_rats[:, :M_bi]
din_bispec_rats = din_bispec_rats[:, M_bi:]

ntr_bispecs_bi = ntr_bispecs[:, :M_bi]
ntr_bispecs = ntr_bispecs[:, M_bi:]

ntr_bispec_rats_bi = ntr_bispec_rats[:, :M_bi]
ntr_bispec_rats = ntr_bispec_rats[:, M_bi:]

# Cross spectra
din_crosses_bi = din_crosses[:, :M_bi]
din_crosses = din_crosses[:, M_bi:]

ntr_crosses_bi = ntr_crosses[:, :M_bi]
ntr_crosses = ntr_crosses[:, M_bi:]


## PK
# din
din_pks_mean = din_pks.mean(axis=1)
din_pks_std = din_pks.std(axis=1)

din_pk_rats_mean = din_pk_rats.mean(axis=1)
din_pk_rats_std = din_pk_rats.std(axis=1)

# ntr
ntr_pks_mean = ntr_pks.mean(axis=1)
ntr_pks_std = ntr_pks.std(axis=1)

ntr_pk_rats_mean = ntr_pk_rats.mean(axis=1)
ntr_pk_rats_std = ntr_pk_rats.std(axis=1)

# BISPEC
# din
din_bispecs_mean = din_bispecs.mean(axis=1)
din_bispecs_std = din_bispecs.std(axis=1)

din_bispec_rats_mean = din_bispec_rats.mean(axis=1)
din_bispec_rats_std = din_bispec_rats.std(axis=1)

# n_tr
ntr_bispecs_mean = ntr_bispecs.mean(axis=1)
ntr_bispecs_std = ntr_bispecs.std(axis=1)

ntr_bispec_rats_mean = ntr_bispec_rats.mean(axis=1)
ntr_bispec_rats_std = ntr_bispec_rats.std(axis=1)


# CROSS
din_crosses_mean = din_crosses.mean(axis=1)
din_crosses_std = din_crosses.std(axis=1)

ntr_crosses_mean = ntr_crosses.mean(axis=1)
ntr_crosses_std = ntr_crosses.std(axis=1)


lw = 0.5 * 8
color_ref = 'tomato'#"k"
ls_ref = "--"
zorder_ref = 10

color_rec_mean = 'royalblue'#'r'
color_rec = 'cornflowerblue'#"lightcoral"
ls_rec = "-"
zorder_rec = 1

ref_label = r"Reference"
rec_label = r"Reconstruction"
extra_label = r"Propagator"

alpha_fill = 0.75


fig, axs = stats_plot_for_article(bk_rat_y_lim = (-6, 6))

axs[0].plot(k_c, din_ref_pk, c=color_ref, ls=ls_ref, lw=lw, zorder=zorder_ref, label=ref_label)
axs[0].plot(k_c, din_pks_mean, c=color_rec_mean, ls=ls_rec, lw=lw, zorder=zorder_rec+1, label=rec_label)
axs[0].fill_between(
    k_c,
    din_pks_mean - din_pks_std,
    din_pks_mean + din_pks_std,
    facecolor=color_rec,
    ls=ls_rec,
    lw=0.0,
    zorder=zorder_rec,
    alpha=alpha_fill
)

axs[2].plot(k_c, din_pk_rats_mean, c=color_rec_mean, ls=ls_rec, lw=lw, zorder=zorder_rec+1)
axs[2].fill_between(
    k_c,
    din_pk_rats_mean - din_pk_rats_std,
    din_pk_rats_mean + din_pk_rats_std,
    facecolor=color_rec,
    ls=ls_rec,
    lw=0.0,
    zorder=zorder_rec,
    alpha=alpha_fill
)

axs[1].plot(thetas, din_ref_bispec, c=color_ref, ls=ls_ref, lw=lw, zorder=zorder_ref)
axs[1].plot(thetas, din_bispecs_mean, c=color_rec_mean, ls=ls_rec, lw=lw, zorder=zorder_rec+1)
axs[1].fill_between(
    thetas,
    din_bispecs_mean - din_bispecs_std,
    din_bispecs_mean + din_bispecs_std,
    facecolor=color_rec,
    ls=ls_rec,
    lw=0.0,
    zorder=zorder_rec,
    alpha=alpha_fill,
)

axs[3].plot(thetas, din_bispec_rats_mean, c=color_rec_mean, ls=ls_rec, lw=lw, zorder=zorder_rec+1)
axs[3].fill_between(
    thetas,
    din_bispec_rats_mean - din_bispec_rats_std,
    din_bispec_rats_mean + din_bispec_rats_std,
    facecolor=color_rec,
    ls=ls_rec,
    lw=0.0,
    zorder=zorder_rec,
    alpha=alpha_fill
)

axs[4].plot(k_c, propagator, c=color_ref, ls='-.', lw=lw, zorder=zorder_rec, label=extra_label)
axs[4].plot(k_c, din_crosses_mean, c=color_rec_mean, ls=ls_rec, lw=lw, zorder=zorder_rec+1)
axs[4].fill_between(
    k_c,
    din_crosses_mean - din_crosses_std,
    din_crosses_mean + din_crosses_std,
    facecolor=color_rec,
    ls=ls_rec,
    lw=0.0,
    zorder=zorder_rec,
    alpha=alpha_fill,
)


axs[0].set_ylim(9e-4, 5)
axs[1].set_ylim(-13, 13)
axs[2].set_ylim(0.95, 1.05)
axs[3].set_ylim(-10, 10)
#axs[3].set_ylim(-6, 6)

fact_legend = 5
axs[0].legend(fontsize=lw*fact_legend, loc=('lower left'))
axs[4].legend(fontsize=lw*fact_legend, loc=(0.01, 0.1))

L = N*4
k_nyq = np.pi / (L/N)
for ax in [axs[0], axs[2], axs[4]]:
    ax.axvline(k_nyq, ls='dashed', color='grey', zorder=101, lw=lw*0.75)


figpath = os.path.join(this_file_dir, 'figures', 'din_stats_with_pk_2.png')
fig.savefig(figpath, bbox_inches='tight')

figpath = os.path.join(this_file_dir, 'figures', 'din_stats_with_pk_2.pdf')
fig.savefig(figpath, bbox_inches='tight')


############################################33
# color_bi = 'grey'
# zorder_bi = -1
# ls_bi = '-'
# alpha_bi = 0.2
# for i in range(M_bi):

#     din_pk_bi = din_pks_bi[:,i]
#     axs[0].plot(k_c, din_pk_bi, c=color_bi, ls=ls_bi, lw=lw, zorder=zorder_bi,  alpha=alpha_bi)

#     din_pk_rat_bi = din_pk_rats_bi[:,i]
#     axs[2].plot(k_c, din_pk_rat_bi, c=color_bi, ls=ls_bi, lw=lw, zorder=zorder_bi, alpha=alpha_bi)

#     din_bispec_bi = din_bispecs_bi[:,i]
#     axs[1].plot(thetas, din_bispec_bi, c=color_bi, ls=ls_bi, lw=lw, zorder=zorder_bi,  alpha=alpha_bi)

#     din_bispec_rat_bi = din_bispec_rats_bi[:,i]
#     axs[3].plot(thetas, din_bispec_rat_bi, c=color_bi, ls=ls_bi, lw=lw, zorder=zorder_bi,  alpha=alpha_bi)

#     din_cross_bi = din_crosses_bi[:,i]
#     axs[4].plot(k_c, din_cross_bi, c=color_bi, ls=ls_bi, lw=lw, zorder=zorder_bi,  alpha=alpha_bi)


# fact_legend = 5
# axs[0].legend(fontsize=lw*fact_legend, loc=('lower left'))
# axs[4].legend(fontsize=lw*fact_legend, loc=(0.01, 0.1))

# L = 1740.8
# k_nyq = np.pi / (L/N)
# for ax in [axs[0], axs[2], axs[4]]:
#     ax.axvline(k_nyq, ls='dashed', color='grey', zorder=101, lw=lw*0.75)


# figpath = os.path.join(this_file_dir, 'figures', 'din_stats_with_pk_2.png')
# fig.savefig(figpath, bbox_inches='tight')

# figpath = os.path.join(this_file_dir, 'figures', 'din_stats_with_pk_2.pdf')
# fig.savefig(figpath, bbox_inches='tight')