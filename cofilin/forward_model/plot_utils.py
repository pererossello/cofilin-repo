import sys

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from .fourier import get_k
from .stats import (
    get_pow_spec_1D,
    get_cross_power_spec_1D,
    get_radial_distro,
)

def plot_cubes(
    list_of_arrays,
    cmap="seismic_r",
    figsize=5,
    axis=0,
    idx=0,
    width=1,
    vlim=None,
    wspace=0.05,
    return_ims=False,
    titles=None,
    title=None,
    ts=1.25,
    interpolation="nearest",
):

    if not isinstance(list_of_arrays, list):
        list_of_arrays = [list_of_arrays]
    M = len(list_of_arrays)

    if not isinstance(cmap, list):
        cmap = [cmap] * M

    if vlim is not None:
        if isinstance(vlim, (float, int)):
            vlim = [(-vlim, vlim)] * M
        elif isinstance(vlim, list):
            for i, vlim_item in enumerate(vlim):
                if isinstance(vlim_item, (int, float)):
                    vlim[i] = (-vlim_item, vlim_item)
        elif isinstance(vlim, tuple):
            vlim = [(vlim[0], vlim[1])] * M

    if M < 4:
        n_col = M
        n_row = 1
    elif M == 4:
        n_col = 2
        n_row = 2

    ratio = n_col / n_row
    figsize_fact = figsize / np.sqrt(ratio)
    fig, axs = plt.subplots(n_row, n_col, figsize=(figsize_fact * ratio, figsize_fact))
    plt.subplots_adjust(wspace=wspace, hspace=0.05)

    ims = []
    axs_flat = axs.flatten() if M > 1 else [axs]
    for i, ax in enumerate(axs_flat):
        ax.set_xticks([])
        ax.set_yticks([])

        arr = list_of_arrays[i]
        arr, min_val, max_val = get_projection(arr, axis, idx, width)

        vmin = min_val if vlim is None else vlim[i][0]
        vmax = max_val if vlim is None else vlim[i][1]

        im = ax.imshow(
            arr.T,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            cmap=cmap[i],
            interpolation=interpolation,
        )
        ims.append(im)

        if titles is not None:
            ax.text(
                0.04,
                0.04,
                titles[i],
                ha="left",
                va="bottom",
                transform=ax.transAxes,
                fontsize=figsize * ts,
                c="k",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.25",
                    alpha=1,
                ),
            )
        if title is not None:
            fig.suptitle(title, fontsize=figsize * ts, y=0.92)

    if return_ims:
        return fig, axs, ims

    return fig, axs

def get_projection(array, axis, idx, width):

    N = array.shape[0]

    if idx > N - 1:
        raise ValueError("idx should be smaller than N-1")

    idx_min = idx
    idx_max = idx + width

    if axis == 0:
        matrix = array[idx_min:idx_max, :, :]
    elif axis == 1:
        matrix = array[:, idx_min:idx_max, :]
    elif axis == 2:
        matrix = array[:, :, idx_min:idx_max]

    matrix = np.sum(matrix, axis=axis) / width
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    return matrix, min_val, max_val

def compare_pow_spec(
    delta_list,
    L,
    n_bins=50,
    labels=None,
    title=None,
    xlog=False,
    no_labels=False,
    sphere_only=False,
    lw=0.5,
    x_lim=None,
    cross=True,
    alpha=0.75,
    y2lim=None,
    ls=None,
    colors=None,
    pk_cube=None,
    k_min=None,
):

    # grid_sizes = [delta.shape[0] for delta in delta_list]
    # deltas_same_gridsize = all([delta.shape==delta_list[0].shape for delta in delta_list])

    M = len(delta_list)

    if pk_cube is not None:
        N = delta_list[0].shape[0]
        print("pk_cube is not None, forcing no cross")
        cross = False
        M += 1

        k = get_k(N, L)
        fact = 1 / jnp.sqrt(3) if sphere_only else 1.0
        bin_range = (0.0, k.max() * fact)
        k, pk_ref = get_radial_distro(k, pk_cube, bin_range=bin_range, n_bins=n_bins)

    else:
        k, pk0 = get_pow_spec_1D(
            delta_list[0], L, n_bins=n_bins, sphere_only=sphere_only
        )

    if labels == None:
        labels = [f"{i}" for i in range(M)]
    if ls == None:
        ls = ["-."] + ["-"] * (M - 1)
    if colors == None:
        cmap = plt.get_cmap("nipy_spectral")
        cs = [cmap(i) for i in np.linspace(0.1, 0.9, M - 1, endpoint=True)]
        cs = ["k"] + cs
    else:
        cs = colors
        if pk_cube is not None:
            cs = ["k"] + cs

    figsize = 5
    lw = lw * figsize

    labelsize = 11

    if cross:
        fig, axs = plt.subplots(1, 3, figsize=(3 * figsize, figsize))
        plt.subplots_adjust(wspace=0.15, hspace=None)
    else:
        fig, axs = plt.subplots(1, 2, figsize=(2 * figsize, figsize))
        plt.subplots_adjust(wspace=0.15, hspace=None)

    if pk_cube is not None:
        pk0 = pk_ref

    for i, delta in enumerate(delta_list):
        if pk_cube is not None:
            i += 1
        k, pk = get_pow_spec_1D(delta, L, n_bins=n_bins, sphere_only=sphere_only)

        axs[0].plot(k, pk, c=cs[i], ls=ls[i], label=labels[i], lw=lw, alpha=alpha)
        axs[1].plot(k, pk / pk0, c=cs[i], ls=ls[i], lw=lw, alpha=alpha)

        if cross:
            k, pk_cross = get_cross_power_spec_1D(
                delta_list[0], delta_list[i], L, n_bins, sphere_only=sphere_only
            )
            axs[2].plot(k, pk_cross, c=cs[i], ls=ls[i], lw=lw, alpha=alpha)

    if pk_cube is not None:
        axs[0].plot(
            k,
            pk_ref,
            c=cs[0],
            ls=ls[0],
            label=labels[0],
            lw=lw,
            alpha=alpha,
            zorder=100,
        )

    axs[0].set_yscale("log")

    if not sphere_only:
        for ax in axs:
            ax.axvline(k.max() / jnp.sqrt(3), color="grey", ls="--")

    if x_lim is not None:
        axs[0].set_xlim(x_lim[0], x_lim[1])
    # axs[0].set_xscale('log')

    if not no_labels:
        axs[0].legend(fontsize=labelsize)

    axs[0].tick_params(labelsize=labelsize, top=True)

    # axs[1].set_ylim(0.5, 2)
    axs[1].tick_params(
        labelsize=labelsize,
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        top=True,
    )

    if cross:
        axs[2].set_ylim(-0.2, 1.1)
        axs[2].tick_params(
            labelsize=labelsize,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
            top=True,
        )

    for ax in axs:

        ax.grid(
            which="major",
            linewidth=0.5,
            color="k",
            alpha=0.25,
        )
        ax.set_xlabel("$k$ [$h$/Mpc]", fontsize=labelsize)

        if xlog:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, None)
            xticks = ax.get_xticks()
            ax2 = ax.twiny()
            ax2.set_xlim(axs[0].get_xlim())  # Ensure the limits of top and
            ax2.xaxis.set_major_locator(FixedLocator(xticks))
            ax2.set_xticklabels(
                [""] + [f"{2*np.pi/label:0.2f}" for label in xticks[1:]]
            )
            ax2.set_xlabel(r"$\lambda$ [Mpc/$h$]", fontsize=labelsize, labelpad=7)
            ax2.tick_params(labelsize=labelsize)

    if y2lim is not None:
        axs[1].set_ylim(y2lim[0], y2lim[1])

    if xlog:
        y = 0.95
    else:
        y = 1.05
    if title is not None:
        fig.suptitle(title, y=y, fontsize=labelsize * 1.3)

    return fig, axs

