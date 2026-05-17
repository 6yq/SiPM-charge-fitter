import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from math import exp, factorial


# ==============================
#     Layout helpers
# ==============================


def n_max(lam, threshold=5e-03):
    """Largest n where Poisson p_n >= threshold * p_mode."""
    mode = max(0, int(lam))
    p_mode = exp(-lam) * lam**mode / factorial(mode)
    n = mode
    while True:
        n += 1
        p_n = exp(-lam) * lam**n / factorial(n)
        if p_n < threshold * p_mode:
            return n - 1


def npe_components(fit, n_max_val):
    """Per-bin count arrays and labels from 1PE up to n_max_val PE."""
    comps = [fit.estimate_count_n(n) for n in range(1, n_max_val + 1)]
    labels = [f"{n} PE" for n in range(1, n_max_val + 1)]
    return comps, labels


def make_figure(n_comps=4):
    """Three-panel figure: main / residual / PE-legend row."""
    n_leg_rows = (n_comps + 2) // 3
    leg_ratio = n_leg_rows * 0.06
    main_ratio, resid_ratio = 0.85, 0.15
    fig_height = 6 + n_leg_rows * 0.35

    fig = plt.figure(figsize=(plt.rcParams["figure.figsize"][0], fig_height))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[main_ratio, resid_ratio, leg_ratio],
    )
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    return fig, ax_main, ax_resid, ax_leg


def plot_channel(fit, hist, bins, ch_id, pp, logscale=False):
    """Render one channel's fit result and append to PdfPages pp."""
    bin_width = float(bins[1] - bins[0])
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n = n_max(fit.lam)
    comps, lbls = npe_components(fit, n)

    fig, ax_main, ax_resid, ax_leg = make_figure(n_comps=len(comps))

    xsp_plot = fit.xsp[abs(fit._shift) :]
    smooth_plot = fit.smooth[abs(fit._shift) :]

    spe_mean = float(fit.ser_args[0])
    spe_mean_std = float(fit.ser_args_std[0])
    spe_sigma = float(fit.ser_args[1])
    spe_sigma_std = float(fit.ser_args_std[1])
    spe_res = spe_sigma / spe_mean * 100.0
    spe_res_std = spe_res * np.sqrt(
        (spe_sigma_std / spe_sigma) ** 2 + (spe_mean_std / spe_mean) ** 2
    )

    plot_histogram_with_fit(
        bins=bins,
        hist=hist,
        xsp=xsp_plot,
        smooth=smooth_plot,
        bin_centers=bin_centers,
        comps=comps,
        labels=lbls,
        params=fit.ser_args,
        occ=1 - np.exp(-fit.lam),
        occ_std=fit.lam_std * np.exp(-fit.lam),
        ped_mean=fit.extra_args[0],
        gm=spe_mean,
        gm_std=spe_mean_std,
        spe_sigma=spe_sigma,
        spe_sigma_std=spe_sigma_std,
        spe_res=spe_res,
        spe_res_std=spe_res_std,
        chiSq=fit.chi_sq_neyman_B,
        ndf=fit.ndf,
        ys=fit.ys,
        logscale=logscale,
        ax_main=ax_main,
        ax_resid=ax_resid,
        ax_leg=ax_leg,
        fig=fig,
    )
    ax_main.set_title(f"channel id {ch_id}")

    if logscale:
        ax_main.set_ylim(bottom=5e-3)

    pp.savefig(fig)
    plt.close(fig)


# ==============================
#     Core drawing function
# ==============================


def plot_histogram_with_fit(
    bins,
    hist,
    xsp,
    smooth,
    comps,
    labels,
    params,
    bin_centers=None,
    ch=None,
    gp=None,
    gp_std=None,
    ped_mean=None,
    gm=None,
    gm_std=None,
    spe_sigma=None,
    spe_sigma_std=None,
    spe_res=None,
    spe_res_std=None,
    occ=None,
    occ_std=None,
    chiSq=None,
    ndf=None,
    zero=None,
    zeroReal=None,
    ys=None,
    logscale=False,
    ax_main=None,
    ax_resid=None,
    ax_leg=None,
    fig=None,
):
    if ax_main is None:
        fig, ax_main = plt.subplots()

    # bin_centers for histogram points and comps; xsp for smooth curve only
    if bin_centers is None:
        bin_centers = (bins[:-1] + bins[1:]) / 2

    # ==============================
    #     Occupancy label
    # ==============================

    if occ is not None and occ < 0.01:
        exp_ = int(np.floor(np.log10(occ)))
        mu = -math.log1p(-occ)
        mu_std = occ_std / (1.0 - occ)
        occ_str = (
            rf"({mu/10**exp_:.3f}\pm{mu_std/10**exp_:.3f})\times10^{{{exp_}}}"
        )
    elif occ is not None:
        mu = -math.log1p(-occ)
        mu_std = occ_std / (1.0 - occ)
        occ_str = rf"{mu:.3f}\pm{mu_std:.3f}"
    else:
        occ_str = ""

    # ==============================
    #     Histogram
    # ==============================

    ax_main.stairs(
        hist, bins, fill=True, color="gray", alpha=0.3, label=rf"$\mu={occ_str}$"
    )
    ax_main.errorbar(
        bin_centers,
        hist,
        yerr=np.sqrt(hist),
        marker="o",
        linestyle="none",
        ecolor="black",
        mec="black",
        mfc="red",
        ms=3,
    )

    # ==============================
    #     Overall fit curve (smooth on xsp grid)
    # ==============================

    ax_main.plot(
        xsp,
        smooth,
        color="red",
        label=(f"$\\chi^2/ndf={chiSq:.2f}/{ndf}$" if chiSq is not None else "fit"),
    )

    # ==============================
    #     Gain marker
    # ==============================

    if gm is not None:
        g_line = (
            f"$G={gm:.2f}\\pm{gm_std:.2f}$" if gm_std is not None else f"G={gm:.2f}"
        )
        sigma_line = (
            f"$\\sigma={spe_sigma:.2f}\\pm{spe_sigma_std:.2f}$"
            if spe_sigma is not None
            else ""
        )
        eta_line = (
            f"$\\eta={spe_res:.2f}\\pm{spe_res_std:.2f}\%$"
            if spe_res is not None
            else ""
        )
        gm_label = "\n".join(filter(None, [g_line, sigma_line, eta_line]))
        ax_main.axvline(
            gm + ped_mean, color="gray", linestyle="--", alpha=0.7, label=gm_label
        )

    # ==============================
    #     Per-PE components (on bin_centers grid)
    # ==============================

    comp_handles = []
    for i, (comp, lbl) in enumerate(zip(comps, labels)):
        (line,) = ax_main.plot(
            bin_centers,
            comp,
            linestyle="--",
        )
        comp_handles.append((line, lbl))

    # ==============================
    #     Axes formatting
    # ==============================

    if logscale:
        ax_main.set_yscale("log")

    ax_main.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_main.set_ylabel("Entries")
    if ch is not None:
        ax_main.set_title(f"channel id {ch}")

    fs_scale = 0.75
    legend_fontsize = (
        FontProperties(size=plt.rcParams["legend.fontsize"]).get_size_in_points()
        * fs_scale
    )
    ax_main.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        labelspacing=0.4,
        handlelength=2.2,
        borderaxespad=0.2,
        prop={"size": legend_fontsize},
    )

    if comp_handles:
        handles, labels_ = zip(*comp_handles)
        leg_target = ax_leg if ax_leg is not None else ax_main
        leg_target.legend(
            handles,
            labels_,
            ncol=min(3, len(handles)),
            frameon=False,
            loc="center",
            columnspacing=4.0,
            handlelength=2.0,
            handletextpad=0.5,
            borderaxespad=0.0,
            prop={"size": legend_fontsize},
        )

    # ==============================
    #     Residual panel
    # ==============================

    if ax_resid is not None and ys is not None:
        ax_resid.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax_resid.plot(bin_centers, hist - ys, "o", color="black", ms=3)
        ax_resid.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_resid.set_ylabel("Res.")
        ax_resid.set_xlabel("Q")
        ax_main.label_outer()
    else:
        ax_main.set_xlabel("Q")

    return fig, ax_main, ax_resid
