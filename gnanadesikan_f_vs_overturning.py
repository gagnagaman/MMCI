"""
Gnanadesikan (1999) Model: Stability Curve F vs. Overturning Strength qn.

Single-panel version of the fourth panel of ``gnanadesikan_model.py``.

The model is imported from ``gnanadesikan_model.py`` rather than duplicated,
so the two scripts cannot drift apart.  The curve is the full equilibrium
manifold: the lower (red dashed) branch is unstable and the upper (blue)
branch is stable; the red dot is the saddle-node tipping point where dF/dqn = 0.

This is the stability curve for the Gnanadesikan model: it shows how the
equilibrium freshwater flux depends on the overturning strength once the
pycnocline depth D(qn) is determined by the wind, eddy, and diapycnal
transports.

Reference:
    Gnanadesikan, A. (1999). A simple predictive model for the structure of the
    oceanic pycnocline. Science, 283(5410), 2077-2079.
"""

import os
import sys
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnanadesikan_model import GnanadesikanBoxModel

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "figures",
    "gnanadesikan_f_vs_overturning.png",
)


def plot_single_f_vs_overturning(save_path: str = None, C: float = 0.15):
    """
    Plot the equilibrium F(qn) curve for one efficiency parameter C.

    C = 0.15 corresponds to eta = 50, the standard Work.nb value.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, left open so that it can be displayed in a notebook.
    """
    if save_path is None:
        save_path = DEFAULT_OUT

    eta = 50.0 * C / 0.15
    model = GnanadesikanBoxModel(eta=eta)

    # Work.nb uses qn = 0 ... 40 Sv for this figure.
    qn = np.linspace(0.0, 40.0, 600)
    F, _ = model.freshwater_flux(qn)

    # Saddle-node point: maximum of F(qn), i.e. dF/dqn = 0.
    qn_crit, F_crit, D_crit = model.tipping_point()

    stable = qn >= qn_crit
    unstable = qn <= qn_crit

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(F[stable], qn[stable], "b-", linewidth=2.8,
            label=r"Stable branch ($q_n > q_c$)")
    ax.plot(F[unstable], qn[unstable], "r--", linewidth=2.2,
            label=r"Unstable branch ($q_n < q_c$)")
    ax.plot(F_crit, qn_crit, "ro", markersize=9,
            label=(rf"Tipping point ($F_c = {F_crit:.3f}$ Sv, "
                   rf"$q_c = {qn_crit:.1f}$ Sv)"))
    ax.axvline(F_crit, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0.0, color="k", linewidth=0.8)

    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(0.0, 42.0)
    ax.set_xlabel(
        r"Atmospheric Freshwater Flux from Tropics to High Latitudes $F$ (Sv)",
        fontsize=11,
    )
    ax.set_ylabel(r"Overturning Circulation Strength $q_n$ (Sv)", fontsize=11)
    ax.set_title(
        "Gnanadesikan (1999) Model: $F$ vs. Overturning Strength $q_n$\n"
        rf"Equilibrium Stability Curve ($C = {C:.2f}$, $\eta = {eta:.1f}$)",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(save_path, dpi=200)

    print(f"Saved single F vs Overturning plot to: {save_path}")
    print(f"Tipping point: F_crit = {F_crit:.4f} Sv at qn = {qn_crit:.2f} Sv")
    print(f"Pycnocline depth at tipping point: {D_crit:.0f} m")

    return fig


if __name__ == "__main__":
    plot_single_f_vs_overturning()
