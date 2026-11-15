"""
Gnanadesikan (1999) Pycnocline Depth Model and Stability Curve.

This module is a backward-compatible interface to
:class:`gnanadesikan_model.GnanadesikanBoxModel`.  The original lecture code
used an eddy closure ``q_ed = c_ed D^2`` that did not reproduce Gnanadesikan's
stability curve.  The shared model now uses the corrected Work.nb balance

    qn = qt - epsilon D + mu / D,

with

    qt      = tau L_gap / (rho0 f)   [Sv]
    epsilon = A_I L_w / L_y          [Sv/m]
    mu      = K_v Area               [Sv m]

and the numerically stable pycnocline root.  The class name and plotting
method are retained so that existing notebooks continue to work.

References:
    Gnanadesikan, A. (1999). A simple predictive model for the structure of the
    oceanic pycnocline. Science, 283(5410), 2077-2079.
"""

import os
import sys
from typing import Tuple

import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gnanadesikan_model import GnanadesikanBoxModel

DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "figures",
    "gnanadesikan_stability.png",
)


class GnanadesikanPycnoclineModel(GnanadesikanBoxModel):
    """
    Gnanadesikan (1999) pycnocline model with the original class name.

    The transport balance is inherited from
    :class:`gnanadesikan_model.GnanadesikanBoxModel`:

        qn = q_Ek - q_ed + q_u
           = qt - epsilon D + mu / D

    The northern sinking is related to the density contrast by
    ``qn = eta Delta_rho D^2``, and the salinity balance supplies
    ``Delta_rho = alpha Theta - beta S0 F / qn``.  Eliminating
    ``Delta_rho`` gives the equilibrium freshwater flux

        F(qn) = (qn / (beta S0)) [alpha Theta - qn / (eta D(qn)^2)].
    """

    def compute_qn(self, D_m: np.ndarray) -> np.ndarray:
        """Northern sinking transport qn (Sv) for a pycnocline depth D (m)."""
        D_m = np.asarray(D_m, dtype=float)
        return self.qt - self.epsilon * D_m + self.mu / D_m

    def solve_stability_curve(
        self,
        D_range: Tuple[float, float] = (200.0, 2000.0),
        n_pts: int = 400,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(D, qn, F)`` along the equilibrium pycnocline balance.

        D is the pycnocline depth (m), qn the overturning transport (Sv), and
        F the atmospheric freshwater flux (Sv).  The curve is parameterized by
        D, which decreases monotonically along the stable branch as qn rises.
        """
        D = np.linspace(D_range[0], D_range[1], n_pts)
        qn = self.compute_qn(D)

        # Salinity closure qn = eta_sv * Delta_rho * D^2.
        Delta_rho = qn / (self.eta_sv * D ** 2)
        F = (qn / (self.beta * self.S0)) * (self.alpha * self.Theta - Delta_rho)
        return D, qn, F

    def plot_gnanadesikan_comparison(
        self,
        save_path: str = None,
    ):
        """
        Plot pycnocline depth D(qn) and the Gnanadesikan stability curve.

        The right-hand panel is the stability curve for the Gnanadesikan
        model: the upper branch is stable, the lower branch is unstable, and
        the red dot marks the saddle-node tipping point.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure, left open so that it can be displayed in a notebook.
        """
        if save_path is None:
            save_path = DEFAULT_OUT

        D, qn, F = self.solve_stability_curve()
        qn_c, F_c, D_c = self.tipping_point()

        valid = qn >= 0.0
        D = D[valid]
        qn = qn[valid]
        F = F[valid]

        stable = qn >= qn_c
        unstable = qn <= qn_c

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Pycnocline depth D vs qn (Slide 19).
        ax1.plot(qn, D, "b-", linewidth=2.5)
        ax1.plot(qn_c, D_c, "ro", markersize=8,
                 label=rf"Tipping point ($q_c = {qn_c:.1f}$ Sv)")
        ax1.set_xlabel("Northern Sinking Flow Rate $q_n$ (Sv)", fontsize=11)
        ax1.set_ylabel("Pycnocline Depth $D$ (m)", fontsize=11)
        ax1.set_title("Gnanadesikan (1999) Pycnocline Depth $D(q_n)$", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=9)

        # Panel 2: Stability curve for the Gnanadesikan model.
        ax2.plot(F[stable], qn[stable], "b-", linewidth=2.6,
                 label=r"Stable branch ($q_n > q_c$)")
        ax2.plot(F[unstable], qn[unstable], "r--", linewidth=2.2,
                 label=r"Unstable branch ($q_n < q_c$)")
        ax2.plot(F_c, qn_c, "ro", markersize=8,
                 label=(rf"Tipping point ($F_c = {F_c:.3f}$ Sv, "
                        rf"$q_c = {qn_c:.1f}$ Sv)"))
        ax2.axvline(F_c, color="gray", linestyle=":", alpha=0.7)
        ax2.axhline(0.0, color="k", linewidth=0.8)
        ax2.set_xlim(-3.6, 3.6)
        ax2.set_ylim(0.0, 42.0)
        ax2.set_xlabel("Atmospheric Freshwater Flux $F$ (Sv)", fontsize=11)
        ax2.set_ylabel("Overturning Strength $q_n$ (Sv)", fontsize=11)
        ax2.set_title("Stability Curve for the Gnanadesikan Model", fontsize=12)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved Gnanadesikan stability plot to: {save_path}")
        print(f"Tipping point: F = {F_c:.3f} Sv at qn = {qn_c:.2f} Sv "
              f"(D = {D_c:.0f} m)")
        return fig


if __name__ == "__main__":
    model = GnanadesikanPycnoclineModel()
    print(f"Southern Ocean Ekman transport qt: {model.qt:.2f} Sv")
    print(f"Eddy coefficient epsilon:          {model.epsilon:.5f} Sv/m")
    print(f"Diapycnal coefficient mu:          {model.mu:.1f} Sv m")
    model.plot_gnanadesikan_comparison()
