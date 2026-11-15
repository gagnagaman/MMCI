"""
Gnanadesikan (1999) Model: Pycnocline Depth and Equilibrium Stability.

Python translation of Section 5 of Mathematica notebook ``Work.nb``:
    "The Gnanadesikan Model"

Units
-----
All transports are in Sv (1 Sv = 1e6 m^3/s), depths in m, and density
differences in kg/m^3.  Work.nb writes the pycnocline balance as

    epsilon * D^2 + (qn - qt) * D - mu = 0

with the *Sv-scaled* coefficients

    epsilon = A_I * L_w / L_y / 1e6     [Sv/m]
    mu      = K_v * A / 1e6             [Sv m]
    qt      = tau * L_gap / (rho0 f) / 1e6   [Sv]

The positive root is evaluated in the numerically stable form

    D = 2 mu / [ sqrt((qn - qt)^2 + 4 epsilon mu) + (qn - qt) ],

which avoids the catastrophic cancellation of the textbook quadratic
formula when qn > qt.

At steady state the salinity balance gives

    qn = eta_sv * Delta_rho * D^2,
    Delta_rho = alpha * Theta - beta * S0 * F / qn,

so the equilibrium freshwater flux is

    F(qn) = (qn / (beta S0)) * [ alpha Theta - qn / (eta_sv D^2) ].

Reference:
    Gnanadesikan, A. (1999). A simple predictive model for the structure of
    the oceanic pycnocline. Science, 283(5410), 2077-2079.
"""

import sys
from typing import Tuple
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GnanadesikanBoxModel:
    """Gnanadesikan (1999) coupled pycnocline box model."""

    def __init__(
        self,
        tau: float = 1.5,           # Wind stress in Drake Passage (Pa = N/m^2)
        f_coriolis: float = 1.0e-4,  # Coriolis parameter (s^-1)
        rho0: float = 1025.0,       # Reference density (kg/m^3)
        L_gap: float = 2.46e6,      # Zonal Ekman fetch in Drake Passage (m)
        L_w: float = 25.0e6,        # Circumpolar width in eddy closure (m)
        L_y: float = 1.5e6,         # Meridional scale of the ACC (m)
        AI: float = 1000.0,         # Eddy thickness diffusivity (m^2/s)
        Kv: float = 0.15e-4,        # Diapycnal diffusivity (m^2/s)
        Area: float = 2.44e14,      # Area of equatorial thermocline (m^2)
        alpha: float = 0.2,         # Thermal expansion (kg/m^3/K)
        beta: float = 0.8,          # Haline contraction (kg/m^3/psu)
        Theta: float = 20.0,        # Thermal contrast (K)
        eta: float = 50.0,          # Efficiency (m^4 s^-1 kg^-1)
        S0: float = 35.0,           # Reference salinity (psu)
    ):
        self.tau = tau
        self.f_coriolis = f_coriolis
        self.rho0 = rho0
        self.L_gap = L_gap
        self.L_w = L_w
        self.L_y = L_y
        self.AI = AI
        self.Kv = Kv
        self.Area = Area
        self.alpha = alpha
        self.beta = beta
        self.Theta = Theta
        self.eta = eta
        self.S0 = S0

        # Sv-scaled coefficients of Work.nb (see module docstring).
        self.qt = tau * L_gap / (rho0 * f_coriolis) * 1e-6       # Sv
        self.epsilon = AI * L_w / L_y * 1e-6                     # Sv/m
        self.mu = Kv * Area * 1e-6                               # Sv m
        self.eta_sv = eta * 1e-6                                 # Sv m^2/kg

    # ------------------------------------------------------------------
    # Pycnocline geometry
    # ------------------------------------------------------------------
    def pycnocline_depth(self, qn):
        """
        Positive root of  epsilon D^2 + (qn - qt) D - mu = 0.

        Parameters
        ----------
        qn : float or array_like
            Northern overturning transport in Sv.

        Returns
        -------
        D : float or ndarray
            Pycnocline depth in m.
        """
        qn = np.asarray(qn, dtype=float)
        disc = (qn - self.qt) ** 2 + 4.0 * self.epsilon * self.mu
        return 2.0 * self.mu / (np.sqrt(disc) + (qn - self.qt))

    def pycnocline_depth_derivative(self, qn):
        """
        dD/dqn obtained by implicit differentiation of the quadratic:

            (2 epsilon D + qn - qt) dD/dqn + D = 0.
        """
        qn = np.asarray(qn, dtype=float)
        D = self.pycnocline_depth(qn)
        return -D / (2.0 * self.epsilon * D + qn - self.qt)

    def freshwater_flux(self, qn, eta=None):
        """
        Equilibrium freshwater flux F(qn) and pycnocline depth D(qn).

        F is positive for a net freshwater flux from the tropics to the
        northern high latitudes (the Stommel/Rahmstorf convention).
        """
        qn = np.asarray(qn, dtype=float)
        D = self.pycnocline_depth(qn)
        eta_sv = self.eta_sv if eta is None else eta * 1e-6

        Delta_rho = qn / (eta_sv * D ** 2)
        F = (qn / (self.beta * self.S0)) * (self.alpha * self.Theta - Delta_rho)
        return F, D

    def tipping_point(self, eta=None, qn_max: float = 80.0, n: int = 40001) -> Tuple[float, float, float]:
        """
        Saddle-node tipping point: the maximum of F(qn).

        Returns
        -------
        qn_c, F_c, D_c : floats
            Overturning strength (Sv), freshwater flux (Sv), and pycnocline
            depth (m) at the turning point.
        """
        qn = np.linspace(0.0, qn_max, n)
        F, D = self.freshwater_flux(qn, eta=eta)
        i = int(np.argmax(F))
        return float(qn[i]), float(F[i]), float(D[i])

    # ------------------------------------------------------------------
    # Implicit cubic / Jacobian routines retained from Work.nb
    # ------------------------------------------------------------------
    def solve_pycnocline_depth(self, Delta_rho: float) -> float:
        """
        Solve the full cubic for D at a prescribed density difference:

            eta_sv * Delta_rho * D^3 + epsilon D^2 - qt D - mu = 0.
        """
        if Delta_rho <= 1e-12:
            return float(self.pycnocline_depth(0.0))

        coeffs = [
            self.eta_sv * Delta_rho,
            self.epsilon,
            -self.qt,
            -self.mu,
        ]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-9].real
        positive = real_roots[real_roots > 0.0]
        if positive.size == 0:
            return float(self.pycnocline_depth(0.0))
        return float(np.min(positive))

    def pycnocline_derivatives(self, D: float, Delta_rho: float) -> Tuple[float, float]:
        """
        Exact implicit derivatives from Work.nb:

            dD/dS2 = - beta eta D^3 /
                     (-qt + 2 epsilon D + 3 eta Delta_rho D^2)

        and dD/dS3 = -dD/dS2.
        """
        denom = -self.qt + 2.0 * self.epsilon * D + 3.0 * self.eta_sv * Delta_rho * D ** 2
        if abs(denom) < 1e-30:
            denom = 1e-30
        dD_dS2 = -(self.beta * self.eta_sv * D ** 3) / denom
        return dD_dS2, -dD_dS2

    def jacobian(self, qn: float, D: float, Delta_rho: float,
                 F1: float, F2: float) -> np.ndarray:
        """
        Coupled 3-box Jacobian including pycnocline-depth derivatives.

        The salinity equations are those of the northern (S2) and equatorial
        (S3) boxes; S1 is the southern box.
        """
        dD_dS2, dD_dS3 = self.pycnocline_derivatives(D, Delta_rho)

        dqn_dS2 = (2.0 * self.eta_sv * D * Delta_rho * dD_dS2
                   + self.eta_sv * D ** 2 * self.beta)
        dqn_dS3 = (2.0 * self.eta_sv * D * Delta_rho * dD_dS3
                   - self.eta_sv * D ** 2 * self.beta)

        M = np.zeros((3, 3), dtype=float)

        diff_2 = F2 * self.S0 / max(qn, 0.1)
        diff_3 = -(F1 + F2) * self.S0 / max(qn, 0.1)

        M[0, 0] = -qn
        M[0, 2] = +qn

        M[1, 1] = -qn + dqn_dS2 * diff_2
        M[1, 2] = +qn + dqn_dS3 * diff_2

        M[2, 0] = +qn
        M[2, 1] = dqn_dS2 * diff_3
        M[2, 2] = -qn + dqn_dS3 * diff_3

        return M

    # ------------------------------------------------------------------
    # Sensitivity suite
    # ------------------------------------------------------------------
    def plot_sensitivity_suite(self, save_path: str = "gnanadesikan_sensitivity_suite.png"):
        """
        Four-panel sensitivity study in the spirit of Section 5 of Work.nb:

        1. D(qn) for several wind stresses tau.
        2. D(qn) for several diapycnal diffusivities Kv.
        3. D(qn) for several eddy diffusivities AI.
        4. F(qn) equilibrium curves for several efficiency parameters C.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        qn_depth = np.linspace(0.0, 50.0, 400)

        # 1. Wind stress in Drake Passage
        ax1 = axes[0, 0]
        for tau_val in (0.5, 1.0, 1.5, 2.0):
            m = GnanadesikanBoxModel(tau=tau_val)
            ax1.plot(qn_depth, m.pycnocline_depth(qn_depth),
                     linewidth=2.0, label=rf"$\tau = {tau_val:.1f}$ Pa")
        ax1.set_xlabel("Overturning Transport $q_n$ (Sv)", fontsize=10)
        ax1.set_ylabel("Pycnocline Depth $D$ (m)", fontsize=10)
        ax1.set_title(r"Sensitivity to Wind Stress ($\tau$)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right", fontsize=9)

        # 2. Diapycnal diffusivity
        ax2 = axes[0, 1]
        for kv_val in (0.5e-5, 1.0e-5, 2.0e-5, 4.0e-5):
            m = GnanadesikanBoxModel(Kv=kv_val)
            ax2.plot(qn_depth, m.pycnocline_depth(qn_depth),
                     linewidth=2.0,
                     label=rf"$K_v = {kv_val * 1e4:.2f}$ cm$^2$/s")
        ax2.set_xlabel("Overturning Transport $q_n$ (Sv)", fontsize=10)
        ax2.set_ylabel("Pycnocline Depth $D$ (m)", fontsize=10)
        ax2.set_title(r"Sensitivity to Diapycnal Diffusivity ($K_v$)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="upper right", fontsize=9)

        # 3. Mesoscale eddy diffusivity
        ax3 = axes[1, 0]
        for ai_val in (500.0, 1000.0, 1500.0, 2000.0):
            m = GnanadesikanBoxModel(AI=ai_val)
            ax3.plot(qn_depth, m.pycnocline_depth(qn_depth),
                     linewidth=2.0,
                     label=rf"$A_I = {ai_val:.0f}$ m$^2$/s")
        ax3.set_xlabel("Overturning Transport $q_n$ (Sv)", fontsize=10)
        ax3.set_ylabel("Pycnocline Depth $D$ (m)", fontsize=10)
        ax3.set_title(r"Sensitivity to Eddy Diffusivity ($A_I$)", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right", fontsize=9)

        # 4. Efficiency parameter C (eta = 50 at C = 0.15 in Work.nb)
        ax4 = axes[1, 1]
        qn_flux = np.linspace(0.0, 40.0, 600)
        for C in (0.05, 0.10, 0.15, 0.20, 0.25):
            eta_val = 50.0 * C / 0.15
            m = GnanadesikanBoxModel(eta=eta_val)
            F, _ = m.freshwater_flux(qn_flux)
            ax4.plot(F, qn_flux, linewidth=2.0,
                     label=rf"$C = {C:.2f}$  ($\eta = {eta_val:.1f}$)")
        ax4.set_xlim(-3.6, 3.6)
        ax4.set_ylim(0.0, 42.0)
        ax4.axhline(0.0, color="k", linewidth=0.8)
        ax4.set_xlabel("Freshwater Flux $F$ (Sv)", fontsize=10)
        ax4.set_ylabel("Overturning Transport $q_n$ (Sv)", fontsize=10)
        ax4.set_title(r"Equilibrium Curves vs Efficiency ($C$)", fontsize=11)
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="lower right", fontsize=8)

        plt.suptitle("Gnanadesikan (1999) Pycnocline Model (Work.nb)",
                     fontsize=13, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved Gnanadesikan sensitivity suite to: {save_path}")


if __name__ == "__main__":
    m = GnanadesikanBoxModel()
    qn_c, F_c, D_c = m.tipping_point()
    print(f"Ekman transport qt:              {m.qt:.2f} Sv")
    print(f"Eddy coefficient epsilon:        {m.epsilon:.5f} Sv/m")
    print(f"Diapycnal coefficient mu:        {m.mu:.1f} Sv m")
    print(f"Tipping point: F = {F_c:.3f} Sv at qn = {qn_c:.2f} Sv "
          f"(D = {D_c:.0f} m)")
    m.plot_sensitivity_suite()
