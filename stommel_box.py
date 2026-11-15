"""
Stommel (1961) Two-Box Model of Thermohaline Circulation.

Reference:
    Stommel, H. (1961). Thermohaline convection with two stable regimes of flow.
    Tellus, 13(2), 224-230.
"""

from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


class StommelBoxModel:
    """
    Two-box model of thermohaline convection.
    
    Equations:
        dT1/dt = -|q| (T1 - T2) / V1 - gamma_T (T1 - T_E)
        dT2/dt = -|q| (T2 - T1) / V2 - gamma_T (T2 - T_P)
        dS1/dt = -|q| (S1 - S2) / V1 + F * S0 / V1
        dS2/dt = -|q| (S2 - S1) / V2 - F * S0 / V2
        
    At steady state with fixed temperature contrast Delta_T = T1 - T2:
        Delta_S = F * S0 / q
        q = k * Delta_rho = k * (alpha * Delta_T - beta * Delta_S)
        q = k * alpha * Delta_T - k * beta * S0 * F / q
        
    Quadratic equation in q:
        q^2 - sigma * q + gamma * F = 0
        where sigma = k * alpha * Delta_T, gamma = k * beta * S0.
    """

    def __init__(
        self,
        alpha: float = 0.15,      # Thermal expansion (kg/m^3/K)
        beta: float = 0.80,       # Haline contraction (kg/m^3/psu)
        S0: float = 35.0,         # Reference salinity (psu)
        Delta_T: float = 20.0,    # Equator-to-pole temperature contrast (K)
        k: float = 1.0,           # Hydraulic conductivity (Sv / (kg/m^3))
    ):
        self.alpha = alpha
        self.beta = beta
        self.S0 = S0
        self.Delta_T = Delta_T
        self.k = k

        # Quadratic parameters:
        # q^2 - sigma * q + gamma * F = 0
        self.sigma = self.k * self.alpha * self.Delta_T  # Maximum thermal driving (Sv)
        self.gamma = self.k * self.beta * self.S0        # Haline braking coefficient (Sv / (Sv_fresh))

    @property
    def critical_flux(self) -> float:
        """Saddle-node bifurcation threshold Fc (critical freshwater flux)."""
        return (self.sigma ** 2) / (4.0 * self.gamma)

    @property
    def critical_flow(self) -> float:
        """AMOC strength qc at the tipping point."""
        return self.sigma / 2.0

    def steady_state_branches(self, F_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the upper (stable) and lower (unstable) forward flow branches.
        
        q = [sigma +/- sqrt(sigma^2 - 4 * gamma * F)] / 2
        """
        disc = (self.sigma ** 2) - 4.0 * self.gamma * F_array
        
        # Real roots only exist where discriminant >= 0
        valid = disc >= 0.0
        q_upper = np.full_like(F_array, np.nan, dtype=float)
        q_lower = np.full_like(F_array, np.nan, dtype=float)

        q_upper[valid] = 0.5 * (self.sigma + np.sqrt(disc[valid]))
        q_lower[valid] = 0.5 * (self.sigma - np.sqrt(disc[valid]))

        return q_upper, q_lower

    def plot_stability_curve(self, save_path: str = None) -> plt.Figure:
        """Plot the classic Stommel stability curve (q vs F)."""
        Fc = self.critical_flux
        F = np.linspace(-0.6 * Fc, 1.02 * Fc, 500)
        q_up, q_down = self.steady_state_branches(F)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(F, q_up, "b-", linewidth=2.5, label="Stable branch ($q > q_c$)")
        ax.plot(F, q_down, "r--", linewidth=2.0, label="Unstable branch ($q < q_c$)")

        # Mark tipping point
        ax.plot(Fc, self.critical_flow, "ro", markersize=8, label=f"Tipping Point ($F_c = {Fc:.3f}$ Sv)")
        ax.axvline(Fc, color="gray", linestyle=":", alpha=0.7)
        ax.axhline(0, color="k", linewidth=0.8)

        ax.set_xlabel("Atmospheric Freshwater Flux $F$ (Sv)", fontsize=11)
        ax.set_ylabel("Overturning Circulation $q$ (Sv)", fontsize=11)
        ax.set_title("Stommel (1961) Stability Curve & Advective Instability", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=10)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200)
            print(f"Saved Stommel stability plot to: {save_path}")

        return fig


if __name__ == "__main__":
    model = StommelBoxModel(Delta_T=20.0, k=1.0)
    print(f"Sigma (thermal driving): {model.sigma:.2f} Sv")
    print(f"Critical freshwater flux Fc: {model.critical_flux:.4f} Sv")
    print(f"Critical overturning qc: {model.critical_flow:.2f} Sv")
    model.plot_stability_curve("stommel_stability.png")
