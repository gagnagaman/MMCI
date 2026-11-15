"""
Stommel-Arons (1960) Abyssal Circulation Model on an Equatorial Beta-Plane.

References:
    Stommel, H., & Arons, A. B. (1960). On the abyssal circulation of the world ocean-I.
    Stationary planetary flow patterns on a sphere. Deep Sea Research, 6, 140-154.
    Straub, D. (1996). An inconsistency between two classical models of the ocean
    buoyancy-driven circulation. Tellus A, 48(3), 477-481.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


class StommelAronsModel:
    """
    Two-layer Stommel-Arons model on an equatorial beta-plane (f = beta * y).
    
    Parameters:
        beta: Rossby beta parameter (m^-1 s^-1)
        H: Total basin depth (m)
        g_prime: Reduced gravity g * Delta_rho / rho0 (m s^-2)
        h0: Interface height at eastern boundary x = Xe (m)
        Lx: Basin width (m)
        Ly: Basin length from -Ly/2 to +Ly/2 (m)
    """

    def __init__(
        self,
        beta: float = 2.0e-11,
        H: float = 4000.0,
        g_prime: float = 5.0e-3,
        h0: float = 2000.0,
        Lx: float = 4.0e6,
        Ly: float = 10.0e6,
    ):
        self.beta = beta
        self.H = H
        self.g_prime = g_prime
        self.h0 = h0
        self.Lx = Lx
        self.Ly = Ly

    def solve_interface_height_cubic(
        self,
        x_dist_from_east: np.ndarray,
        y_grid: np.ndarray,
        e_upwelling: np.ndarray
    ) -> np.ndarray:
        """
        Solve the cubic polynomial for interior interface height eta_i:
            -g'/3 * eta^3 + g'*H/2 * eta^2 + c4 = 0
        where
            c4 = (h0^3)*g'/3 - (h0^2)*g'*H/2 - e*beta*H * (X - Xe) * y^2

        Using the analytical trigonometric cubic solution (Cardano's method),
        which evaluates instantaneously in vectorized form.
        """
        c4_base = (self.h0**3) * self.g_prime / 3.0 - (self.h0**2) * self.g_prime * self.H / 2.0
        forcing_term = e_upwelling * self.beta * self.H * x_dist_from_east * (y_grid**2)
        c4 = c4_base - forcing_term

        # Standard form: eta^3 + a1*eta^2 + a2*eta + a3 = 0
        a1 = -1.5 * self.H
        a3 = -(3.0 / self.g_prime) * c4

        Q = -(self.H**2) / 4.0
        R = -0.5 * a3 - (a1**3) / 27.0

        # Discriminant is negative (three real roots)
        sqrt_neg_Q3 = np.sqrt(-Q**3)
        arg = np.clip(R / sqrt_neg_Q3, -1.0, 1.0)
        theta = np.arccos(arg)

        # The physical root in the range [0, H] corresponds to branch (theta + 4*pi)/3:
        eta = 2.0 * np.sqrt(-Q) * np.cos((theta + 4.0 * np.pi) / 3.0) - a1 / 3.0
        return eta

    def compute_fields(
        self,
        nx: int = 50,
        ny: int = 100,
        e_mean: float = 6.7e-7,
        so_enhanced: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Compute the 2D grid, upwelling field e(x, y), interface height eta(x, y),
        and geostrophic flow (u2, v2).
        """
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(-self.Ly / 2.0, self.Ly / 2.0, ny)
        X, Y = np.meshgrid(x, y)

        # Distance from eastern boundary (negative, X - Xe)
        XE = X - self.Lx

        # Vertical upwelling field e(x, y)
        E = np.full_like(X, e_mean)
        if so_enhanced:
            # Enhanced upwelling in the Southern Ocean (simulating Drake Passage winds)
            E = E / 1.5
            E[:10, :] *= 2.5
            E[10:20, :] *= 2.0

        # For non-uniform e, integrate e westward from eastern boundary
        # integral_e = - sum_{x}^{Xe} e dx
        dx = self.Lx / (nx - 1)
        integrated_e = np.zeros_like(X)
        for i in range(ny):
            integrated_e[i, :] = -np.cumsum(E[i, ::-1])[::-1] * dx

        eta = self.solve_interface_height_cubic(integrated_e, Y, 1.0) if so_enhanced \
              else self.solve_interface_height_cubic(XE, Y, E)

        # Geostrophic velocities in lower layer:
        # u2 = -g' / (H * beta * y) * d(eta)/dy
        # v2 =  g' / (H * beta * y) * d(eta)/dx
        dy = y[1] - y[0]
        deta_dy, deta_dx = np.gradient(eta, dy, dx)

        # Avoid divide-by-zero at the equator y = 0
        f_coriolis = self.beta * Y
        safe_f = np.where(np.abs(f_coriolis) < 1e-12, np.nan, f_coriolis)

        u2 = -(self.g_prime / (self.H * safe_f)) * deta_dy
        v2 = (self.g_prime / (self.H * safe_f)) * deta_dx

        # Zonal average interface height
        eta_zonal = np.nanmean(eta, axis=1)

        # Total upwelling / overturning transport (Sv)
        total_q_Sv = np.sum(E) * dx * dy * 1e-6

        return {
            "x": x, "y": y, "X": X, "Y": Y,
            "E": E, "eta": eta,
            "u2": u2, "v2": v2,
            "eta_zonal": eta_zonal,
            "total_q_Sv": total_q_Sv
        }

    def plot_solution(self, fields: dict, title: str = "Stommel-Arons Abyssal Flow", save_path: str = None):
        """Plot the interface height contours and velocity vectors."""
        X_km = fields["X"] * 1e-3
        Y_km = fields["Y"] * 1e-3
        eta = fields["eta"]
        u2 = fields["u2"]
        v2 = fields["v2"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.5, 1]})

        # Panel 1: 2D interface height and velocity vectors
        levs = np.linspace(np.nanmin(eta), np.nanmax(eta), 21)
        cf = ax1.contourf(X_km, Y_km, eta, levels=levs, cmap="viridis")
        cbar = fig.colorbar(cf, ax=ax1)
        cbar.set_label("Interface Height $\\eta$ (m)", fontsize=11)

        cs = ax1.contour(X_km, Y_km, eta, levels=levs[::2], colors="k", linewidths=0.6, alpha=0.6)
        ax1.clabel(cs, inline=True, fontsize=8, fmt="%d")

        # Overlay velocity vectors (subsampled for clarity)
        skip = (slice(None, None, 5), slice(None, None, 3))
        ax1.quiver(X_km[skip], Y_km[skip], u2[skip], v2[skip],
                   color="white", scale=1e-1, width=0.004, headwidth=4)

        # Indicate Western Boundary Current
        ax1.annotate("", xy=(0, 4500), xytext=(0, -4500),
                    arrowprops=dict(facecolor="red", edgecolor="black", width=3, headwidth=10))
        ax1.text(100, 0, "Deep Western Boundary Current (DWBC)", color="red",
                 fontsize=10, rotation=90, va="center", weight="bold")

        ax1.set_xlabel("Zonal Distance (km)", fontsize=11)
        ax1.set_ylabel("Meridional Distance from Equator (km)", fontsize=11)
        ax1.set_title(title, fontsize=12)

        # Panel 2: Zonally averaged interface height (Straub's paradox)
        ax2.plot(fields["eta_zonal"], fields["y"] * 1e-3, "b-", linewidth=2.5)
        ax2.set_xlabel("Zonally Averaged $\\bar{\\eta}$ (m)", fontsize=11)
        ax2.set_ylabel("Latitude $y$ (km)", fontsize=11)
        ax2.set_title("Zonal Mean $\\bar{\\eta}(y)$\n(Highest in Tropics!)", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200)
            print(f"Saved Stommel-Arons plot to: {save_path}")
        return fig


if __name__ == "__main__":
    sa = StommelAronsModel()
    res = sa.compute_fields(so_enhanced=False)
    print(f"Computed standard Stommel-Arons field. Total upwelling q = {res['total_q_Sv']:.2f} Sv")
    sa.plot_solution(res, "Standard Stommel-Arons Abyssal Flow", "stommel_arons_standard.png")
