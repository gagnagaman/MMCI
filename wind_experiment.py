"""
Reproduce the Southern Ocean Wind Experiment from Lecture MMCI (Slides 25-28).

Demonstrates the Greatbatch & Lu (2003) mechanism:
In the Stommel-Arons dynamical regime, non-uniform vertical upwelling
(concentrated in the Southern Ocean by wind-driven Ekman divergence)
creates an apparent linear relation between overturning strength q and
interhemispheric density difference Delta_rho.
The density contrast is an emergent consequence, not the causal driver!

Reference:
    Greatbatch, R. J., & Lu, J. (2003). Reconciling the Stommel box model with the
    Stommel-Arons model: A possible role for Southern Hemisphere wind forcing?
    Journal of Physical Oceanography, 33(7), 1618-1632.
"""

import os
import sys
from typing import Dict
import numpy as np
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stommel_arons import StommelAronsModel


def run_wind_experiment(num_cases: int = 19, save_path: str = None):
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "figures",
            "wind_experiment_results.png",
        )

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    model = StommelAronsModel(
        beta=2.0e-11,
        H=4000.0,
        g_prime=5.0e-3,
        h0=2000.0,
        Lx=4.0e6,
        Ly=10.0e6
    )

    q_vals_Sv = np.zeros(num_cases)
    delta_rho_vals = np.zeros(num_cases)

    # Base parameters for grid
    nx = 50
    ny = 100
    x = np.linspace(0, model.Lx, nx)
    y = np.linspace(-model.Ly / 2.0, model.Ly / 2.0, ny)
    dx = model.Lx / (nx - 1)
    dy = model.Ly / (ny - 1)

    print(f"Running Southern Ocean wind experiment over {num_cases} overturning strengths...")

    for k in range(num_cases):
        # Base upwelling e scaled linearly
        e_base = (6.7e-7 / 3.0) * (1.0 + (k + 1) / 10.0)
        
        # Spatial pattern: Southern Ocean upwelling enhancement (Toggweiler & Samuels effect)
        E = np.full((ny, nx), e_base)
        E_enhanced = E / 1.5
        E_enhanced[:10, :] = E[:10, :] * 2.5   # South of ~40°S
        E_enhanced[10:20, :] = E[10:20, :] * 2.0

        # Total overturning circulation in Sv:
        q_tot = np.sum(E_enhanced) * dx * dy * 1e-6
        q_vals_Sv[k] = q_tot

        # Integrate e westward from eastern boundary (X = Xe = Lx)
        integrated_e = np.zeros((ny, nx))
        for i in range(ny):
            integrated_e[i, :] = -np.cumsum(E_enhanced[i, ::-1])[::-1] * dx

        # Solve interface height cubic at all points
        eta = model.solve_interface_height_cubic(integrated_e, np.meshgrid(x, y)[1], 1.0)

        # Zonally averaged interface height (mean along x)
        eta_zonal = np.nanmean(eta, axis=1)

        # Layer-averaged density:
        # Lower layer has density rho2 = 1025 + Delta_rho_strat = 1025 + 5 kg/m^3
        # Upper layer has density rho1 = 1025 kg/m^3
        # Depth-averaged density: rhobar(y) = 1025 + (eta_zonal / H) * 5
        rhobar = 1025.0 + (eta_zonal / model.H) * 5.0

        # Interhemispheric density difference: Northern hemisphere minus Southern hemisphere
        # Rows 0..49 are South, rows 50..99 are North
        rho_north = np.mean(rhobar[50:])
        rho_south = np.mean(rhobar[:50])
        delta_rho_vals[k] = rho_north - rho_south

    # Perform linear regression to demonstrate the apparent flow rule: q = k * Delta_rho + c
    poly = np.polyfit(delta_rho_vals, q_vals_Sv, 1)
    k_slope, intercept = poly

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Vertical velocity profile e(y)
    ax1.plot(y * 1e-3, E_enhanced[:, 0] * 1e6, "r-", linewidth=2.5)
    ax1.set_xlabel("Latitude $y$ from Equator (km)", fontsize=11)
    ax1.set_ylabel("Vertical Upwelling $e$ ($10^{-6}$ m/s)", fontsize=11)
    ax1.set_title("Southern Ocean Enhanced Upwelling $e(y)$\n(Drake Passage Wind-Driven Ekman Divergence)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Apparent flow rule Delta_rho vs q (reproducing Slide 28)
    ax2.plot(delta_rho_vals, q_vals_Sv, "bo", markersize=8, label="Numerical Experiment")
    fit_x = np.linspace(min(delta_rho_vals), max(delta_rho_vals), 50)
    ax2.plot(fit_x, k_slope * fit_x + intercept, "r--", linewidth=2.0,
             label=f"Linear Fit ($q = {k_slope:.1f} \\Delta\\rho + {intercept:.1f}$)")
    ax2.set_xlabel("Interhemispheric Density Difference $\\Delta\\bar{\\rho} = \\bar{\\rho}_N - \\bar{\\rho}_S$ (kg/m$^3$)", fontsize=11)
    ax2.set_ylabel("Overturning Circulation $q$ (Sv)", fontsize=11)
    ax2.set_title("Emergent Linear Flow Rule ($q \\propto \\Delta\\rho$)\nResult of Southern Ocean Winds, NOT Causal!", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"Saved experiment plot to: {save_path}")
    print(f"Apparent hydraulic conductivity k: {k_slope:.2f} Sv / (kg/m^3)")

    return q_vals_Sv, delta_rho_vals


if __name__ == "__main__":
    run_wind_experiment()
