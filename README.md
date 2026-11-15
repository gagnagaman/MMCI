# Large Scale vs Small Scale: Is the THC Driven by Large-Scale Density Differences?

**Lecture & Research Code by Halldór Björnsson**  
Accompanying presentation: **`MMCI/MMCI.pdf`** (local materials, not included in this repository)

---

## 1. Overview of the Scientific Question

In the early 1960s, Henry Stommel published two distinct and foundational theories of the ocean thermohaline circulation (THC):

1. **The Stommel–Arons Model (1960):**  
   A dynamical, planetary-scale geostrophic model of abyssal flow on a sphere / equatorial $\beta$-plane. Uniform upwelling $e$ in the interior is balanced by vortex stretching ($f \partial w / \partial z = \beta v$), which dynamically forces the interior abyssal flow to move **poleward** (towards the deep water formation source). Intense **Deep Western Boundary Currents (DWBCs)** are required along the western wall to close the mass budget.
   
2. **The Thermohaline 2-Box Model (1961):**  
   A conceptual thermodynamic model assuming the overturning strength $q$ is driven by large-scale lateral density differences:
   $$q = k \Delta\rho = k [\beta (S_1 - S_2) - \alpha (T_1 - T_2)]$$
   This leads to a quadratic equation with two stable regimes: a thermal "forward" mode and a haline "reversed" mode, and predicts catastrophic **advective collapse** under freshwater forcing.

### The Paradox: The Straub (1996) Inconsistency
David Straub (Tellus 1996) showed that these two classical models are mathematically and dynamically inconsistent:
- In the Stommel–Arons dynamical model, potential vorticity conservation requires that the interface height $\eta(x, y)$ slopes upwards towards the tropics.
- When the column-averaged density is computed across the basin, **the ocean column is actually denser in the tropics than at high latitudes** ($\bar{\rho}_{\text{tropics}} > \bar{\rho}_{\text{polar}}$).
- Therefore, the assumption $q = k \Delta\rho$ **directly contradicts thermal wind balance** in an inviscid, geostrophic basin!

---

## 2. The Resolution: Southern Ocean Winds (Greatbatch & Lu 2003; Björnsson)

If $q = k \Delta\rho$ contradicts thermal wind, why do Ocean General Circulation Models (OGCMs) consistently exhibit a linear relationship between $q$ and $\Delta\rho$?

### Greatbatch & Lu (2003) & Numerical Experiment (`wind_experiment.py` / `StommAr3.m`)
Greatbatch & Lu demonstrated that in a two-layer shallow-water system:
- When the anomaly damping timescale is shorter than the propagation timescale ($R \ll 1$), Stommel–Arons dynamics hold.
- When Southern Ocean winds (Toggweiler & Samuels 1995; Gnanadesikan 1999) are introduced, Ekman divergence creates concentrated upwelling in the Southern Ocean ($e(y)$ is enhanced south of 40°S).
- **The Experiment:** By varying the overturning strength $q \in [10, 25]\text{ Sv}$ and solving for the Stommel–Arons interface height, the resulting interhemispheric density difference $\Delta\bar{\rho} = \bar{\rho}_N - \bar{\rho}_S$ is computed.
- **The Result:** An almost perfect linear relation emerges:
  $$q \approx k \Delta\bar{\rho} + c$$
- **The Physical Implication:** The linear flow law is **an emergent consequence of the dynamics and Southern Ocean winds, NOT the causal driver!** The THC is not driven by large-scale density differences; rather, **the circulation creates the density differences**.

---

## 3. Python Scripts & Structure

```
Rannsoknir/MMCI/
├── MMCI_Lecture_Notebook.ipynb # Complete interactive slide-by-slide Jupyter Notebook
├── stommel_box.py              # Stommel (1961) 2-box model & stability curve
├── stommel_arons.py            # Stommel-Arons (1960) 2D geostrophic flow & interface solver
├── gnanadesikan_model.py       # Corrected Gnanadesikan (1999) box model (shared)
├── gnanadesikan_pycnocline.py  # Pycnocline depth D(qn) & stability-curve wrapper
├── gnanadesikan_f_vs_overturning.py # Stability curve F vs qn (single panel)
├── wind_experiment.py          # Southern Ocean wind experiment (reproducing Slide 28)
├── MMCI/                       # Original presentation materials (local only)
│   ├── MMCI.pdf                # Full presentation slides
│   ├── Fyrirlestur.tex         # LaTeX Beamer / Prosper slide source
│   ├── StommAr.m, StommAr2.m, StommAr3.m  # Original MATLAB scripts
│   └── *.eps, *.tif            # Lecture figures and diagrams
└── README.md                   # This documentation
```

> This repository tracks the Python routines, the Jupyter notebook, and the README.
> Generated PNG figures and the local `MMCI/` presentation folder are intentionally excluded.

---

## 4. Usage Instructions

### Running the Python Scripts

1. **Stommel 2-Box Model & Stability Curve:**
   ```bash
   python3 stommel_box.py
   ```
   Generates `stommel_stability.png` illustrating the stable and unstable branches and tipping point $F_c = \sigma^2 / (4\gamma)$.

2. **Stommel-Arons 2-Layer Model:**
   ```bash
   python3 stommel_arons.py
   ```
   Generates `stommel_arons_standard.png` showing the 2D interface height $\eta(x, y)$, geostrophic velocity vectors, and the zonal mean profile $\bar{\eta}(y)$ demonstrating Straub's paradox.

3. **Gnanadesikan (1999) Pycnocline Model:**
   ```bash
   python3 gnanadesikan_pycnocline.py
   ```
   Generates `gnanadesikan_stability.png` showing pycnocline depth scaling $D(q_n)$ and the resulting asymmetric stability curve. The implementation uses the corrected Work.nb closure from the shared `gnanadesikan_model.py`.

4. **Gnanadesikan Model Stability Curve:**
   ```bash
   python3 gnanadesikan_f_vs_overturning.py
   ```
   Generates `gnanadesikan_f_vs_overturning.png`, the dedicated stability curve for the Gnanadesikan model: the stable and unstable branches of $q_n(F)$ and the saddle-node tipping point ($F_c \approx 3.1$ Sv at $q_c \approx 27$ Sv).

5. **Southern Ocean Wind Experiment (Slides 25–28):**
   ```bash
   python3 wind_experiment.py
   ```
   Generates `wind_experiment_results.png`, proving the emergent linear relation $q \propto \Delta\bar{\rho}$.

---

## 5. The Interactive Jupyter Notebook

Launch Jupyter to explore the presentation interactively:

```bash
jupyter notebook MMCI_Lecture_Notebook.ipynb
```

The notebook contains:
- Full mathematical derivations in LaTeX.
- Interactive code cells running every model.
- High-resolution inline visualizations for each stage of the lecture.
- A dedicated **stability curve for the Gnanadesikan model** ($q_n$ vs. $F$), showing the stable and unstable branches and the saddle-node tipping point.
