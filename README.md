# Dark Matter Halo Modeling using Machine Learning

This project combines **astrophysical modeling and machine learning** to study **dark matter halos in galaxies** using the **SPARC rotation curve dataset**.

The pipeline:

1. Download SPARC galaxy rotation curve data
2. Parse rotation curve measurements
3. Fit **NFW dark matter halos**
4. Generate galaxy feature vectors
5. Train ML models to predict halo parameters
6. Visualize dark matter density profiles

---

# Dataset

SPARC Database  
http://astroweb.cwru.edu/SPARC/

Contains high-quality rotation curves for ~175 galaxies.

---

# Physics Model

Dark matter halos are modeled using the **Navarro–Frenk–White (NFW) profile**

Density profile:

\[
\rho(r) = \frac{\rho_s}{(r/r_s)(1+r/r_s)^2}
\]

Rotation velocity:

\[
V(r) = \sqrt{\frac{GM(r)}{r}}
\]

---

# Machine Learning Models

The project trains several regression models to predict halo parameters:

- Linear Regression
- Gaussian Process Regression
- Neural Networks

Targets predicted:

- Halo mass \(M_{200}\)
- Halo scale radius \(r_s\)

---

# Project Structure

dark-matter-halo-ml
│
├── notebooks
│ └── sparc_dark_matter_halo_fitting.ipynb
│
├── data
│
├── results
│
├── figures
│
├── src
│
├── requirements.txt
└── README.md


---

# Example Outputs

The project produces:

- Galaxy rotation curve fits
- Dark matter density profiles
- ML predicted vs actual halo parameters
- Statistical evaluation of models

---

# Technologies Used

- Python
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Astrophysical modeling

---

# Future Work

Possible extensions:

- Burkert halo profiles
- Bayesian halo inference
- Graph Neural Networks for galaxy dynamics
- Cosmological simulation integration

---

# Author

Vishal Chowdhary

