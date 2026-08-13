"""Regenerate `cantilever_sweep.csv` — the Module 1 dataset, offline.

Module 1 (`01-opensees-ml/`) produces this table by running `cantilever.py` 75 times
on Stampede3, one PyLauncher task per grid point. The MLP module needs the same
table but must not depend on an HPC allocation, so we reproduce it here.

This is exact, not an approximation. `cantilever.py` builds a single
`elasticBeamColumn` with a `Linear` geometric transformation and a lumped mass on
the free node, so both quantities it records have closed forms:

    k     = 3*E*Iz / L^3                 lateral stiffness of a cantilever
    T     = 2*pi*sqrt(M/k)               single-DOF period from the eigen solve
    delta = P/k,  P = 100 kip            tip deflection from the pushover

Running the OpenSees version would return these same numbers to solver
tolerance. Once you swap in an inelastic section, that stops being true and you
have to run the real sweep — which is exactly the point Module 1 makes.

Usage
-----
    python3 make_sweep_csv.py                   # writes cantilever_sweep.csv
    python3 make_sweep_csv.py --out other.csv
"""

import argparse
import csv
import math

# --- The sweep grid, verbatim from 01-opensees-ml/01-opensees-ml-regression.ipynb ---
NODAL_MASS = [4.19, 4.39, 4.59, 4.79, 4.99]  # kip*s^2/in
LCOL = [100, 200, 300, 400, 500]  # in
EMOD = [3600, 4227, 5000]  # ksi

# --- Fixed section properties, verbatim from 01-opensees-ml/cantilever.py ---
A = 3600.0  # in^2
IZ = 1.08e6  # in^4
PUSHOVER_LOAD = 100.0  # kip

FIELDNAMES = [
    "NodalMass",
    "LCol",
    "E",
    "A",
    "Iz",
    "period",
    "k_theory",
    "top_disp_at_100kip",
]


def run_one(nodal_mass, l_col, e_mod):
    """Mirror one `cantilever.py` task: return its `metrics.json` as a dict."""
    k = 3.0 * e_mod * IZ / (l_col**3)
    period = 2.0 * math.pi * math.sqrt(nodal_mass / k)
    return {
        "NodalMass": nodal_mass,
        "LCol": l_col,
        "E": e_mod,
        "A": A,
        "Iz": IZ,
        "period": period,
        "k_theory": k,
        "top_disp_at_100kip": PUSHOVER_LOAD / k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cantilever_sweep.csv")
    args = ap.parse_args()

    rows = [
        run_one(m, l, e) for m in NODAL_MASS for l in LCOL for e in EMOD
    ]

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} runs to {args.out}")
    print(f"  period            {min(r['period'] for r in rows):.4f} .. "
          f"{max(r['period'] for r in rows):.4f} s")
    print(f"  top_disp_at_100kip {min(r['top_disp_at_100kip'] for r in rows):.5f} .. "
          f"{max(r['top_disp_at_100kip'] for r in rows):.5f} in")


if __name__ == "__main__":
    main()
