# Module 2 — Multi-Layer Perceptrons

**2026 SPARC · Day 3, Session 3b · CPU only**

Module 1 recovered `T = 2π√(ML³/3EI)` exactly with a linear regression — because
we knew to take logarithms first. This module asks what to do when you don't.

## The arc

1. **The same 75 OpenSees runs, raw features.** Linear regression scores
   `R² = 0.984` and predicts *negative* fundamental periods (121% error on the
   stiffest columns). A high R² hid a broken model; the residuals plotted against
   column length show the structure immediately.
2. **Why not stack more linear layers?** `W₂(W₁x + b₁) + b₂ = W̃x + b̃`. The
   notebook collapses a 1,217-parameter linear stack to a 1×3 matrix by hand.
3. **An MLP on the same raw features** recovers the physics with no logarithms
   supplied.
4. **UAT** and its four caveats: width, extrapolation, trainability,
   generalisation. Three of them bite before the module ends.
5. **AD for design sensitivities** — `∂T/∂L` from the trained surrogate, checked
   against the analytic `1.5·T/L`. Same machinery Module 4 uses for PDE residuals.
6. **Overfitting**, honestly (see below).
7. **From parameters to fields** — fit the deflection curve `w(x)`, first with 100
   readings, then with 8. The sparse fit passes through every point and is still
   wrong at the support. Adding a boundary-condition penalty fixes it, which is
   the one-line preview of Module 4.

## Two results worth knowing before you teach it

Both are measured live in the notebook and both contradict the usual story.

**Width past a threshold is noise.** Averaged over three seeds, the test error
drops ~8× from width 1 to width 4, then is essentially flat to width 32 — while
the seed-to-seed spread is 1.3–1.5×. Changing the seed moves the error as much as
an 8× change in width. The notebook plots the spread band for this reason.

**The train/test gap is a bad overfitting diagnostic.** Run as a 2×2 —
noiseless / 2% label noise × 1,217 / 198,657 parameters:

| data | architecture | params | train | test | ratio |
| --- | --- | --- | --- | --- | --- |
| noiseless | width 32, depth 3 | 1,217 | 7.1e-07 | 2.5e-05 | 35× |
| noiseless | width 256, depth 5 | 198,657 | 1.6e-08 | 1.5e-04 | 9,209× |
| 2% noise | width 32, depth 3 | 1,217 | 2.5e-05 | 3.1e-03 | 122× |
| 2% noise | width 256, depth 5 | 198,657 | 4.1e-11 | 3.1e-03 | 75,675,604× |

The oversized network genuinely interpolates — it passes through all 60 points,
noise included. But its *test* error is only ~6× worse when the data is clean and
**identical** under noise. The ratio reads 7.6e+07 and would have you believe the
model is ruined; its predictions are as good as the small network's.

So: judge on test error, not on the gap. And note which change actually moved the
test error — adding 2% noise cost one to two orders of magnitude, far more than
any architecture change. On this smooth 3D problem capacity is nearly free and
**data quality is the binding constraint**, which is the argument for putting
physics in the loss.

(This is not universal. On high-dimensional or badly conditioned problems the
classic U-shaped curve does appear. The point is to measure rather than assume.)

## Files

| File | Role |
| --- | --- |
| `02-mlp-cantilever.ipynb` | Solution notebook |
| `02-mlp-cantilever-exercise.ipynb` | Same, with the key code blanked (`# TODO`) |
| `cantilever_sweep.csv` | The Module 1 dataset — 75 runs |
| `make_sweep_csv.py` | Regenerates that CSV offline, no allocation needed |
| `figs/` | Figures borrowed from the SciML course and the DesignSafe PINN training |

## The dataset

`cantilever_sweep.csv` is the Module 1 sweep: `NodalMass × LCol × E` = 5 × 5 × 3 =
75 runs, with `period` and `top_disp_at_100kip`.

Module 1 produces it by running `01-opensees-ml/cantilever.py` 75 times on
Stampede3. Because that model is a single `elasticBeamColumn` with a lumped mass,
both recorded quantities have closed forms, so `make_sweep_csv.py` reproduces the
table exactly without OpenSees or an allocation:

```bash
python3 make_sweep_csv.py          # -> cantilever_sweep.csv
```

Verified against the real thing: log-feature OLS on this CSV returns
`[0.5000, 1.5000, -0.5000]` with `R² = 1.000000`, matching the HPC job's
`coef = [0.500, 1.500, -0.500]`.

Swap in an inelastic section and the closed form disappears — which is exactly
when you need the real sweep, and exactly when the MLP starts earning its place.

## Going deeper

- [SciML — MLP and function approximation](https://kks32-courses.github.io/sciml/00-mlp/mlp.html)
- [SciML — Universal Approximation Theorem](https://kks32-courses.github.io/sciml/00-mlp/uat.html)
- [SciML — automatic differentiation](https://kks32-courses.github.io/sciml/00-mlp/ad.html)
- [SciML — regularization](https://kks32-courses.github.io/sciml/00-mlp/regularization.html)
