# Module 5 — Operator Learning with DeepONet

**2026 SPARC · Day 3, Session 3b · ~30 min · CPU only**

> Cantilever DeepONet example after **Somdatta Goswami** (Johns Hopkins).

Module 4's PINN solved the beam from the equation alone — and solved exactly
**one load case**. Change the load and every weight is wrong. A design study needs
hundreds.

This module learns the solution *operator* instead.

## Where this sits

| Module | Learns | Maps | New query costs |
| --- | --- | --- | --- |
| 1 Regression | 3 coefficients | ℝ³ → ℝ | instant |
| 2 MLP | a nonlinear surrogate | ℝ³ → ℝ | instant |
| 4 PINN | one solution field | `x ↦ w(x)` | **retrain (minutes)** |
| 5 DeepONet | the solution **operator** | `q(·) ↦ w(·)` | instant |

## The problem

A 2 m × 0.2 m linear-elastic cantilever under a displacement-controlled boundary
condition drawn from a Gaussian random field, so no two load cases look alike.

| | |
| --- | --- |
| Load cases | 1000 (800 train / 200 test) |
| Input | applied displacement at **100 sensors** along the span |
| Output | displacement field on **1314 FE nodes**, two components (`u_x`, `u_y`) |
| Total | 2.6 M field values |

## The arc

1. **Functions to operators.** An operator maps a function to a function. The
   input is discretised at `m = 100` sensors; the output stays mesh-free. Note the
   asymmetry — fixed input grid, arbitrary output points.
2. **A naive-MLP baseline.** Concatenate `[u₁…u₁₀₀, x, y]` → 102 inputs → `(uₓ, u_y)`.
   Held to the same parameter budget, optimiser, and epoch count as the DeepONet,
   so the architecture has to earn its place rather than being assumed.
3. **The operator UAT** (Chen & Chen, 1995) read literally as an architecture:
   `G(u)(y) ≈ Σₖ bₖ(u)·tₖ(y)` — branch net for coefficients, trunk net for a
   basis, combined by one `einsum`.
4. **Implementation.** Two output components via `2p` outputs reshaped to
   `(2, p)`; the contraction is `torch.einsum("bop,qop->bqo", B, T)`.
5. **The learned basis** — the payoff, see below.
6. **Zero-shot prediction** on unseen loads, evaluation on a grid the mesh never
   had, and the honest limits.

## Why the factorisation matters

The trunk depends only on the query point, never on the load case. So it learns
**one shared basis** for the whole family, and the branch only says how much of
each basis function this load needs.

Two consequences:

- The trunk is evaluated **once** per batch and reused for every load case. The
  naive MLP pushes all 100 sensor values through its layers 1314 times per case —
  the redundancy DeepONet's structure removes.
- You get an interpretable object out: a basis.

## The learned basis is the point of the module

The trunk never sees a load case, so its `p` outputs form a basis for every field
the operator can produce. Plot the dominant modes and they look like mode
shapes — smooth and organised by spatial frequency, with no beam mode shape,
polynomial, or Fourier term ever supplied.

Then the module does something most treatments skip: it checks whether that basis
is any *good*. It is not, and the diagnosis is the most useful content here.

**The physics is genuinely low-rank.** A POD of the 800 training fields needs
**4 modes for 90%** of the field energy and **6 for 99%**. The solution manifold
is about six-dimensional; we gave the trunk `p = 100`.

**The trunk found the right subspace.** Principal angles between the trunk's
leading four directions and the four leading POD modes are **within ~12°**. The
network independently rediscovered the dominant deformation modes of a cantilever.

**But it is a terrible basis for that subspace:**

| Measurement | Value | POD would give |
| --- | --- | --- |
| Mean \|cosine\| between distinct modes | ~0.4 | 0 (orthogonal) |
| Condition number | ~10⁵–10⁶ | ~10 |
| Largest principal angle vs POD at k=6 | >80° | 0° |
| Modes to reach 90% of contribution | 82 of 100 | 4 |

Nothing in the loss asked for orthogonality, parsimony, or an ordering, so there
is none past the leading few directions.

**Consequence: it cannot be truncated.** POD carries the Eckart–Young guarantee —
top-`k` is the provably optimal rank-`k` approximation, and it is essentially
below 1% by 10 modes. The DeepONet basis truncated the same way is still ~35%
wrong at 50 of 100 modes, because the information is smeared across all of them and
dropping any destroys cancellations the network relied on. The notebook plots both
curves side by side.

So a DeepONet is a **learned spectral method with an unmanaged basis**: right
subspace, wasteful representation. The standard fixes, in increasing order of
effort — orthogonalise post hoc (QR/SVD of the trunk, free, enough for
interpretation); add `‖TᵀT − I‖²` to the loss; or use **POD-DeepONet**, replacing
the learned trunk with precomputed POD modes so the branch learns only
coefficients.

A pretty mode plot is not evidence of a well-conditioned model.

## The honest limits

- **The training data is the expensive part.** 800 FE solves. DeepONet moved the
  cost from query time to a one-off offline campaign; it did not remove it. The
  economics only work for many-query workflows.
- **It knows only the family it trained on.** Our loads are GRF draws with a
  particular length scale. A point load, or a much shorter correlation length,
  extrapolates badly and *without warning*. This is Module 2's UAT domain caveat,
  now applying to a space of functions — where "out of distribution" is far harder
  to detect.
- **The input grid is fixed.** Change the sensor layout and the branch is invalid.
  Fourier Neural Operators fix exactly this by working in Fourier space, giving
  discretisation invariance.
- **No physics is enforced.** Purely data-driven; nothing in the loss requires
  equilibrium.

## Closing the circle

That last limitation has an obvious fix given where the session has been — add
Module 4's residual to Module 5's loss:

```
L = ‖G_θ(u) - s‖²  +  λ‖N[G_θ(u)]‖²
     data (Mod 5)        PDE residual (Mod 4)
```

Differentiate the trunk with respect to its inputs — the same `create_graph=True`
trick — and the residual can be evaluated at any point for any input function.
With enough physics weight you can train an operator with **very few or no
labelled solutions**. That is the synthesis of the whole session in one model.

## Files

| File | Role |
| --- | --- |
| `05-deeponet-cantilever.ipynb` | Solution notebook |
| `05-deeponet-cantilever-exercise.ipynb` | Same, with the key code blanked (`# TODO`) |
| `figs/` | Operator concept, DeepONet and PI-DeepONet architectures, basis analysis, cantilever schematic |

## The dataset

`cantilever_beam_deflection.mat` (20 MB) is **committed alongside the notebook**, so
a `git clone` or the one-click DesignSafe git-pull gives you a fully working module
with no network dependency at run time.

The notebook's `find_mat()` still searches in order, so it works from any of them:

1. `CommunityData/Training/2026-SPARC/Day3/Session3b/`
2. the notebook's own folder ← the committed copy
3. `../training-deeponet/`
4. a GitHub download (one time, last resort)

**For the live session, also put the `.mat` in the Community Data folder** so
attendees opening the notebook from there hit a local file. 40 people each pulling
20 MB over the network at the same moment is a failure mode you do not want to
discover during a workshop.

The same file also lives in
[`DesignSafe-Training/deeponet`](https://github.com/DesignSafe-Training/deeponet)
and in this repo's sibling `training-deeponet/`.

| Variable | Shape | Meaning |
| --- | --- | --- |
| `app_disp` | (1000, 100) | input functions at the sensors |
| `sensor_loc_disp` | (1, 100) | sensor positions, x ∈ [0, 2] |
| `coord_x`, `coord_y` | (1, 1314) | FE node coordinates |
| `disp_x`, `disp_y` | (1000, 1314) | output displacement fields |

## A note on the port to PyTorch

The original of this example is JAX/Flax with 200,000 training epochs — not
live-demoable, and a third framework for the audience to install.

This version is PyTorch, so Modules 2, 4, and 5 all share one API, and it trains
in well under a minute on CPU. Normalisation matters: the coordinates are rescaled
to [-1, 1]² because `y` spans 0.2 m against `x`'s 2 m, and without it the trunk
barely sees `y`. Output components are scaled separately (`u_x` and `u_y` differ by
roughly 5× in magnitude).

The original JAX version, with its pretrained weights, remains at
[DesignSafe-Training/deeponet](https://github.com/DesignSafe-Training/deeponet).

## Going deeper

- [SciML — DeepONet from scratch](https://kks32-courses.github.io/sciml/02-deeponet/deeponet.html)
- [SciML — physics-informed DeepONet](https://kks32-courses.github.io/sciml/02-deeponet/pideeponet.html)
- [SciML — how to read a DeepONet](https://kks32-courses.github.io/sciml/02-deeponet/deeponet-explanation.html)
- [SciML — the batched einsum, in detail](https://kks32-courses.github.io/sciml/02-deeponet/einsum.html)
- [SciML — Fourier Neural Operators](https://kks32-courses.github.io/sciml/05-fno/fno.html)
- [SciML — function encoders](https://kks32-courses.github.io/sciml/03-function-encoder/03-function-encoder.html)
- Lu, Jin & Karniadakis (2019), *DeepONet* · Chen & Chen (1995), operator UAT ·
  Li et al. (2020), *Fourier Neural Operator*
