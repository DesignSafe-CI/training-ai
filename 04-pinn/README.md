# Module 4 — Physics-Informed Neural Networks

**CPU only**

Module 2 ended with a network fitted to eight deflection readings. It passed
through all eight and was still wrong — nonzero slope at the clamped support,
nonsense past the tip. We also knew the governing equation for that beam and
ignored it.

This module puts the whole equation in the loss.

## The change of viewpoint

| | Surrogate (Modules 2–3) | PINN (this module) |
| --- | --- | --- |
| Network input | design parameters `(M, L, E)` | a **coordinate** `x` |
| Network output | a quantity of interest | the **field** `w(x)` |
| Trained on | labelled simulation runs | the **PDE residual** |
| Labelled data | thousands | **zero** (optional) |
| The network is | an approximation of a solver | the *solution itself* |

## The problem

Tip-loaded cantilever, clamped at `x = 0`:

```
EI·w''(x) = M(x) = -P(L - x),     w(0) = 0,   w'(0) = 0
```

We use the **second-order** moment–curvature form rather than `EI·w'''' = q`.
Both describe the same beam, but the second-order form needs two derivatives of
the network instead of four, and every differentiation amplifies the network's own
wiggle. Part 5 shows exactly how much.

## Non-dimensionalise, or it will not train

This is the part most tutorials skip and the most common reason a hand-rolled PINN
fails. In SI units `EI = 2×10⁶` while the deflection is `~0.02 m`, so the residual
`EI·w'' + P(L-x)` has magnitude `~5000` — five orders of magnitude above the
quantity of interest. Gradient descent on that is hopeless.

With `ξ = x/L` and `W = w/(PL³/3EI)` the problem becomes `O(1)`:

```
W''(ξ) = -3(1 - ξ),    W(0) = W'(0) = 0,    W_exact = -(3ξ² - ξ³)/2
```

Everything in the notebook solves that; results convert back to millimetres only
for plotting.

## The arc

1. **The residual via autograd** — `create_graph=True`, checked against the
   analytic `W'' = -3(1-ξ)` to machine precision. AD is not finite differences:
   no step size, no truncation error.
2. **Soft boundary conditions** — solve the beam from `L = residual + λ·BC` with
   **zero labelled data**. Trains in ~4 s to 0.003 mm accuracy.
3. **A λ sweep** that does not say what you expect (see below).
4. **Hard boundary conditions** — `W(ξ) = ξ²·N(ξ)` satisfies *both* `W(0) = 0` and
   `W'(0) = 0` identically, for any weights. Both come out as exactly `0.000e+00`,
   the loss loses a term, and λ disappears. The `ξ²` factor is not arbitrary: a
   clamped end needs two conditions, so it needs a double root.
5. **The inverse problem** — the headline.
6. **When PINNs are the wrong tool** — three caveats, measured.

## The inverse problem is the real application

You have a beam in the field, you can measure deflection at a handful of points,
and you do not know its stiffness. Make the unknown a trainable parameter:

```python
self.log_kappa = nn.Parameter(...)      # optimise log κ: positive by construction
```

From **8 readings with 0.15 mm noise**, the notebook recovers
`E = 199.1 GPa` against a true `200 GPa` — **0.43% error** — and returns the full
deflection field as a by-product, including at the support where nothing was
measured. A noise study shows the degradation: 0.4 mm noise still gives 1.1%.

This is where PINNs beat classical approaches. A classical inverse solve wraps an
outer optimiser around a full forward solve per iteration; the PINN needs one
backward pass, the PDE acts as the regulariser for sparse noisy data, and adding
another unknown *field* `EI(x)` means adding a second small network rather than
reformulating the problem.

## Three results that complicate the story

All measured in the notebook.

**λ is not a dial you turn up.** Raising `λ_BC` does not tighten the boundary
condition — it sits near `10⁻⁶` from λ=1 to λ=1000 then gets *worse* — while the
solution error climbs monotonically, ~3 orders of magnitude by λ=10⁴. The common
advice to "use a big λ to really enforce the BCs" is actively destructive here.
Our own initial guess of 100 was already 3× worse than λ=1. Sweep it; hard
constraints delete the question.

**A PINN is not a fast forward solver.** ~90,000× slower than the exact solution
and less accurate. That ratio does not tune away — a PINN solves a global
optimisation where FEM solves a sparse linear system. Do not sell it as a speedup;
sell it on inverse problems, missing physics, high-dimensional parametric PDEs, and
awkward geometry.

**Accurate only where constrained.** The trained network's `W''` — the quantity in
the loss — matches the exact value to `3e-03`. Its `W''''`, which nothing
constrained and which should be exactly zero, reaches `2.3`. If you need bending
moments, put moments in the loss; do not differentiate a displacement network
twice more and hope. For genuinely fourth-order problems use a mixed formulation:
two networks coupled by `EI·w'' = M` and `M'' = q`.

## Files

| File | Role |
| --- | --- |
| `04-pinn-cantilever.ipynb` | Solution notebook |
| `04-pinn-cantilever-exercise.ipynb` | Same, with the key code blanked (`# TODO`) |
| `figs/` | PINN schematic, strong-BC and soft-vs-hard figures, inverse-problem setup |

Runs in float64 (`torch.set_default_dtype`) — second derivatives deserve it.

## What this module cannot do

It produced **one deflection field for one load case**. Change `P` and every
weight is wrong; change to a distributed load and you retrain from scratch. A
design study needs hundreds of load cases and a PINN, as built here, cannot serve
one of them.

That is Module 5.

## Going deeper

- [SciML — PINNs from scratch](https://kks32-courses.github.io/sciml/01-pinns/pinns.html)
- [SciML — soft vs hard constraints](https://kks32-courses.github.io/sciml/01-pinns/poisson.html)
- [SciML — adaptive loss weights](https://kks32-courses.github.io/sciml/01-pinns/adaptive-weights.html)
- [SciML — collocation strategies](https://kks32-courses.github.io/sciml/01-pinns/collocation.html)
- [SciML — inverse problems](https://kks32-courses.github.io/sciml/01-pinns/inverse-heat.html)
- [SciML — Burgers' equation](https://kks32-courses.github.io/sciml/01-pinns/burgers.html) — nonlinear, shock-forming
- [DesignSafe PINN training](https://github.com/DesignSafe-Training/pinn) — heat transfer and Burgers
- Raissi, Perdikaris & Karniadakis (2019)
