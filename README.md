# DesignSafe AI Training

[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue?logo=github)](https://designsafe-ci.github.io/training-ai)
[![Jupyter Book](https://img.shields.io/badge/Powered%20by-Jupyter%20Book-orange)](https://jupyterbook.org)

Hands-on **AI training** — running modern AI workflows on
**[DesignSafe](https://www.designsafe-ci.org/)** with the DesignSafe API
([`dapi`](https://designsafe-ci.github.io/dapi/)).

---

## Scientific Machine Learning for Engineers

**Six modules. Five of them are the same cantilever beam.**

Modules 1–5 are built around a single structure so that each answers a question the
previous one *cannot*. The beam from Module 1's OpenSees sweep is the same beam
whose deflection field Module 5 learns an operator for. Module 6 then leaves the
beam behind for real post-hurricane imagery and a pre-trained foundation model.

| # | Module | The network learns | Maps | Cost of a new query |
| --- | --- | --- | --- | --- |
| 1 | [Regression](#module-1--regression-on-hpc) | 3 coefficients | $\mathbb{R}^3 \to \mathbb{R}$ | instant |
| 2 | [MLP](#module-2--multi-layer-perceptrons) | a nonlinear surrogate | $\mathbb{R}^3 \to \mathbb{R}$ | instant |
| 3 | [XAI](#module-3--explainable-ai) | *why* the model said that | — | instant |
| 4 | [PINN](#module-4--physics-informed-neural-networks) | one solution field | $x \mapsto w(x)$ | **retrain** |
| 5 | [DeepONet](#module-5--operator-learning) | the solution **operator** | $q(\cdot) \mapsto w(\cdot)$ | instant |
| 6 | [CLIPSeg](#module-6--computer-vision-at-scale) | someone else's foundation model | image $\mapsto$ debris map | instant |

Modules 1–3 are **data-driven**: fit a flexible model to simulation output, then
interrogate it. Modules 4–5 are **physics-driven**: build the governing equation
into the model so it cannot learn something the physics forbids. Module 6 leaves
the beam behind for real imagery and a pre-trained foundation model — the applied
counterpart to everything before it.

### Running order

| Module | Notes |
| --- | --- |
| Framing — the six questions | |
| **1. Regression** on Stampede3 | **Submit the HPC job first**, harvest at the end |
| **2. MLP** | |
| **3. XAI** | |
| *break* | falls on the data-driven → physics-driven boundary |
| **4. PINNs** | |
| **5. Operator learning** | |
| **6. CLIPSeg** | walkthrough, not hands-on — see below |
| Job results + wrap-up | |

Module 1 submits to a TACC queue, so it goes first and its results are collected
at the end. Sections marked `optional` in Modules 2–5 can be skipped and read
later.

### Requirements

| Modules | Needs |
| --- | --- |
| **2–5** | CPU only. A few minutes each on DesignSafe JupyterHub or Colab. No allocation, no GPU. |
| **1** | A TACC allocation (submits to Stampede3 via `dapi`). |
| **6** | Parts 1–2: a DesignSafe session, GPU recommended. Parts 3–4: a TACC allocation. |

---

## Open the notebooks

Modules 2–5 each ship an **exercise** notebook (code blanked, `# TODO`) and a
**solution** notebook. Modules 1 and 6 ship one notebook each — Module 1 submits a
real HPC job, and Module 6 drives a production pipeline whose code would simply
break if blanked, so its exercises are inline as variations to make on working
code.

### On DesignSafe (Community Data)

Notebooks are published to
`CommunityData/Training/2026-SPARC/Day3/3b-SciML/`.

| Module | Notebook |
| --- | --- |
| 1. Regression | [![Try on DesignSafe](DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/2026-SPARC/Day3/3b-SciML/01-opensees-ml-regression.ipynb) |
| 2. MLP | [![Try on DesignSafe](DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/2026-SPARC/Day3/3b-SciML/02-mlp-cantilever.ipynb) |
| 3. XAI | [![Try on DesignSafe](DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/2026-SPARC/Day3/3b-SciML/03-xai-lateral-spreading.ipynb) |
| 4. PINN | [![Try on DesignSafe](DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/2026-SPARC/Day3/3b-SciML/04-pinn-cantilever.ipynb) |
| 5. DeepONet | [![Try on DesignSafe](DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/2026-SPARC/Day3/3b-SciML/05-deeponet-cantilever.ipynb) |
| 6. CLIPSeg | [![Try on DesignSafe](DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/2026-SPARC/Day3/3b-SciML/06-clipseg-debris.ipynb) |

Data files to place in the same Community Data folder:
`cantilever_sweep.csv` (Module 2), `RF_YN_Model3.csv` (Module 3),
`cantilever_beam_deflection.mat` (Module 5, 20 MB). All three are also committed
to this repo, and every notebook searches Community Data first, then its own
folder, then falls back to a GitHub download — so a partial upload degrades
gracefully rather than failing.

### Clone the whole repo into DesignSafe (one click) 

[![Open in DesignSafe](https://img.shields.io/badge/Open%20in%20DesignSafe-clone%20%2B%20open-006FBA?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.designsafe-ci.org/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FDesignSafe-CI%2Ftraining-ai&branch=main&targetpath=MyData%2Ftraining-ai&urlpath=lab%2Ftree%2FMyData%2Ftraining-ai%2FREADME.md)

Clones — or fast-forwards — this repo into `~/MyData/training-ai/`. Re-click any
time to pull the latest version.

---

## The modules

### Module 1 — Regression on HPC

`01-opensees-ml/` · [details](https://github.com/DesignSafe-CI/training-ai/blob/main/01-opensees-ml/README.md)

Sweeps a 2D cantilever pushover in OpenSeesPy across `NodalMass x LCol x E` on
Stampede3 via `dapi` and PyLauncher, then fits a linear regression that recovers

```
log T = const + 0.5·log M + 1.5·log L - 0.5·log E     <=>     T = 2π·√(M·L³/3EI)
```

The coefficients come back as `[0.500, 1.500, -0.500]` with `R² = 1.0`. They are
not abstract weights — they are the physical exponents, so a correct pipeline
*must* recover them. That makes the regression a test of the entire workflow.

**The catch that sets up Module 2:** it worked because we knew to take logarithms.

The folder also holds an **optional variant**, `01-opensees-ml-workflow-dag.ipynb`,
which runs the same study as an explicit two-job DAG (`sweep → train`) through
DesignSafe's workflow service, so a retrain costs one core instead of a 48-core
resweep. Not part of the three-hour agenda; published as a sub-section.

### Module 2 — Multi-Layer Perceptrons

`02-mlp/`

Hands the same 75 runs to a linear model on **raw** features. It scores
`R² = 0.984` — and predicts *negative* fundamental periods, with 121% error on the
stiffest columns. An MLP on the same raw features recovers the physics without
being told about logarithms.

Covers: why stacked linear layers collapse (`W₂W₁ = W̃`), the Universal
Approximation Theorem and its four caveats, the role of width, reverse-mode AD (used
here for design sensitivities `∂T/∂L`, and the mechanism behind Module 4), and
overfitting. Ends by fitting the beam's deflection *field* from 8 sparse readings
— which fails at the support, motivating physics in the loss.

Two findings worth flagging, both measured live in the notebook and both
contradicting the usual story:

- Changing the random seed moves the test error as much as an 8× change in width.
  Report seed spread before claiming an architecture is better.
- A 198,657-parameter network on 60 points interpolates the training set
  (loss `4e-11`) while its **test** error stays level with a 1,217-parameter
  one's. The train/test ratio screams `7.6e+07`; the predictions are fine. Judge
  on test error, not the gap — and note that 2% label noise costs far more
  accuracy than any architecture change did.

### Module 3 — Explainable AI

`03-xai/`

Module 2 traded three interpretable exponents for thousands of opaque weights.
This module buys the interpretation back, on 7,291 real field observations of
**lateral spreading** (four predictors: groundwater depth, distance to free face,
slope, PGA).

Decision tree (readable) → XGBoost (accurate, opaque) → three definitions of
"feature importance" compared → SHAP for attributing a *single* site's
prediction, with the efficiency axiom verified numerically → beeswarm and
dependence plots for global structure.

The importance comparison lands somewhere more useful than "they disagree": all
three agree on the *ranking*, but slope scores ~15% by gain and ~3% by
permutation. That gap is a diagnosis, not a contradiction — the model *uses*
slope but does not *need* it, because the other site-geometry features carry the
same information. It is also why permutation importance is unreliable under
correlated features.

The real payoff is stated plainly: XAI's value is not explaining a model that is
right, it is **catching one that is wrong for right-looking reasons** — a SHAP
direction that contradicts the physics is a bug you would otherwise have shipped.

### Module 4 — Physics-Informed Neural Networks

`04-pinn/`

The viewpoint shifts: the network's input becomes a **coordinate**, its output the
**field**, and training *is* solving the PDE. Solves `EI·w'' = -P(L-x)` for the
cantilever with **zero labelled data**.

Covers non-dimensionalisation (without it the residual sits 10⁵ above the
solution — the most common reason a hand-rolled PINN will not train), the residual
via `create_graph=True`, soft vs **hard** boundary conditions (`W = ξ²·N(ξ)`
satisfies the clamped end *exactly*), and the inverse problem.

The inverse problem is the headline: **recover `EI` from 8 noisy deflection
readings** as one extra `nn.Parameter`. It lands within 0.5%.

Part 5 is deliberately unflattering and important:

- As a forward solver a PINN is ~10⁵× slower than the exact solution, and less
  accurate. Do not sell it as a fast solver.
- A PINN is accurate **only in the quantities you put in the loss** — the
  constrained `w''` is excellent, the unconstrained `w''''` is garbage.
- No λ for the boundary penalty was best at everything; turning λ *up* — the
  usual advice — degraded the solution by three orders of magnitude.

### Module 5 — Operator Learning

`05-operator-learning/` · cantilever example after **Somdatta Goswami**

Module 4's PINN solved *one* load case. DeepONet learns the whole solution
operator `q(·) ↦ w(·)`: 1000 GRF-loaded finite element solves of a 2 m × 0.2 m
cantilever, 100 input sensors, 1314-node output field.

Covers the Chen & Chen (1995) operator UAT read literally as an architecture
(branch × trunk, combined by one `einsum`) and a naive-MLP baseline whose cost is
**measured**, not asserted — more than 10× the cost per step at full resolution, because it
re-encodes the input function at every query point.

The payoff is **the learned trunk basis**, examined properly. The trunk's leading
four directions land within ~12° of the four leading POD modes of the FE data — it
rediscovered the dominant deformation modes unaided. But the basis is
non-orthogonal (mean |cosine| ~0.4), ill-conditioned (~10⁵), and unordered past
those few, so unlike POD it cannot be truncated: 50 of 100 modes still leaves 35%
error where POD is under 1% by 10. A DeepONet is a learned spectral method with an
*unmanaged* basis — right subspace, wasteful representation — and the module
covers the three standard fixes, including POD-DeepONet.

Ends with the limits (800 FE solves are the real cost; out-of-distribution input
*functions* fail silently; the input grid is fixed — which is what FNO fixes) and
with physics-informed DeepONet, which combines Module 4's residual with Module 5's
operator structure.

Ported to PyTorch so the whole session uses one framework; the original JAX/Flax
version is at [DesignSafe-Training/deeponet](https://github.com/DesignSafe-Training/deeponet).

### Module 6 — Computer Vision at Scale

`06-clipseg/` · [details](https://github.com/DesignSafe-CI/training-ai/blob/main/06-clipseg/README.md)
· *Kooshan Amini and Jamie E. Padgett, Rice University*

Modules 1–5 built models of a cantilever beam — small, tabular, always checkable
against a closed form. This module is the other kind of work.

[**CLIPSeg-debris**](https://github.com/Way-Yuhao/CLIPSeg-debris) is a
text-prompted, 3-class segmentation model that maps **hurricane debris** (no /
low-density / high-density) in post-event aerial imagery, built on the CLIP
foundation model. The notebook runs it end to end:

1. **Official inference** on the published debris dataset (**PRJ-6029**).
2. A **regional debris map** for **[Hurricane Ian (2022)](https://storms.ngs.noaa.gov/storms/ian/index.html#14.4/26.45472/-81.94856)**
   on Estero Island, FL, from live **NOAA Emergency Response Imagery**.
3. **Scale-out** inference as a **GPU HPC job** via `dapi`.
4. **Fine-tuning** on TACC **Vista (GH200)**, tracked on Weights & Biases or DAPI.

What changes from the earlier modules: 75–1000 simulated cases become published
imagery and live rasters; a hand-written network with thousands of parameters
becomes a pre-trained model with ~150 M; minutes of CPU training becomes hours on a
GH200; and a closed-form check becomes hand-labelled masks.

The model is deliberately **not re-implemented** — the notebook loads and runs the
official published repository (**PRJ-6225**) as distributed. That is the lesson:
most applied ML is not writing architectures, it is getting someone else's model to
run correctly on your data, at scale, reproducibly.

The single best exercise here is the first one in *Your turn*: CLIPSeg is
**text-prompted**, so the class names are an *input*, not a fixed head. Change
"debris" to "rubble" and watch the map move — that is what separates a foundation
model from a fixed classifier.

> **Parts 3–4** need a TACC allocation; **Parts 1–2** run without one (GPU session
> recommended). Weights & Biases is optional — training can be tracked through DAPI.

---

## Going deeper

These modules are distilled from a full semester course. The long-form
treatments — with the proofs, the derivations, and the interactive demos — live in
the [**SciML course**](https://kks32-courses.github.io/sciml/), which also covers
Fourier Neural Operators, graph network simulators, SINDy, neural ODEs, function
encoders, and Bayesian SciML.

Related DesignSafe training repos:
[PINNs](https://github.com/DesignSafe-Training/pinn) ·
[DeepONet](https://github.com/DesignSafe-Training/deeponet) ·
[XAI](https://github.com/DesignSafe-Training/xai)

## Authors

- Krishna Kumar, University of Texas at Austin
- Kooshan Amini, Rice University
- Jamie Ellen Padgett, Rice University

Module 5's cantilever DeepONet example is due to Somdatta Goswami (Johns Hopkins).
