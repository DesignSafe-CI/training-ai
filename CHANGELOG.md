# Changelog

All notable changes to this repository.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Four new scientific-ML modules**, threaded
  through a single cantilever beam so each answers a question the previous one
  cannot. All CPU-only; no allocation or GPU required.

  - `02-mlp/` — **Multi-Layer Perceptrons.** Reuses Module 1's 75-run OpenSees
    sweep. A linear model on raw features scores `R² = 0.984` yet predicts negative
    fundamental periods; an MLP on the same features recovers the physics without
    being given logarithms. Covers linear-layer collapse, the Universal
    Approximation Theorem and its caveats, width vs depth, reverse-mode AD for
    design sensitivities (`∂T/∂L`, checked against the analytic `1.5·T/L`),
    overfitting, and a closing deflection-field fit that fails at the support.
  - `03-xai/` — **Explainable AI.** Distilled from
    [DesignSafe-Training/xai](https://github.com/DesignSafe-Training/xai) (59 cells
    → 40). Lateral-spreading classification on 7,291 field observations: decision
    tree, XGBoost, three definitions of feature importance compared, and SHAP
    with the efficiency axiom verified numerically.
  - `04-pinn/` — **Physics-Informed Neural Networks.** Solves `EI·w'' = -P(L-x)`
    for the cantilever with zero labelled data. Covers non-dimensionalisation,
    autograd residuals, soft vs hard boundary conditions (`W = ξ²·N(ξ)` satisfies
    the clamped end exactly), and recovery of `EI` from 8 noisy readings to within
    0.5%. Includes a deliberately unflattering section on when PINNs are the wrong
    tool.
  - `05-operator-learning/` — **DeepONet.** Learns the operator `q(·) ↦ w(·)` on
    1000 GRF-loaded FE solves of a 2 m × 0.2 m cantilever. Covers the Chen & Chen
    (1995) operator UAT as an architecture, a naive-MLP baseline held to the same
    budget, a measured cost comparison, the learned trunk basis and its
    conditioning, zero-shot prediction, and physics-informed DeepONet. Cantilever example after Somdatta
    Goswami.

- Modules 2–5 each ship an **exercise notebook** (key code blanked with `# TODO`)
  alongside the solution, matching the convention in the other DesignSafe training
  repos. Both are generated from one source so they cannot drift apart. Modules 1
  and 6 ship a single notebook each — Module 1 submits a real HPC job, and blanking
  Module 6's production pipeline would just break it, so its exercises are inline
  (see below).
- `06-clipseg/` **is now Module 6** rather than a standalone chapter.
  Its notebook, `utils/`, `designsafe_job/`, `weights/`, and requirements moved into
  the module folder. Because the notebook resolves
  `REPO_ROOT = NB_DIR if (NB_DIR/"utils").exists() else NB_DIR.parent`, the move
  needed **no code changes** — which matters, because its outputs come from a run
  against DesignSafe + DAPI + a GPU that cannot be reproduced in CI. All 19 code
  cells and 46 outputs were preserved byte-for-byte; only markdown was edited.
- A **"Your turn"** section in Module 6, replacing the exercise notebook it cannot
  usefully have: prompt rewording (CLIPSeg is text-prompted, so class names are an
  *input* — the fastest way to feel what a foundation model is), changing region and
  grid size, reading the failure cases, and the with-an-allocation path.
- `PREVIEW.md` — how to build and serve the book locally, verified from a clean venv
  built only from `requirements.txt` (no torch or scientific stack needed, because
  the notebooks ship executed).
- `01-opensees-ml/figs/period-power-law.png` — a three-panel explainer for Module 1,
  generated from the real 75-run sweep: **A** the cantilever with $M$, $L$, $E$, $I$
  and $T = 2\pi\sqrt{ML^3/3EI}$; **B** the same data in raw units, curved, with no
  straight line able to fit it; **C** the same data in log space, straight and
  parallel, with the measured slope annotated. The slope measures exactly `1.500000`
  from the data. Nothing existing in the repos or the sciml slide decks showed the
  log-linearisation, so it was drawn for this.
- `02-mlp/cantilever_sweep.csv` and `02-mlp/make_sweep_csv.py` — the Module 1
  dataset, reproducible offline. The beam is elastic, so `period` and
  `top_disp_at_100kip` have closed forms; the script regenerates the table without
  OpenSees or a TACC allocation. Verified to return the same
  `[0.5, 1.5, -0.5]`/`R² = 1.0` as the HPC job.
- `requirements-sciml.txt` — dependencies for Modules 2–5 (torch, scikit-learn,
  xgboost, shap, scipy).
- Per-module `README.md` files documenting the pedagogical arc, the measured
  results, and the caveats.
- `CHANGELOG.md` (this file).

### Changed

- **Module folders renumbered so the folder number equals the module number.**
  `opensees_ml/` was unnumbered while the new folders started at `01-`, which made
  `01-mlp/` Module *2* — an off-by-one waiting to confuse people:

  | Was | Now | Module |
  | --- | --- | --- |
  | `opensees_ml/` | `01-opensees-ml/` | 1. Regression |
  | `01-mlp/` | `02-mlp/` | 2. MLP |
  | `02-xai/` | `03-xai/` | 3. XAI |
  | `03-pinn/` | `04-pinn/` | 4. PINNs |
  | `04-operator-learning/` | `05-operator-learning/` | 5. DeepONet |
  | *(repo root)* | `06-clipseg/` | 6. CLIPSeg |

  Notebook filenames were renumbered to match, so the flat Community Data folder
  sorts in teaching order:

  | Was | Now |
  | --- | --- |
  | `DS_OpenSees_ML_Example.ipynb` | `01-opensees-ml-regression.ipynb` |
  | `DS_OpenSees_ML_Workflow_DAG.ipynb` | `01-opensees-ml-workflow-dag.ipynb` |
  | `01-mlp-cantilever.ipynb` | `02-mlp-cantilever.ipynb` |
  | `02-xai-lateral-spreading.ipynb` | `03-xai-lateral-spreading.ipynb` |
  | `03-pinn-cantilever.ipynb` | `04-pinn-cantilever.ipynb` |
  | `04-deeponet-cantilever.ipynb` | `05-deeponet-cantilever.ipynb` |
  | `CLIPSeg_debris_DesignSafe_NHERI2026.ipynb` | `06-clipseg-debris.ipynb` |

  Output-file prefixes (`opensees_ml_model.json`, `opensees_ml_report.txt`) are
  unchanged — they are artefact names produced by the HPC job, not paths.
- `_toc.yml` reorganised into three parts that mirror the arc —
  *Data-driven surrogates* (1–3), *Physics-driven models* (4–5), *Applied at
  scale* (6) — with exercise notebooks as sub-sections.
- `01-opensees-ml-workflow-dag.ipynb` is now **published** as an optional
  sub-section of Module 1 rather than excluded from the book. It runs the identical
  75-run study as an explicit two-job DAG (`sweep → train`) via DesignSafe's
  workflow service, so a retrain costs one core instead of a 48-core resweep — real
  content that was invisible while it was excluded. Its title now says plainly that
  it is optional and needs the main notebook's staged inputs first, and its internal
  link to the (renamed) main notebook was fixed.
- Module 1's notebook intro was rewritten: the period equation and its log-linear
  form now render as LaTeX (they were a plain fenced code block), the explainer
  figure is embedded, and a **"What came back"** section reports the recovered
  coefficients — `+0.500000`, `+1.500000`, `−0.500000`, `R² = 1.0000` — next to the
  exact exponents, with what each one means physically. Two bugs were fixed in the
  same pass: its badge still linked to the pre-rename `DS_OpenSees_ML_Example.ipynb`,
  and an earlier edit of mine had truncated the intro blockquote to a single line.
  All 11 code cells and 54 outputs from the verified run were left untouched.
- Module 6's notebook had **two competing numbering schemes**: sections `0.`–`8.`
  alongside Parts 1–4, so "Part 1" lived under heading "4.". The section numbers
  were dropped so the Parts carry the numbering, and the `0a.`/`7c.`-style
  sub-numbering went with them.
- Module 6 gained the this training module header, DesignSafe/Colab badges, a
  **"what runs where"** table (which parts need a GPU, which need an allocation,
  which run in-session), and a *Where we are* section contrasting it with Modules
  1–5. Authorship (Kooshan Amini, Jamie E. Padgett) is stated up front.
- The three-hour agenda was rebuilt for six modules with explicit per-module
  minutes, and states plainly that six *hands-on* modules do not fit: Module 6 is
  a guided walkthrough, and `⏱ optional` cells in Modules 2–5 are the slack.
- `05-operator-learning/cantilever_beam_deflection.mat` (20 MB) is now **committed**,
  so a clone or the one-click DesignSafe git-pull yields a fully working Module 5
  with no run-time network dependency.
- `README.md` rewritten as the landing page: the six-module table, a
  three-hour agenda, and a DesignSafe/Colab link table pointing at
  `CommunityData/Training/2026-SPARC/Day3/3b-SciML/` (PRJ-1305). The CLIPSeg-debris material
  is retained as its own section.
- `_toc.yml` restructured into two parts — the the modules (with exercise
  notebooks as sub-sections) and the CLIPSeg-debris chapter.
- Module 5 ported from JAX/Flax to PyTorch so Modules 2, 4, and 5 share one
  framework, and so it trains in under a minute on CPU instead of requiring
  200,000 epochs. The original JAX version remains at
  [DesignSafe-Training/deeponet](https://github.com/DesignSafe-Training/deeponet).

### Removed

- `.github/workflows/deploy.yml` (the `deploy-docs` workflow). It built with
  `npx myst build --html`, but this repository is a Jupyter Book (`_config.yml` +
  `_toc.yml`) and carries no `myst.yml`, so the build wrote nothing and
  `upload-pages-artifact` died on `tar: _build/html: Cannot open`. Because it
  shared the repo-wide `concurrency: pages` group with `deploy-book`, every push
  to `main` started two runs that then cancelled or raced each other. Deploys now
  come only from `publish.yml`, which archives `./out/_build/html` and hands it to
  `actions/deploy-pages` — artifact-based, so no `gh-pages` branch is written.

### Notes

Every solution notebook was executed end to end before commit, and the committed
outputs are from those runs. Executing them corrected four claims that the prose
had asserted but the numbers did not support:

- The MLP training loop was unstable at `lr = 1e-2` (546 loss spikes, 6643×
  overshoot). Cosine annealing at `lr = 5e-3` removes them entirely.
- The width sweep does not "flatten" — error drops ~8× to width 4, then
  seed-to-seed scatter (1.3–1.5×) rivals the effect of an 8× width change. It is
  now averaged over three seeds and plotted with a spread band.
- The overfitting demonstration did not show what the prose claimed, twice over.
  Measured as a 2×2 (noiseless / 2% noise × 1,217 / 198,657 parameters): the
  oversized network *does* interpolate the training set (loss `1.6e-08` clean,
  `4.1e-11` noisy), but its **test** error is only ~6× worse when clean and
  statistically identical under noise. The train/test ratio reads `7.6e+07` while
  predictions are as good as the small network's. The module now teaches that —
  judge on test error, not the gap — and identifies noise, not capacity, as the
  binding constraint on this problem.
- Raising the PINN's boundary-condition weight `λ_BC` does *not* tighten the
  boundary condition; the solution error grows monotonically and is ~3 orders of
  magnitude worse at `λ = 10⁴`. The notebook now says so.

- The DeepONet's learned basis is **not** low-rank in the way the draft claimed.
  Measured: 82 of 100 trunk modes are needed for 90% of the total contribution,
  and truncating to the top 50 still leaves 35% error. A POD of the same fields
  needs 6 modes for 99% of the energy. Module 5 now teaches the real finding —
  the trunk's leading four directions sit within ~12° of the four leading POD
  modes (it *did* find the right subspace) but the basis is non-orthogonal
  (mean |cosine| ~0.4), ill-conditioned (~10⁵), and unordered past the leading
  few, so unlike POD it carries no Eckart–Young truncation guarantee. The three
  standard remedies are covered.
- All three feature-importance definitions in Module 3 turned out to *agree* on
  ranking, so the "they disagree" framing was replaced with the sharper
  finding: they disagree on **magnitude** — slope is ~15% by gain and
  ~3% by permutation — which is the signature of redundancy with the other site
  geometry features, not of an ill-posed question.

Also fixed before commit: binary sklearn classifiers return SHAP values with a
per-class axis that binary XGBoost does not, which crashed the model-comparison
cell; the lateral-spreading dataset has four predictors, not five, after dropping
`Test ID` and `Elevation`; the naive-MLP operator baseline was >10x more expensive
per step than the DeepONet at full resolution (now measured explicitly in the
notebook as the architecture's justification, and trained on sampled query points
so the module fits its slot); and `%matplotlib inline` was missing, so no figure
was being captured into any notebook and the published book would have had no
plots at all.
