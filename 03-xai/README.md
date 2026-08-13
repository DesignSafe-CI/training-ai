# Module 3 — Explainable AI

**2026 SPARC · Day 3, Session 3b · ~25 min · CPU only**

Module 2 traded three interpretable exponents (`0.5`, `1.5`, `-0.5` — the physics)
for a few thousand opaque weights. On a cantilever we could check the answer
against a closed form. On real data there is nothing to check against, and "the
network said so" is not defensible in a report.

This module buys the interpretation back.

## The problem

**Lateral spreading** — down-slope movement of liquefied ground during an
earthquake, which wrecks pipelines, bridge abutments, and quay walls. 7,291 field
observations, each labelled *did* or *did not* spread.

Four predictors, after dropping `Test ID` (an index — leaving it in is a classic
leakage bug) and `Elevation` (largely redundant with the slope and free-face
geometry):

| Feature | Physical role |
| --- | --- |
| `GWD (m)` | Groundwater depth — liquefaction needs saturated soil |
| `L (km)` | Distance to the free face — spreading needs somewhere to go |
| `Slope (%)` | The driving force |
| `PGA (g)` | The demand — how hard the ground shook |

Class balance is 58/42, so plain accuracy is a fair headline metric.

## The arc

1. **Decision tree**, depth 3 — the entire model fits on one page. This is the
   baseline XAI is trying to get *back to*: the model is its own explanation.
2. **The same overfitting curve as Module 2**, in a different model class — deeper
   trees hit 100% training accuracy while validation accuracy decays. Overfitting
   is about capacity versus data, not about neural networks.
3. **XGBoost**, 400 trees — gradient descent in function space. A few points more
   accuracy, and now a black box.
4. **Three definitions of "feature importance"** — weight, gain, permutation —
   computed side by side (see below).
5. **SHAP** — Shapley values from cooperative game theory, the unique attribution
   satisfying efficiency, symmetry, and dummy. The notebook **verifies efficiency
   numerically**: the attributions sum to `f(x) - E[f]` to machine precision,
   which is what makes SHAP defensible where "gain" is not.
6. **Waterfall plots** for the most-confident positive and negative sites —
   attributing a *single* prediction, which is the question an engineer assessing
   a specific site actually has.
7. **Beeswarm and dependence plots** for global structure that keeps the per-site
   detail.

## What the importance comparison actually shows

Worth knowing before you teach it, because it is not the usual "the definitions
disagree" story. Measured on this data:

| Feature | weight | gain | permutation |
| --- | --- | --- | --- |
| GWD (m) | 0.229 | 0.255 | 0.231 |
| L (km) | 0.278 | 0.274 | 0.335 |
| Slope (%) | 0.167 | 0.143 | **0.026** |
| PGA (g) | 0.326 | 0.328 | 0.408 |

All three produce the **same ranking** — PGA > L > GWD > Slope. That is genuine
reassurance and worth reporting when it happens.

The disagreement is in *magnitude*: **slope is worth ~15% by gain and ~3% by
permutation**, a factor of five or six. The tree splits on slope often and
collects impurity reduction when it does, yet shuffling the slope column barely
dents accuracy.

That is a diagnosis, not a contradiction. Gain answers "does the model use this?"
(yes); permutation answers "does the model *need* this?" (no — when slope is
destroyed, the other features cover for it). Which is what you would expect
physically: slope, elevation, and free-face distance all describe the same site
geometry, and `Elevation` was dropped for that very reason.

Two rules follow, and the notebook states both: a gain/permutation mismatch means
**redundancy**, so do not conclude "slope doesn't matter for lateral spreading";
and permutation importance is unreliable under correlated features precisely
because it destroys one feature while its correlates stay intact.

## The point of the module

Stated plainly in the notebook: XAI's value is **not** explaining a model that is
right. It is catching one that is wrong for right-looking reasons.

Each SHAP direction has a physical prediction attached — deeper groundwater should
push *away* from spreading, steeper slope *toward* it. If a direction comes out
backwards you have found either a data problem (sign convention, units, a leaking
column) or a real interaction captured through a proxy. Either is worth chasing
before the model informs a decision.

Closing caveat, which sets up Modules 4–5: SHAP tells you what the **model** did,
not what the **ground** does. Correlation in the training data becomes attribution
in the explanation. That is why the next two modules build physics into the model
instead of interrogating a flexible fit after the fact.

## Files

| File | Role |
| --- | --- |
| `03-xai-lateral-spreading.ipynb` | Solution notebook |
| `03-xai-lateral-spreading-exercise.ipynb` | Same, with the key code blanked (`# TODO`) |
| `RF_YN_Model3.csv` | 7,291 lateral-spreading observations |
| `figs/` | Liquefaction, decision tree, XGBoost, and SHAP schematics |

## A note on SHAP output shapes

A binary **sklearn** classifier returns SHAP values of shape
`(n_rows, n_features, n_classes)` — one attribution per class. Binary **XGBoost**
returns `(n_rows, n_features)`, because it models a single log-odds output. The
notebook's `mean_abs_shap` helper handles both so the two models are compared on
the same quantity. This bites people; it is called out in the code comments.

## Going deeper

- [Full DesignSafe XAI training](https://github.com/DesignSafe-Training/xai) — the
  longer version, with the Gini derivations worked out
- [SciML — classification and decision trees](https://kks32-courses.github.io/sciml/00-mlp/00a-classification.html)
- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*
- Molnar, [*Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/)
