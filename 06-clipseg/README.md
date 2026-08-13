# Module 6 — Computer Vision at Scale: CLIPSeg-debris

**2026 SPARC · Day 3, Session 3b · ~20 min walkthrough · Parts 3–4 need a TACC allocation**

> **Authors:** Kooshan Amini and Jamie E. Padgett (Rice University), Padgett
> Research Group. Model: [CLIPSeg-debris](https://github.com/Way-Yuhao/CLIPSeg-debris)
> (Amini, Liu, Padgett, Balakrishnan & Veeraraghavan).

Modules 1–5 built models of a cantilever beam — small, tabular, self-contained, and
always checkable against ground truth. This module is the other kind of work: a
**foundation model** fine-tuned to segment hurricane debris in post-event aerial
imagery, running on real data at HPC scale.

## What changes from Modules 1–5

| | Modules 1–5 | This module |
| --- | --- | --- |
| Data | 75–1000 simulated cases | published imagery (PRJ-6029) + live NOAA rasters |
| Model | written from scratch, thousands of parameters | pre-trained foundation model, ~150 M parameters |
| Training | seconds to minutes on CPU | hours on a GH200 GPU |
| Ground truth | a closed-form solution | hand-labelled masks |
| The lesson | how the method works | how to **run someone else's model at scale** |

The model is deliberately **not re-implemented**. The notebook loads and runs the
official published repository (**PRJ-6225**) as distributed — which is the point.
Most applied ML is not writing architectures; it is getting a published model to
run correctly on your data, reproducibly, at scale.

## What runs where

Half the notebook needs an allocation. Know which half before you start.

| Part | What it does | Needs | In-session? |
| --- | --- | --- | --- |
| Setup | environment, paths, DAPI login, model, weights | DesignSafe login | yes, ~3 min |
| **1** | official `evaluate()` on the published dataset | GPU session recommended | yes |
| **2** | regional debris map — Estero Island FL, Hurricane Ian (2022) | GPU session recommended | yes, ~5 min |
| **3** | scale the same inference out as an HPC GPU job | **TACC allocation** | submit only |
| **4** | fine-tune on Vista (GH200) | **TACC allocation** | submit only |

Parts 1–2 run in a DesignSafe JupyterHub session with no allocation. Parts 3–4
submit real jobs; every cell ships its saved output from a completed run, so the
pattern is readable without running it.

## No separate exercise notebook — and why

Modules 2–5 ship `-exercise.ipynb` pairs. This one does not: blanking the code of a
production pipeline whose every call depends on external services mostly just
breaks it, and a broken exercise teaches nothing.

Instead the exercises are **inline**, in the notebook's *Your turn* section, as
variations to make on working code. The best of them is the first: CLIPSeg is
**text-prompted**, so the class names are an *input*, not a fixed head. Changing
"debris" to "rubble" or "storm debris" and watching the map move is the fastest way
to feel what separates a foundation model from a fixed classifier.

## Files

| Path | Role |
| --- | --- |
| `06-clipseg-debris.ipynb` | The notebook — headline calls and the story |
| `utils/clipseg_official.py` | Resolves and runs the official model (no vendoring) |
| `utils/regional.py` | Grid construction + NOAA ERI imagery fetch |
| `utils/dapi_helpers.py` | DesignSafe API, HPC and Vista job submission |
| `utils/finetune_data.py` | Prepares the debris dataset for fine-tuning |
| `utils/debris_common.py`, `utils/viz.py` | Shared helpers and plotting |
| `designsafe_job/job_infer.sh`, `run_inference.py` | Part 3 — HPC inference job |
| `designsafe_job/train_clipseg.sh` | Part 4 — Vista fine-tuning job |
| `weights/` | Model weights, downloaded at run time (not committed) |
| `requirements.txt` | Notebook dependencies |

Paths resolve automatically: the notebook sets
`REPO_ROOT = NB_DIR if (NB_DIR/"utils").exists() else NB_DIR.parent`, so it works
whether you launch it from this folder or from the repo root.

## Running it

On DesignSafe JupyterHub, prefer a **GPU** session
([user guide](https://www.designsafe-ci.org/user-guide/tools/jupyterhub/#launch-the-jupyter-lab-hpc-gpu)).
The first cell installs anything missing:

```bash
pip install -r requirements.txt
```

Weights come from Hugging Face or DesignSafe at run time, so `weights/` stays out
of git.

## Citations

Full BibTeX is in the notebook's closing section. In brief:

- **Paper** — Amini, Liu, Padgett, Balakrishnan & Veeraraghavan, *Debris
  segmentation using post-hurricane aerial imagery*, Computer-Aided Civil and
  Infrastructure Engineering 40(25), 2025. [doi:10.1111/mice.70033](https://doi.org/10.1111/mice.70033)
- **Dataset** — Hurricane-Induced Debris Segmentation Dataset Using Aerial Imagery,
  DesignSafe PRJ-6029. [doi:10.17603/DS2-JVPS-2N95](https://doi.org/10.17603/ds2-jvps-2n95)
- **Software** — CLIPSeg-debris, DesignSafe PRJ-6225.
  [doi:10.17603/DS2-YT43-HW55](https://doi.org/10.17603/ds2-yt43-hw55)

Background: CLIPSeg (Lüddecke & Ecker, CVPR 2022); CLIP (Radford et al., 2021);
NOAA Emergency Response Imagery; TACC Vista; DesignSafe / `dapi`.
