# NHERI Computational Symposium AI Training

[![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-blue?logo=github)](https://designsafe-ci.github.io/training-ai)
[![Jupyter Book](https://img.shields.io/badge/Powered%20by-Jupyter%20Book-orange)](https://jupyterbook.org)

Hands-on training for the **NHERI Computational Symposium** — running modern AI
workflows on **[DesignSafe](https://www.designsafe-ci.org/)** with the DesignSafe API
([`dapi`](https://designsafe-ci.github.io/dapi/)).

## CLIPSeg-debris on DesignSafe

[**CLIPSeg-debris**](https://github.com/Way-Yuhao/CLIPSeg-debris) is a text-prompted,
3-class segmentation model that maps **hurricane debris** (no / low-density /
high-density) in post-event aerial imagery. The chapter notebook runs it end-to-end on
DesignSafe:

1. **Official inference** on the published debris dataset (**PRJ-6029**).
2. A **regional debris map** for **[Hurricane Ian (2022)](https://storms.ngs.noaa.gov/storms/ian/index.html#14.4/26.45472/-81.94856)**
   on Estero Island, FL, from **NOAA Emergency Response Imagery**.
3. **Scale-out** inference as a **GPU HPC job** via `dapi`.
4. **Fine-tuning** on TACC **Vista (GH200)**, tracked on Weights & Biases or DAPI.

### Quick start on DesignSafe (one click)

[![Open in DesignSafe](https://img.shields.io/badge/Open%20in%20DesignSafe-clone%20%2B%20open-006FBA?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.designsafe-ci.org/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FDesignSafe-CI%2Ftraining-ai&branch=main&targetpath=MyData%2Ftraining-ai&urlpath=lab%2Ftree%2FMyData%2Ftraining-ai%2FCLIPSeg_debris_DesignSafe_NHERI2026.ipynb)

Clicking the button (uses `nbgitpuller`) will:
1. Open your **DesignSafe JupyterHub** session (sign in if you aren't already).
2. Clone — or fast-forward — this repo into **`~/MyData/training-ai/`** on your account.
3. Open JupyterLab directly on the chapter notebook, ready to run.

Re-click any time to pull the latest version of the code.

<!-- ### Or via the curated DesignSafe data path

[![Try on DesignSafe](https://raw.githubusercontent.com/DesignSafe-CI/training-ai/main/DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/clipseg-debris/CLIPSeg_debris_DesignSafe_NHERI2026.ipynb)

This badge opens the chapter from DesignSafe's **Community Data ▸ Training** mount
once the folder has been placed there by the DesignSafe team (a one-time curation step). -->

> The chapter notebook and its `utils/` + `designsafe_job/` helpers run top to bottom in
> a DesignSafe JupyterHub session. **Parts 3–4** (HPC inference and Vista fine-tuning)
> need a TACC allocation; **Parts 1–2** run without one. Weights & Biases is optional —
> training can be tracked through DAPI instead.

## Authors
- Krishna Kumar, University of Texas at Austin
- Kooshan Amini, Rice University
- Jamie Ellen Padgett, Rice University