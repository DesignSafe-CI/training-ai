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

[![Try on DesignSafe](https://raw.githubusercontent.com/DesignSafe-CI/training-ai/main/DesignSafe-Badge.svg)](https://jupyter.designsafe-ci.org/hub/user-redirect/lab/tree/CommunityData/Training/clipseg-debris/CLIPSeg_debris_DesignSafe_NHERI2026.ipynb)

> The chapter notebook and its `utils/` + `designsafe_job/` helpers run top to bottom in
> a DesignSafe JupyterHub session. **Parts 3–4** (HPC inference and Vista fine-tuning)
> need a TACC allocation; **Parts 1–2** run without one. Weights & Biases is optional —
> training can be tracked through DAPI instead.

## Authors
- Krishna Kumar, University of Texas at Austin
- Kooshan Amini, Rice University
- Jamie Ellen Padgett, Rice University