# `weights/` — CLIPSeg-debris model checkpoint

The notebook resolves the checkpoint in this order (Step 3):

1. **Local file** here — `weights/clipseg_debris_weight.ckpt` (preferred) or
   `weights/clipseg_debris.safetensors`.
2. **DesignSafe via DAPI** — set `DS_WEIGHTS_URI` (e.g.
   `/MyData/clipseg_debris/clipseg_debris_weight.ckpt`); the notebook pulls it with
   `ds.files.download`.
3. **Hugging Face fallback** (`YuhaoL/CLIPSeg-debris`):
   - <https://huggingface.co/YuhaoL/CLIPSeg-debris/resolve/main/clipseg_debris_weight.ckpt>
   - <https://huggingface.co/YuhaoL/CLIPSeg-debris/resolve/main/clipseg_debris.safetensors>

## Which file?

- **`clipseg_debris_weight.ckpt`** (~612 MB) — the Lightning checkpoint. Use this
  for **Part 1** (the official `src.eval.evaluate`, which loads via
  `trainer.predict(ckpt_path=...)`). It loads because the official repo is on
  `sys.path` (Step 2).
- **`clipseg_debris.safetensors`** (~603 MB) — tensors only; portable (no code
  dependency). Works for the regional model path (`clipseg_official.load_model`)
  and is what the local tests use.

> This folder is intentionally **not** committed with the large checkpoint — drop
> the file here, or let the notebook fetch it.

Architecture the checkpoint expects (handled in `utils/clipseg_official.py`):
`CLIPDensePredT(version="ViT-B/16", reduce_dim=64, complex_trans_conv=True,
extract_layers=(3,7,9), fix_shift=False)`; the trained decoder is loaded with
`strict=False` after stripping the Lightning `model.` prefix, and the CLIP
ViT-B/16 backbone is fetched by `clip.load`.
