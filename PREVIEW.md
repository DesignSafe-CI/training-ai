# Previewing the book locally

The notebooks are committed **already executed**, and `_config.yml` sets
`execute_notebooks: "off"`, so building the book only renders saved outputs. That
means **you do not need torch, scikit-learn, xgboost, shap, or scipy to preview** —
only Jupyter Book.

## One-time setup

```bash
cd ~/dev/DesignSafe-Training/training-ai
python3 -m venv .venv-book
.venv-book/bin/pip install -r requirements.txt
```

## Build and open

```bash
.venv-book/bin/jupyter-book build .
open _build/html/index.html          # macOS
```

`_build/` is already gitignored.

## Rebuild after edits

Jupyter Book caches aggressively. If a change does not show up:

```bash
.venv-book/bin/jupyter-book build . --all
```

## Serve instead of file://

Some browsers restrict `file://` pages. To serve over HTTP:

```bash
python3 -m http.server 8000 --directory _build/html
# then open http://localhost:8000
```

## Check for problems

A clean build reports no warnings. To see only those:

```bash
.venv-book/bin/jupyter-book build . 2>&1 | grep WARNING
```

## Re-running the notebooks (optional)

Only needed if you change notebook *code* and want fresh outputs. This does need
the scientific stack:

```bash
python3 -m venv .venv-run
.venv-run/bin/pip install -r requirements-sciml.txt nbconvert ipykernel

for nb in 02-mlp/02-mlp-cantilever \
          03-xai/03-xai-lateral-spreading \
          04-pinn/04-pinn-cantilever \
          05-operator-learning/05-deeponet-cantilever; do
  ( cd "$(dirname $nb)" && \
    ../.venv-run/bin/python -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 "$(basename $nb).ipynb" )
done
```

All four are CPU-only and take a couple of minutes each. Keep
`%matplotlib inline` in the setup cells — without it no figure is captured and the
built book has no plots.

Leave the `-exercise.ipynb` notebooks unexecuted; they ship blank by design.
