# Deploying to Streamlit Community Cloud

The console has two interchangeable front-ends over one inspection pipeline:

| Entry point | Framework | Use |
|---|---|---|
| `streamlit_app.py` | Streamlit | Streamlit Community Cloud |
| `app.py` | Gradio | local use, Hugging Face Spaces, Docker |

Both import the same `app.py` logic, so a verdict can never differ between them.

## 1. This folder must be the repository root

Streamlit Community Cloud looks for `requirements.txt` in the **root of the
repository**. Initialise git *inside this folder* rather than its parent:

```bash
cd "TABLET DEFECT DETECTOR"
git init
git add -A
git commit -m "Tablet Inspection Console"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

If you instead push the parent folder, set the main file path to
`TABLET DEFECT DETECTOR/streamlit_app.py` **and** copy `requirements.txt` to the
repository root, or the build will install nothing and fail on `import numpy`.

## 2. Create the app

On <https://share.streamlit.io> → **New app**:

| Field | Value |
|---|---|
| Repository | `<you>/<repo>` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python version | 3.11 or 3.12 |

Nothing else needs configuring. `.streamlit/config.toml` pins the light theme
the shared stylesheet is written for and caps uploads at 25 MB.

## 3. The model must be committed

`artifacts/tablet_defect_model.joblib` is **1.0 MB** and is deliberately *not*
gitignored — the console loads it at startup and a deployment should never train
on boot. Confirm it is tracked before pushing:

```bash
git ls-files artifacts/
```

Expected: the `.joblib`, the `.json` metadata, and the three example photos.
`artifacts/previews/` is gitignored; it is training output and is not needed.

### Protect it from line-ending normalisation

`.gitattributes` currently contains only `* text=auto`. Git detects binary
content and leaves it alone, so this is safe in practice — but if you want
certainty for the model and the example PNGs, add:

```gitattributes
*.joblib binary
*.png binary
```

A normalised `.joblib` is a corrupted model, and the failure appears only at
runtime as an unpickling error.

## 4. What the deployment already handles

- **Oversized uploads** — downscaled to 1600 px on the longest edge before
  inspection. Detection takes 3.0 s at 11.3 MP versus 0.08 s at 0.3 MP and finds
  the same tablets either way, so a 48 MP phone photo cannot stall the worker.
  Override with the `TIC_MAX_EDGE` environment variable.
- **Read-only filesystem** — if `artifacts/` cannot be written, the example
  photos are skipped with a log line instead of crashing at import.
- **Missing model** — the header chip reads "No model — run train.py" and the
  About tab says so, rather than erroring on every request.

## 5. Local check before pushing

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py          # http://localhost:8501
pytest                                  # 139 tests
```

## Caveat

Community Cloud runs on a shared container with roughly 1 GB of RAM. The model
is small and inference is CPU-only, but both Gradio and Streamlit are installed
because the two front-ends share one pipeline. If image size becomes a problem,
drop `gradio>=6.0,<7` from `requirements.txt` and replace the `import app` in
`streamlit_app.py` with a copy of the functions it uses — at the cost of the two
consoles being able to drift apart.
