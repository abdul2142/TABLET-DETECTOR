---
title: Tablet Inspection Console
emoji: 💊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: mit
short_description: Educational tablet counting and visual defect detection
---

# Tablet Inspection Console — Python Mock Project

A **fully runnable educational prototype** for visual tablet defect detection. It demonstrates:

- **Tablet count** — how many tablets are in the photo (in packs vs. loose) with a normal/defective/review breakdown, also stamped on the result image
- **Count reliability check** — if the detected tablets vary wildly in size, the console says the count may be wrong instead of silently under-reporting
- **Blister packs and loose tablets** — the pack is located first, then every pocket inside it; loose tablets are found individually
- **Per-tablet verdicts** — every tablet is numbered and boxed (green = normal, red = defective, orange = needs review)
- **Close-ups of defective tablets** — each flagged tablet enlarged with its number, defect type, confidence and location
- **Defect type** — crack, chip, color anomaly or surface spots
- **Overall decision** — PASS / REJECT / MANUAL REVIEW, with a timestamped, copyable inspection report
- **Per-class accuracy, published** — the About tab reports recall for every defect type, not just an aggregate
- Synthetic training images (loose and in blister pockets) with augmentation: position, size, rotation, lighting, noise
- Shape, **silhouette**, edge and colour (HSV) features with a gradient boosting classifier

**Photo tips:** use a plain background that contrasts with the tablets or pack, light it evenly, and leave a gap between loose tablets — loose tablets that touch are detected as one.

> **Disclaimer:** This project is only for education and demonstration. It is not certified for pharmaceutical quality assurance, production inspection, or regulatory deployment. Accuracy figures are measured on synthetic images only.

## Accuracy

Measured on held-out synthetic crops (6,000 training samples, 20% held out):

| | |
|---|---|
| Normal vs. defective | **98.3%** |
| Defect type | **97.1%** |

Recall per class, and how often each defect is wrongly passed as normal — the
expensive error in an inspection tool:

| Class | Recall | Missed as normal |
|---|---|---|
| none (correctly passed) | 99.7% | — |
| crack | 90.5% | 3.4% |
| color anomaly | 99.3% | 0.0% |
| chip | 92.0% | 8.0% |
| surface spots | 95.9% | 0.7% |

Independently re-measured end-to-end on **fresh seeds** through the real
`detect` → `classify` pipeline, chip recall is **82.9%** — lower than the
held-out figure above, which is the honest number to plan around. Tablet
counting is exact on **99.7%** of generated photos (5 miscounts in 1,600).

### Two accuracy problems that were fixed

**Chips were being missed.** A chip bites a notch out of the tablet's outline
while barely changing its colour or texture, and the only shape feature in the
model was the crop's aspect ratio — so the defect was nearly invisible. For
tablets in blister pockets it was worse: the segmentation locked onto the clear
pocket rather than the tablet, hiding the notch entirely. Adding **silhouette
descriptors** (fill ratio, convex-hull solidity, and a rotation-invariant radial
profile, each measured at three background-distance thresholds) took
end-to-end chip recall from **58.1% to 82.9%**.

**Train/serve skew.** The model was trained on centred 128 px tiles but, at
inference, only ever sees crops that `detect` produced from a whole photo — a
different framing. Half the training set is now built by running the real
detector over generated photos and pairing each crop with its ground-truth
label, which closed that gap.

## 1. Create virtual environment

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Two floors are deliberate: **Gradio 6** (the console uses `gr.Textbox(buttons=…)`
and passes `theme`/`css` to `launch()`) and **Pillow 10.1**
(`ImageFont.load_default(size=…)`). Earlier versions fail at runtime.

## 3. Train the model

```bash
python train.py            # 6000 samples by default (~10 minutes)
```

Paths are resolved relative to this folder, so the command works from any directory. This creates:

- `artifacts/tablet_defect_model.joblib`
- `artifacts/tablet_defect_model.json` (metadata, test accuracy and per-class recall)
- `artifacts/previews/` with sample training images

## 4. Launch the application

```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

## 5. Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

133 tests. The vision and presentation layers need no model; the end-to-end
tests are marked `model` and skip automatically if `artifacts/` is empty:

```bash
pytest -m "not model"      # skip everything that needs a trained model
```

## 6. Test workflow

1. Click an **example photo** (blister packs or loose tablets), upload your own photo, or use the **Test image generator** (choose blister pack layout or number of loose tablets, set the defect chance, click **Generate & inspect**).
2. Read the verdict banner: PASS, REJECT (with the tablet numbers to remove) or MANUAL REVIEW.
3. If a **"Check the tablet count"** caution appears, the detection was internally inconsistent — retake the photo before trusting the result.
4. Check the detection overlay and the **Defective tablets — close-up** gallery (click a close-up to enlarge it).
5. See the **Tablet details** tab for every tablet, or copy the **Inspection report**.
6. For generated images, compare against the **Ground truth** box.

## Project files

- `image_utils.py` — synthetic image generation (loose tablets, blister packs), tablet/pack detection, detection-quality checks and feature extraction
- `train.py` — dataset creation, training, and evaluation
- `ui.py` — stylesheet, theme and HTML fragments (all markup, no vision logic)
- `app.py` — Gradio inspection console
- `tests/` — automated tests
- `requirements.txt` / `requirements-dev.txt` — packages

Responsibilities run one way: `app.py` imports the others, `image_utils.py`
holds no markup, and `ui.py` holds no vision logic and loads no model.

## Design & accessibility

- Verdicts are carried by **colour, icon and text together**, never colour alone.
- A count-reliability caution uses `role="alert"` so a screen reader announces it.
- Motion is brief and purposeful: panels rise in, the count counts up, bars grow
  from zero, and the "model ready" dot pulses. **A PASS may shimmer once; a
  REJECT never does** — it is information, not a celebration. Only ambient
  texture and that status dot loop.
- Everything switches off under `prefers-reduced-motion`, and a print stylesheet
  turns the page into a clean inspection record.
- Surfaces follow Gradio's light/dark theme through its own CSS variables.

Tests assert the animation vocabulary, the looping allowlist, the reduced-motion
block and the print rules, so the policy cannot erode silently.

## Limitations

- Trained and measured on **synthetic images only**. Performance on real
  photographs has not been validated.
- Loose tablets that touch are detected as one. The count-reliability check
  usually catches this, but it cannot find a tablet missed outright.
- Chip recall is the weakest class; treat a PASS as "no defect found", not "no
  defect present".
- Visual inspection cannot confirm a medicine is safe: it says nothing about
  expiry, authenticity or contamination.
