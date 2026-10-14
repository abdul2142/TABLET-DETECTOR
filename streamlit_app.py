"""Streamlit front-end for the Tablet Inspection Console.

This is an **additive** entry point. Nothing in `app.py`, `ui.py`,
`image_utils.py` or `train.py` is modified: the whole inspection pipeline --
detection, classification, verdicts, positions, warnings and the report -- is
imported from `app.py` and reused as-is, so the Streamlit and Gradio consoles
can never disagree about a result.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    main file path = tablet_defect_mock/streamlit_app.py

Two notes on reuse:

* Importing `app` pulls in Gradio. That is deliberate -- duplicating the
  verdict logic in a defect-detection tool is a correctness hazard, and Gradio
  is already a declared dependency. `app.py` guards `demo.launch()` behind
  `__main__`, so importing it never starts a second server.
* `ui.CSS` is written against Gradio's CSS variables. The shim below defines
  those names so the shared markup renders identically here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

import app
import ui
from image_utils import make_blister_pack_image, make_multi_tablet_image

st.set_page_config(
    page_title="Tablet Inspection Console",
    page_icon="\U0001f48a",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---------------------------------------------------------------
# ui.CSS targets Gradio's theme variables. Supplying them here lets the shared
# HTML fragments (count card, verdict banner, tables, metric tiles) render
# unchanged instead of losing their borders and surfaces.
CSS_SHIM = """
<style>
:root {
  --block-background-fill: #ffffff;
  --border-color-primary: #e2e8f0;
  --body-text-color: #0f172a;
  --body-text-color-subdued: #64748b;
  --font-mono: ui-monospace, "JetBrains Mono", Consolas, monospace;
}
/* Streamlit has no .gradio-container, so re-declare the brand tokens here. */
:root, .stApp {
  --tic-brand: #2563eb;
  --tic-brand-2: #0ea5e9;
  --tic-accent: #06b6d4;
  --tic-pass: #16a34a;
  --tic-reject: #dc2626;
  --tic-review: #ea8a00;
  --tic-radius: 14px;
  --tic-ease: cubic-bezier(.22, 1, .36, 1);
  --tic-shadow: 0 1px 2px rgba(15, 23, 42, .06), 0 8px 22px -14px rgba(37, 99, 235, .3);
  --tic-shadow-lg: 0 2px 6px rgba(15, 23, 42, .07), 0 20px 38px -18px rgba(37, 99, 235, .4);
}
.block-container { max-width: 1340px; padding-top: 1.6rem; }
[data-testid="stSidebar"] { border-right: 1px solid var(--border-color-primary); }
#MainMenu, [data-testid="stDecoration"] { visibility: hidden; }
/* The shared stylesheet scopes a few rules to .gradio-container; mirror them. */
.stApp button { transition: transform .18s var(--tic-ease), box-shadow .22s ease; }
.stApp [data-testid="stBaseButton-primary"]:hover { transform: translateY(-2px); box-shadow: var(--tic-shadow-lg); }
</style>
"""

st.markdown(CSS_SHIM, unsafe_allow_html=True)
st.markdown(f"<style>{ui.CSS}</style>", unsafe_allow_html=True)


def render(markup: str) -> None:
    """Render a shared HTML fragment, flattened for Streamlit's markdown."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


# --- State -----------------------------------------------------------------

for key, value in {"image": None, "truth": "", "results": None, "error": ""}.items():
    st.session_state.setdefault(key, value)


def set_image(image: Image.Image, truth: str = "") -> None:
    """Adopt a new photo and invalidate any previous result."""
    st.session_state.image = image
    st.session_state.truth = truth
    st.session_state.results = None
    st.session_state.error = ""


# --- Header ----------------------------------------------------------------

render(ui.header(
    "Tablet Inspection ", "Console",
    "Automated tablet counting and visual defect detection · loose tablets and blister packs",
    [("Model ready" if app.MODEL_PATH.exists() else "No model — run train.py",
      app.MODEL_PATH.exists()),
     ("Educational prototype · not for clinical use", False)],
))


# --- Sidebar: choose a photo ----------------------------------------------

with st.sidebar:
    render(ui.step(1, "Load a photo", "upload or generate"))

    uploaded = st.file_uploader("Inspection photo", type=["png", "jpg", "jpeg", "bmp", "webp"],
                                label_visibility="collapsed")
    if uploaded is not None:
        try:
            set_image(Image.open(uploaded).convert("RGB"))
        except OSError:
            st.error("That file could not be read as an image.")

    st.divider()
    st.caption("Example photos")
    examples = {
        "Blister pack 2 × 5": "blister_pack_2x5.png",
        "Blister pack 3 × 4": "blister_pack_3x4.png",
        "Loose tablets": "loose_tablets.png",
    }
    available = {label: Path(path) for label, path in
                 zip(examples, (Path(p) for p in app.example_images()))}
    for label, path in available.items():
        if st.button(label, width="stretch", key=f"ex_{label}"):
            set_image(Image.open(path).convert("RGB"))

    st.divider()
    with st.expander("Test image generator", expanded=False):
        kind = st.radio("Kind", ["Blister pack", "Loose tablets"], horizontal=True,
                        label_visibility="collapsed")
        if kind == "Blister pack":
            layout = st.selectbox("Pack layout (columns × rows)", list(app.PACK_LAYOUTS))
            count = 6
        else:
            layout = "2 × 5"
            count = st.slider("Number of tablets", 2, 12, 6)
        defect_rate = st.slider("Defect chance (%)", 0, 100, 30, step=5)
        if st.button("Generate & inspect", type="primary", width="stretch"):
            image, truth = app.generate_demo(kind, layout, count, defect_rate)
            defects = [f"#{i} {t}" for i, t in enumerate(truth, start=1) if t != "none"]
            set_image(image, f"{len(truth)} tablets · defective: "
                             + (", ".join(defects) if defects else "none"))

    st.divider()
    st.caption("Use a plain background that contrasts with the tablets, light it evenly, "
               "and leave a gap between loose tablets — touching tablets are counted as one.")


# --- Inspection ------------------------------------------------------------

image = st.session_state.image

# Inspect once per photo, not on every widget interaction: a rerun is cheap but
# detection plus classification is not.
if image is not None and st.session_state.results is None and not st.session_state.error:
    with st.spinner("Detecting and classifying tablets…"):
        try:
            st.session_state.results = app.inspect(image)
        except Exception as exc:  # gr.Error, and anything the pipeline raises
            st.session_state.error = str(exc) or "Inspection failed."

left, right = st.columns([5, 6], gap="large")

with left:
    render(ui.step(2, "Photo"))
    if image is None:
        st.info("Upload a photo, pick an example, or generate a test image to begin.",
                icon=":material/image:")
    else:
        st.image(image, width="stretch")
        if st.session_state.truth:
            st.caption(f"Ground truth — {st.session_state.truth}")
        if max(image.size) > app.MAX_EDGE:
            st.caption(f"Downscaled to {app.MAX_EDGE}px on the longest edge before inspection.")
        if st.button("Inspect again", width="stretch"):
            st.session_state.results = None
            st.session_state.error = ""
            st.rerun()

with right:
    render(ui.step(3, "Results"))
    if st.session_state.error:
        st.error(st.session_state.error, icon=":material/error:")
        render(app.summary_html())
    elif st.session_state.results:
        summary, annotated, *_ = st.session_state.results
        render(summary)
        st.image(annotated, caption="Detection overlay", width="stretch")
    else:
        render(app.summary_html())

if st.session_state.results:
    summary, annotated, closeups, table, report = st.session_state.results

    render(ui.step(4, "Flagged tablets", "close-ups of defective and uncertain tablets"))
    if not closeups:
        st.success("No tablet was flagged. Nothing to review.", icon=":material/check_circle:")
    else:
        for row_start in range(0, len(closeups), 6):
            row = closeups[row_start:row_start + 6]
            for column, (crop, caption) in zip(st.columns(6), row):
                column.image(crop, caption=caption, width="stretch")

    render(ui.step(5, "Details & report"))
    tab_details, tab_report, tab_model = st.tabs(
        ["Tablet details", "Inspection report", "About the model"])

    with tab_details:
        st.dataframe([dict(zip(app.TABLE_HEADERS, row)) for row in table],
                     hide_index=True, width="stretch")

    with tab_report:
        st.code(report, language=None, wrap_lines=True)
        st.download_button(
            "Download report", report,
            file_name=f"tablet-inspection-{datetime.now():%Y%m%d-%H%M}.txt",
            mime="text/plain", icon=":material/download:",
        )

    with tab_model:
        st.markdown(app.model_summary())
        metrics = app.model_metrics_html()
        if metrics:
            render(metrics)

render(ui.footer(
    "Visual inspection cannot confirm that a medicine is safe to take. "
    "Always check packaging and expiry, and consult a pharmacist."))
