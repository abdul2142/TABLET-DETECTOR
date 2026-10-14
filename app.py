from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ui
from image_utils import (DEFECT_TYPES, detect, detection_warnings, extract_features,
                         make_blister_pack_image, make_multi_tablet_image)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "tablet_defect_model.joblib"
EXAMPLES_DIR = ROOT / "artifacts" / "examples"

UNCERTAIN_BELOW = 70.0  # % confidence under which we refuse to give a verdict
COLORS = ui.COLORS
PACK_COLOR = ui.PACK_COLOR

#: Uploads are downscaled to this longest edge before inspection. Detection
#: time grows with pixel count (0.08s at 0.3 MP, 3.0s at 11.3 MP) while the
#: tablet count is unchanged, so a 48 MP phone photo would stall a worker for
#: no benefit. Coordinates in the report refer to the processed image, which is
#: also the overlay the operator sees, so the two always agree.
MAX_EDGE = int(os.environ.get("TIC_MAX_EDGE", "1600"))

#: Refuse uploads larger than this outright, before decoding.
MAX_UPLOAD = os.environ.get("TIC_MAX_UPLOAD", "25mb")
TABLE_HEADERS = ["Tablet", "In", "Position", "Photo area", "Centre x, y (px)", "Verdict", "Defect type", "Confidence"]
PACK_LAYOUTS = {"2 × 5": (5, 2), "2 × 6": (6, 2), "3 × 4": (4, 3), "4 × 3": (3, 4), "1 × 4": (1, 4)}


# --------------------------------------------------------------------------- model

@lru_cache(maxsize=1)
def load_model():
    import joblib

    if not MODEL_PATH.exists():
        raise gr.Error("Model not found. Run: python train.py")
    model = joblib.load(MODEL_PATH)
    if len(model.classes_) != len(DEFECT_TYPES):
        raise gr.Error("Model is from an older version. Retrain with: python train.py")
    return model


@lru_cache(maxsize=1)
def model_metadata() -> dict:
    meta_path = MODEL_PATH.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def model_summary() -> str:
    """Markdown table describing the trained model."""
    meta = model_metadata()
    if not meta:
        return "_No trained model found — run `python train.py`._"
    binary = meta.get("test_accuracy_binary")
    if binary is None:
        return "_Model metadata is from an older version — retrain with `python train.py`._"

    sourced = meta.get("detector_sourced_share")
    training = f"{meta['samples']:,} synthetic tablet crops"
    if sourced:
        training += f" ({sourced:.0%} taken from whole-photo detections)"
    return f"""
| | |
|---|---|
| **Classifier** | {meta.get('model', 'Gradient Boosting')} |
| **Training data** | {training} |
| **Features** | {meta['features']:,} shape, silhouette, edge and colour features per tablet |
| **Accuracy — normal vs. defective** | {binary:.1%} |
| **Accuracy — defect type** | {meta['test_accuracy_defect_type']:.1%} |
| **Defect classes** | {', '.join(DEFECT_TYPES[1:])} |
| **Uncertain band** | confidence below {UNCERTAIN_BELOW:.0f}% is sent for manual review |

Accuracy is measured on held-out synthetic tablets. Performance on real
photographs has not been validated.
"""


def model_metrics_html() -> str:
    """Per-class recall tiles, or nothing if the model predates them."""
    meta = model_metadata()
    recalls = meta.get("recall_per_class")
    if not recalls:
        return ""
    return ui.metrics_grid(recalls, meta.get("missed_as_normal", {}))


def classify(tablet: Image.Image) -> dict:
    """Classify one cropped tablet."""
    model = load_model()
    probs = dict(zip((DEFECT_TYPES[int(c)] for c in model.classes_), model.predict_proba([extract_features(tablet)])[0]))
    p_defective = 1.0 - probs["none"]
    confidence = max(p_defective, 1.0 - p_defective) * 100
    defect_type = max(DEFECT_TYPES[1:], key=probs.get)

    if confidence < UNCERTAIN_BELOW:
        verdict = "uncertain"
    elif p_defective >= 0.5:
        verdict = "defective"
    else:
        verdict, defect_type = "normal", "—"
    return {"verdict": verdict, "defect_type": defect_type, "confidence": confidence}


# --------------------------------------------------------------------------- drawing

@lru_cache(maxsize=8)
def font(size: int):
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _tag(draw: ImageDraw.ImageDraw, xy, text: str, color, size: int) -> None:
    """Filled label tag with white text, anchored at the top-left of a box."""
    f = font(size)
    x, y = xy
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    pad = max(3, size // 4)
    h = b - t + 2 * pad
    y = max(y - h, 0)
    draw.rounded_rectangle((x, y, x + r - l + 2 * pad, y + h), radius=pad, fill=color)
    draw.text((x + pad - l, y + pad - t), text, fill="white", font=f)


def count_strip(image: Image.Image, results) -> Image.Image:
    """Add a header strip above the photo stating how many tablets were found."""
    n = len(results)
    tally = {v: sum(r["verdict"] == v for r in results) for v in COLORS}
    size = max(14, image.width // 34)
    pad = size // 2
    h = size + 2 * pad
    out = Image.new("RGB", (image.width, image.height + h), (15, 23, 42))
    out.paste(image, (0, h))
    draw = ImageDraw.Draw(out)

    x = pad
    main = f"{n} TABLET{'S' if n != 1 else ''} DETECTED"
    draw.text((x, h // 2), main, fill="white", font=font(size), anchor="lm")
    x += draw.textlength(main, font=font(size)) + 2 * pad
    small = font(max(11, int(size * 0.8)))
    for verdict, label in (("normal", "NORMAL"), ("defective", "DEFECTIVE"), ("uncertain", "CHECK")):
        if not tally[verdict]:
            continue
        text = f"{tally[verdict]} {label}"
        r = size // 3
        draw.ellipse((x, h // 2 - r, x + 2 * r, h // 2 + r), fill=COLORS[verdict])
        x += 2 * r + pad // 2
        draw.text((x, h // 2), text, fill=(203, 213, 225), font=small, anchor="lm")
        x += draw.textlength(text, font=small) + pad * 1.5
    return out


def annotate(image: Image.Image, boxes, results, packs) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    base = min(out.size)
    width = max(2, base // 180)
    size = max(12, base // 32)

    for n, box in enumerate(packs, start=1):
        draw.rounded_rectangle(box, radius=10, outline=PACK_COLOR, width=max(1, width - 1))
        _tag(draw, (box[0], box[1]), f"PACK {n}" if len(packs) > 1 else "BLISTER PACK", PACK_COLOR, size)

    for i, (box, res) in enumerate(zip(boxes, results), start=1):
        color = COLORS[res["verdict"]]
        draw.rounded_rectangle(box, radius=6, outline=color, width=width if res["verdict"] == "normal" else width + 1)
        label = {"normal": f"{i}", "defective": f"{i} DEFECT", "uncertain": f"{i} CHECK"}[res["verdict"]]
        _tag(draw, (box[0], box[1] + 1), label, color, size)
    return count_strip(out, results)


def zoom(crop: Image.Image, color, size: int = 320) -> Image.Image:
    """Enlarge a tablet crop for close-up viewing, with a coloured frame."""
    scale = size / max(crop.size)
    big = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
    framed = Image.new("RGB", (big.width + 10, big.height + 10), color)
    framed.paste(big, (5, 5))
    return framed


# --------------------------------------------------------------------------- report

def location_of(box, packs) -> str:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for n, (l, t, r, b) in enumerate(packs, start=1):
        if l <= cx <= r and t <= cy <= b:
            return f"Pack {n}" if len(packs) > 1 else "Blister pack"
    return "Loose"


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def grid_positions(boxes, idxs) -> dict:
    """Row/column (1-based) of each box in `idxs`, grouping boxes into rows by vertical centre."""
    idxs = sorted(idxs, key=lambda i: _centre(boxes[i])[1])
    tol = 0.5 * float(np.median([boxes[i][3] - boxes[i][1] for i in idxs]))
    rows = []
    for i in idxs:
        cy = _centre(boxes[i])[1]
        if rows and abs(cy - np.mean([_centre(boxes[j])[1] for j in rows[-1]])) < tol:
            rows[-1].append(i)
        else:
            rows.append([i])
    return {i: (r, c) for r, row in enumerate(rows, start=1)
            for c, i in enumerate(sorted(row, key=lambda j: _centre(boxes[j])[0]), start=1)}


def image_area(box, size) -> str:
    """Coarse area of the photo: 'top-left', 'centre', 'bottom-right', ..."""
    cx, cy = _centre(box)
    v = ("top", "middle", "bottom")[min(int(3 * cy / size[1]), 2)]
    h = ("left", "centre", "right")[min(int(3 * cx / size[0]), 2)]
    return "centre" if (v, h) == ("middle", "centre") else f"{v}-{h}"


def positions(boxes, locations, size) -> list[dict]:
    """Exact position of every tablet: group (pack / loose), row & column within it, photo area, pixel centre."""
    grid = {}
    for group in dict.fromkeys(locations):
        grid.update(grid_positions(boxes, [i for i, loc in enumerate(locations) if loc == group]))
    return [{"group": loc, "row": grid[i][0], "col": grid[i][1], "area": image_area(box, size),
             "xy": tuple(round(v) for v in _centre(box))}
            for i, (box, loc) in enumerate(zip(boxes, locations))]


def where_text(p: dict, pixels: bool = True) -> str:
    s = f"{p['group']}, row {p['row']}, col {p['col']} ({p['area']})"
    return s + (f" · x={p['xy'][0]}, y={p['xy'][1]} px" if pixels else "")


def summary_html(results=None, locations=None, packs=0, where=None, warnings=()) -> str:
    """Count card, detection caution, verdict banner and the defect locations."""
    if not results:
        return ui.count_card() + ui.verdict_banner(
            "idle", "Awaiting image",
            "Upload a photo, pick an example, or generate a test image to begin the inspection.")

    grouped = {v: [i for i, r in enumerate(results, start=1) if r["verdict"] == v] for v in COLORS}
    defective, uncertain = grouped["defective"], grouped["uncertain"]

    def ids(items):
        return ", ".join(f"#{i} (row {where[i - 1]['row']}, col {where[i - 1]['col']})" for i in items)

    if defective:
        kind = "reject"
        title = f"REJECT — {len(defective)} of {len(results)} tablets defective"
        subtitle = f"Remove tablet{'s' if len(defective) > 1 else ''} {ids(defective)}."
        if packs:
            subtitle += " Do not dispense this strip — a defective tablet is sealed in the pack."
        if uncertain:
            subtitle += f" Also check {ids(uncertain)} manually."
    elif uncertain:
        kind, title = "review", "MANUAL REVIEW REQUIRED"
        subtitle = (f"No clear defects, but the model is unsure about {ids(uncertain)}. "
                    "Retake the photo or inspect by hand.")
    else:
        kind = "pass"
        title = f"PASS — no visual defects on {len(results)} tablet{'s' if len(results) > 1 else ''}"
        subtitle = "Visual check only. Still verify packaging, expiry date and batch details."

    return (ui.count_card(results, locations, packs)
            + ui.detection_notice(warnings)
            + ui.verdict_banner(kind, title, subtitle)
            + ui.locations_table(results, where))


def build_report(results, locations, n_packs: int, where, warnings=()) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    defective = [i for i, r in enumerate(results, start=1) if r["verdict"] == "defective"]
    uncertain = [i for i, r in enumerate(results, start=1) if r["verdict"] == "uncertain"]
    normal = [i for i, r in enumerate(results, start=1) if r["verdict"] == "normal"]
    lines = [
        "TABLET VISUAL INSPECTION REPORT",
        f"Inspection time : {stamp}",
        f"Tablets counted : {len(results)}  ({len(results) - locations.count('Loose')} in blister packs · "
        f"{locations.count('Loose')} loose)",
        f"Blister packs   : {n_packs}",
        f"Breakdown       : normal {len(normal)} · defective {len(defective)} · uncertain {len(uncertain)}",
        f"Overall result  : {'REJECT' if defective else 'MANUAL REVIEW' if uncertain else 'PASS (visual only)'}",
        "",
    ]
    if warnings:
        lines.append("COUNT RELIABILITY WARNING:")
        lines += [f"  - {w}" for w in warnings]
        lines.append("")
    if defective:
        lines.append("DEFECTIVE — do NOT consume / dispense:")
        lines += [f"  #{i:<3} {results[i-1]['defect_type']:<14} {results[i-1]['confidence']:>3.0f}%   {where_text(where[i-1])}"
                  for i in defective]
    if uncertain:
        lines.append("UNCERTAIN — inspect manually:")
        lines += [f"  #{i:<3} possible {results[i-1]['defect_type']:<14} {where_text(where[i-1])}" for i in uncertain]
    if normal:
        lines.append("Appear normal: " + ", ".join(f"#{i}" for i in normal))
    lines += [
        "",
        "A tablet that looks normal can still be expired, counterfeit or contaminated. Check packaging, "
        "expiry date and ask a pharmacist. Educational prototype — not a medical or pharmaceutical approval system.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- actions

def downscale(image: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """Shrink an oversized upload so inspection stays responsive.

    Returns the image untouched when it is already small enough.
    """
    longest = max(image.size)
    if longest <= max_edge:
        return image
    scale = max_edge / longest
    return image.resize((max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))), Image.LANCZOS)


def inspect(image: Image.Image | None):
    if image is None:
        raise gr.Error("Please upload, choose or generate an image first.")
    image = downscale(image.convert("RGB"))
    boxes, packs = detect(image)
    if not boxes:
        raise gr.Error("No tablets found. Use a plain, evenly lit background that contrasts with the tablets.")

    crops = [image.crop(box) for box in boxes]
    results = [classify(crop) for crop in crops]
    locations = [location_of(box, packs) for box in boxes]
    where = positions(boxes, locations, image.size)
    warnings = detection_warnings(boxes)

    closeups = [
        (zoom(crop, COLORS[r["verdict"]]),
         f"#{i} · {r['defect_type'].capitalize()} · {r['confidence']:.0f}%"
         + (" · needs review" if r["verdict"] == "uncertain" else "")
         + f" · {where_text(p, pixels=False)}")
        for i, (crop, r, p) in enumerate(zip(crops, results, where), start=1)
        if r["verdict"] != "normal"
    ]
    table = [
        [f"#{i}", p["group"], f"Row {p['row']}, Col {p['col']}", p["area"], f"{p['xy'][0]}, {p['xy'][1]}",
         r["verdict"].capitalize(), r["defect_type"], f"{r['confidence']:.1f}%"]
        for i, (r, p) in enumerate(zip(results, where), start=1)
    ]
    return (
        summary_html(results, locations, len(packs), where, warnings),
        annotate(image, boxes, results, packs),
        closeups,
        table,
        build_report(results, locations, len(packs), where, warnings),
    )


def generate_demo(kind: str, layout: str, count: int, defect_rate: int):
    if kind == "Blister pack":
        rows, cols = PACK_LAYOUTS[layout]
        image, truth = make_blister_pack_image(rows=rows, cols=cols, defect_rate=defect_rate / 100)
    else:
        image, truth = make_multi_tablet_image(count=int(count), defect_rate=defect_rate / 100)
    defects = [f"#{i} {t}" for i, t in enumerate(truth, start=1) if t != "none"]
    return image, f"{len(truth)} tablets · defective: " + (", ".join(defects) if defects else "none")


def toggle_kind(kind: str):
    return gr.update(visible=kind == "Blister pack"), gr.update(visible=kind != "Blister pack")


def example_images() -> list[str]:
    """Deterministic demo photos, written once.

    Generating these needs a writable ``artifacts/`` directory. A container with
    a read-only filesystem is a normal deployment, so a failure here degrades to
    "no example photos" rather than taking the whole console down at import.
    """
    specs = {
        "blister_pack_2x5.png": lambda: make_blister_pack_image(5, 2, 0.3, seed=11)[0],
        "blister_pack_3x4.png": lambda: make_blister_pack_image(4, 3, 0.25, seed=24)[0],
        "loose_tablets.png": lambda: make_multi_tablet_image(8, 0.35, seed=3)[0],
    }
    paths = []
    try:
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        for name, make in specs.items():
            path = EXAMPLES_DIR / name
            if not path.exists():
                make().save(path)
            paths.append(str(path))
    except OSError as exc:
        print(f"[tablet-console] example photos unavailable ({exc}); continuing without them.")
        return [p for p in paths if Path(p).exists()]
    return paths


# --------------------------------------------------------------------------- UI

def step(n: int, title: str, hint: str = "") -> gr.HTML:
    return gr.HTML(ui.step(n, title, hint))


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
).set(
    body_background_fill="#f1f5f9",
    body_background_fill_dark="#0b1120",
    block_radius="14px",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(15,23,42,.06)",
    button_large_radius="11px",
    button_small_radius="9px",
    button_primary_background_fill="linear-gradient(135deg, #2563eb, #0ea5e9)",
    button_primary_background_fill_hover="linear-gradient(135deg, #1d4ed8, #0284c7)",
    input_radius="11px",
)

with gr.Blocks(title="Tablet Inspection Console") as demo:
    gr.HTML(ui.header(
        "Tablet Inspection ", "Console",
        "Automated tablet counting and visual defect detection · loose tablets and blister packs",
        [("Model ready", True), ("Educational prototype · not for clinical use", False)],
    ))

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            step(1, "Load a photo", "upload, paste, webcam or generate")
            input_image = gr.Image(type="pil", label="Inspection photo", height=420,
                                   sources=["upload", "clipboard", "webcam"])
            with gr.Row():
                clear_button = gr.Button("Clear", variant="secondary", scale=1)
                predict_button = gr.Button("Count & inspect tablets", variant="primary", size="lg", scale=3)

            with gr.Accordion("Test image generator", open=False):
                kind = gr.Radio(["Blister pack", "Loose tablets"], value="Blister pack", show_label=False)
                with gr.Row():
                    layout = gr.Dropdown(list(PACK_LAYOUTS), value="2 × 5", label="Pack layout (columns × rows)")
                    count = gr.Slider(2, 12, value=6, step=1, label="Number of tablets", visible=False)
                    defect_rate = gr.Slider(0, 100, value=30, step=5, label="Defect chance (%)")
                generate_button = gr.Button("Generate & inspect", variant="secondary")
                ground_truth = gr.Textbox(label="Ground truth", interactive=False, lines=1)

            examples_slot = gr.Column()

        with gr.Column(scale=6):
            step(2, "Results")
            summary = gr.HTML(summary_html())
            annotated = gr.Image(type="pil", label="Detection overlay", height=480, interactive=False)

    step(3, "Flagged tablets", "close-ups of defective and uncertain tablets")
    closeups = gr.Gallery(show_label=False, columns=6, height="auto", object_fit="contain", preview=False)

    step(4, "Details & report")
    with gr.Tabs():
        with gr.Tab("Tablet details"):
            table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True)
        with gr.Tab("Inspection report"):
            report = gr.Textbox(show_label=False, lines=14, buttons=["copy"])
        with gr.Tab("About the model"):
            gr.Markdown(model_summary())
            gr.HTML(model_metrics_html())

    gr.HTML(ui.footer(
        "Visual inspection cannot confirm that a medicine is safe to take. "
        "Always check packaging and expiry, and consult a pharmacist."))

    outputs = [summary, annotated, closeups, table, report]
    with examples_slot:
        gr.Examples(
            example_images(), inputs=input_image, outputs=outputs, fn=inspect, run_on_click=True,
            cache_examples=False, label="Example photos — click to inspect",
            example_labels=["Blister pack 2 × 5", "Blister pack 3 × 4", "Loose tablets"],
        )
    predict_button.click(inspect, inputs=input_image, outputs=outputs)
    input_image.upload(inspect, inputs=input_image, outputs=outputs)
    clear_button.click(
        lambda: (None, summary_html(), None, [], None, "", ""),
        outputs=[input_image, summary, annotated, closeups, table, report, ground_truth],
    )
    kind.change(toggle_kind, inputs=kind, outputs=[layout, count])
    generate_button.click(
        generate_demo, inputs=[kind, layout, count, defect_rate], outputs=[input_image, ground_truth]
    ).then(inspect, inputs=input_image, outputs=outputs)


def server_host() -> str:
    """Interface to bind.

    Localhost by default, so a local run is never exposed on the network by
    accident. Hugging Face Spaces and the Dockerfile both set
    GRADIO_SERVER_NAME=0.0.0.0; SPACE_ID is checked as well so a Space still
    binds correctly even if that variable is ever absent.
    """
    configured = os.environ.get("GRADIO_SERVER_NAME")
    if configured:
        return configured
    return "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"  # noqa: S104


def server_port() -> int:
    """Port to listen on. Most hosts inject it as $PORT; Spaces expects 7860."""
    return int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 7860)


if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css=ui.CSS,
        server_name=server_host(),
        server_port=server_port(),
        max_file_size=MAX_UPLOAD,
        show_api=False,
    )
