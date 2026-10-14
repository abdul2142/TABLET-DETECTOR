from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

IMAGE_SIZE = 128
LABELS = {0: "Normal Tablet", 1: "Defective Tablet"}
DEFECT_TYPES = ["none", "crack", "color anomaly", "chip", "surface spots"]

Box = Tuple[int, int, int, int]  # left, top, right, bottom

# Silhouette features are measured at several background-distance thresholds.
# A loose tablet separates from the table at the lowest threshold; a tablet in a
# blister pocket only separates from the clear plastic at the highest one, so
# one threshold alone cannot describe both.
SILHOUETTE_THRESHOLDS = (40, 90, 150)
RADIAL_BINS = 24
SILHOUETTE_FEATURES = 6  # per threshold; see _silhouette


def _random_bg(rng: random.Random) -> tuple:
    return (rng.randint(205, 245), rng.randint(205, 245), rng.randint(205, 245))


def _random_foil(rng: random.Random) -> tuple:
    grey = rng.randint(170, 195)
    return (grey, grey + rng.randint(0, 6), grey + rng.randint(4, 12))


def _draw_tablet(rng: random.Random, bg: tuple, defect_type: str, packaged: bool = False) -> Image.Image:
    """Draw one tablet (rotated, randomly placed) on an IMAGE_SIZE tile of colour `bg`.

    With `packaged`, `bg` is blister foil and the tablet sits in a clear plastic pocket.
    """
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), bg)
    draw = ImageDraw.Draw(img)

    jitter = 3 if packaged else 10
    cx = IMAGE_SIZE // 2 + rng.randint(-jitter, jitter)
    cy = IMAGE_SIZE // 2 + rng.randint(-jitter, jitter)
    rx, ry = rng.randint(34, 44 if packaged else 46), rng.randint(24, 34)
    base = rng.choice([(236, 230, 207), (222, 218, 193), (245, 238, 214), (214, 208, 183), (240, 240, 240)])
    outline = (115, 108, 91)

    if packaged:
        # Clear blister pocket: lighter than the foil, with a rim and a specular highlight.
        pr = rx + rng.randint(5, 8)
        pocket = tuple(int(c + (255 - c) * 0.45) for c in bg)
        draw.ellipse((cx - pr, cy - pr, cx + pr, cy + pr), fill=pocket, outline=tuple(c - 35 for c in bg), width=2)
        draw.arc((cx - pr + 5, cy - pr + 5, cx + pr - 5, cy + pr - 5), 200, 250, fill=(252, 252, 252), width=3)
    # A chipped-off piece reveals whatever is behind the tablet.
    behind = pocket if packaged else bg

    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=base, outline=outline, width=2)
    draw.line((cx - rx + 8, cy, cx + rx - 8, cy), fill=(175, 166, 145), width=1)

    if defect_type == "crack":
        points = [(cx + rng.randint(-12, 12), cy + rng.randint(-8, 8))]
        for _ in range(rng.randint(3, 6)):
            last_x, last_y = points[-1]
            points.append((last_x + rng.randint(-15, 15), last_y + rng.randint(-12, 12)))
        draw.line(points, fill=(75, 58, 45), width=2)
        if rng.random() > 0.45:
            draw.line(points[:3], fill=(95, 72, 53), width=1)
    elif defect_type == "color anomaly":
        x1, y1 = cx - rng.randint(8, 20), cy - rng.randint(8, 18)
        x2, y2 = x1 + rng.randint(12, 27), y1 + rng.randint(10, 25)
        tint = rng.choice([(176, 119, 90), (150, 160, 110), (190, 140, 150)])
        draw.ellipse((x1, y1, x2, y2), fill=tint)
    elif defect_type == "chip":
        side = rng.choice(["left", "right", "top", "bottom"])
        if side == "left":
            draw.polygon([(cx-rx, cy-10), (cx-rx+14, cy-18), (cx-rx+10, cy+8), (cx-rx, cy+13)], fill=behind)
        elif side == "right":
            draw.polygon([(cx+rx, cy-12), (cx+rx-13, cy-17), (cx+rx-9, cy+10), (cx+rx, cy+13)], fill=behind)
        elif side == "top":
            draw.polygon([(cx-13, cy-ry), (cx+10, cy-ry), (cx+16, cy-ry+12), (cx-8, cy-ry+9)], fill=behind)
        else:
            draw.polygon([(cx-12, cy+ry), (cx+14, cy+ry), (cx+9, cy+ry-12), (cx-15, cy+ry-9)], fill=behind)
    elif defect_type == "surface spots":
        for _ in range(rng.randint(5, 12)):
            x = cx + rng.randint(-rx + 10, rx - 10)
            y = cy + rng.randint(-ry + 8, ry - 8)
            r = rng.randint(1, 3)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(153, 125, 99))

    return img.rotate(rng.uniform(-30, 30), resample=Image.BILINEAR, fillcolor=bg)


def _camera_effects(img: Image.Image, rng: random.Random, seed: int | None) -> Image.Image:
    """Lighting, focus blur and sensor noise applied to the whole photo."""
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.12))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.7)))
    noise = np.random.default_rng(seed).normal(0, rng.uniform(1, 5), (img.height, img.width, 3))
    arr = np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pick_defect(rng: random.Random, defective: bool) -> str:
    return rng.choice(DEFECT_TYPES[1:]) if defective else "none"


def make_tablet_image(
    defective: bool | None = None,
    seed: int | None = None,
    defect_type: str | None = None,
    packaged: bool | None = False,
) -> Tuple[Image.Image, str, str]:
    """Create an educational synthetic image of a single tablet for mock testing.

    Position, size, rotation, lighting and sensor noise are randomised so the
    model learns the defect, not the exact framing of the synthetic image.
    `packaged=True` puts the tablet in a blister pocket; `None` picks at random.
    """
    rng = random.Random(seed)
    if defect_type is None:
        if defective is None:
            defective = rng.random() < 0.5
        defect_type = _pick_defect(rng, defective)
    if packaged is None:
        packaged = rng.random() < 0.5

    bg = _random_foil(rng) if packaged else _random_bg(rng)
    img = _camera_effects(_draw_tablet(rng, bg, defect_type, packaged), rng, seed)
    return img, LABELS[int(defect_type != "none")], defect_type


def make_multi_tablet_image(
    count: int = 6,
    defect_rate: float = 0.4,
    seed: int | None = None,
) -> Tuple[Image.Image, List[str]]:
    """Create a photo of `count` tablets laid out in a grid.

    Returns the image and the true defect type of each tablet in reading
    order (left-to-right, top-to-bottom), matching `detect_tablets` ordering.
    """
    rng = random.Random(seed)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    cell = IMAGE_SIZE + 32  # extra spacing so neighbouring tablets never touch

    bg = _random_bg(rng)
    canvas = Image.new("RGB", (cols * cell, rows * cell), bg)
    truth = []
    for i in range(count):
        defect_type = _pick_defect(rng, rng.random() < defect_rate)
        tile = _draw_tablet(rng, bg, defect_type)
        r, c = divmod(i, cols)
        canvas.paste(tile, (c * cell + 16, r * cell + 16))
        truth.append(defect_type)

    return _camera_effects(canvas, rng, seed), truth


def make_blister_pack_image(
    rows: int = 5,
    cols: int = 2,
    defect_rate: float = 0.3,
    seed: int | None = None,
) -> Tuple[Image.Image, List[str]]:
    """Create a photo of a blister pack (foil card with clear pockets) lying on a table.

    Returns the image and the true defect type of each pocket in reading order.
    """
    rng = random.Random(seed)
    table = rng.choice([(58, 66, 80), (120, 92, 66), (236, 236, 232), (40, 40, 44), (150, 170, 185)])
    foil = _random_foil(rng)
    band = rng.choice([(28, 86, 160), (160, 38, 52), (26, 112, 86), (90, 60, 140)])

    cell, margin, header = IMAGE_SIZE + 6, 18, 34
    pw, ph = cols * cell + 2 * margin, rows * cell + 2 * margin + header
    pack = Image.new("RGB", (pw, ph), foil)
    draw = ImageDraw.Draw(pack)
    draw.rectangle((0, 0, pw, header), fill=band)
    draw.text((margin, header // 2), "TAB 500 mg  |  LOT 24A17  |  EXP 08/27", fill="white", anchor="lm",
              font=ImageFont.load_default(size=13))

    truth = []
    top = header + margin
    for i in range(rows * cols):
        defect_type = _pick_defect(rng, rng.random() < defect_rate)
        r, c = divmod(i, cols)
        pack.paste(_draw_tablet(rng, foil, defect_type, packaged=True), (margin + c * cell + 3, top + r * cell + 3))
        truth.append(defect_type)

    # Perforation lines between pockets.
    perf = tuple(v - 6 for v in foil)
    for c in range(1, cols):
        x = margin + c * cell
        for y in range(top, ph - margin, 8):
            draw.line((x, y, x, y + 4), fill=perf, width=1)
    for r in range(1, rows):
        y = top + r * cell
        for x in range(margin, pw - margin, 8):
            draw.line((x, y, x + 4, y), fill=perf, width=1)

    shape = Image.new("L", pack.size, 0)
    ImageDraw.Draw(shape).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=14, fill=255)
    gap = 40
    canvas = Image.new("RGB", (pw + 2 * gap, ph + 2 * gap), table)
    canvas.paste(pack, (gap, gap), shape)
    canvas = canvas.rotate(rng.uniform(-4, 4), resample=Image.BILINEAR, fillcolor=table)
    return _camera_effects(canvas, rng, seed), truth


def _foreground(arr: np.ndarray, background: np.ndarray) -> np.ndarray:
    return np.abs(arr - background).sum(axis=2) > 40


def tablet_mask(image: Image.Image) -> np.ndarray:
    """Pixels that differ clearly from the background (estimated from the image border)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    return _foreground(arr, np.median(border, axis=0))


def crop_to_tablet(image: Image.Image, pad: int = 4) -> Image.Image:
    """Crop to the tablet by comparing pixels against the border (background) colour.

    Makes features robust to where the tablet sits in the frame and to
    uploaded photos with lots of empty space around the tablet.
    """
    rgb = image.convert("RGB")
    mask = tablet_mask(rgb)
    if mask.sum() < 0.01 * mask.size:
        return rgb
    ys, xs = np.where(mask)
    h, w = mask.shape
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad + 1, w), min(ys.max() + pad + 1, h))
    return rgb.crop(box)


def _blobs(mask: np.ndarray, grow: int):
    """Grow foreground so parts of one object merge, then yield (slice, blob mask) per object."""
    blobs = ndimage.binary_fill_holes(ndimage.binary_dilation(mask, iterations=grow))
    labels, _ = ndimage.label(blobs)
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        yield sl, labels[sl] == idx


def _box(sl, dy: int = 0, dx: int = 0) -> Box:
    return (sl[1].start + dx, sl[0].start + dy, sl[1].stop + dx, sl[0].stop + dy)


def _split_pack(arr: np.ndarray, region: np.ndarray, grow: int, edge: int) -> List[Box] | None:
    """If `region` is a blister pack, return the box of each pocket (in region coordinates).

    The foil is the region's dominant colour; pockets are the
    large, round, similar-sized blobs that differ from the foil. A single loose
    tablet has no such blobs (its spots/cracks are too small), so returns None.
    """
    arr = ndimage.uniform_filter(arr, size=(3, 3, 1))  # suppress foil texture / sensor noise
    # `region` was grown by `edge` px; trim that plus a few px of blurred pack edge.
    inner = ndimage.binary_erosion(region, iterations=edge + grow + 4)
    if inner.sum() < 500:
        return None
    # Foil colour = the dominant brightness in the region (peak of a smoothed histogram).
    pixels = arr[inner]
    gray = pixels.mean(axis=1)
    hist = ndimage.gaussian_filter1d(np.bincount(gray.astype(int), minlength=256).astype(float), 3)
    peak = int(hist.argmax())
    foil = np.median(pixels[np.abs(gray - peak) <= 6], axis=0)

    min_side = min(region.shape)
    candidates = []
    for sl, blob in _blobs(_foreground(arr, foil) & inner, grow):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        solid = blob.sum() / (bh * bw) >= 0.55  # filled pockets, not arcs of a tablet outline
        if solid and min(bh, bw) / max(bh, bw) >= 0.5 and min(bh, bw) >= 0.12 * min_side:
            candidates.append((_box(sl), int(blob.sum())))
    if len(candidates) < 2:
        return None

    typical = float(np.median([a for _, a in candidates]))
    pockets = [(b, a) for b, a in candidates if 0.4 * typical <= a <= 2.5 * typical]
    if len(pockets) < 2 or sum(a for _, a in pockets) < 0.2 * region.sum():
        return None
    return [b for b, _ in pockets]


def detect(image: Image.Image, pad: int = 6) -> Tuple[List[Box], List[Box]]:
    """Find every tablet in a photo, loose or inside blister packs.

    Returns (tablet boxes in reading order, blister pack boxes). Loose tablets
    that touch each other will be detected as one.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mask = tablet_mask(image)
    h, w = mask.shape
    grow = max(3, min(h, w) // 60)

    found, packs = [], []
    for sl, blob in _blobs(mask, grow):
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if mask[sl].sum() < 30 or min(bh, bw) < 2 * grow + 8:  # dust / noise
            continue
        if min(bh, bw) / max(bh, bw) < 0.3:  # long thin strips: labels, edges, pens — not tablets
            continue
        pockets = _split_pack(arr[sl], blob, max(2, grow // 2), grow)
        if pockets:
            packs.append(_box(sl))
            found += [(l + sl[1].start, t + sl[0].start, r + sl[1].start, b + sl[0].start) for l, t, r, b in pockets]
        else:
            found.append(_box(sl))
    if not found:
        return [], packs

    # Drop blobs much smaller than a typical tablet (crumbs, specks).
    areas = [(r - l) * (b - t) for l, t, r, b in found]
    typical = float(np.median(areas))
    found = [box for box, a in zip(found, areas) if a >= 0.2 * typical]

    # Reading order: group into rows by vertical centre, then sort each row left-to-right.
    found.sort(key=lambda b: (b[1] + b[3]) / 2)
    row_tol = 0.5 * float(np.median([b[3] - b[1] for b in found]))
    rows: List[List[Box]] = []
    for box in found:
        cy = (box[1] + box[3]) / 2
        if rows and abs(cy - np.mean([(b[1] + b[3]) / 2 for b in rows[-1]])) < row_tol:
            rows[-1].append(box)
        else:
            rows.append([box])
    ordered = [box for row in rows for box in sorted(row, key=lambda b: b[0])]

    # Grown blobs already include `grow` px of margin; add a little more context.
    extra = max(pad - grow, 0)
    tablets = [(max(l - extra, 0), max(t - extra, 0), min(r + extra, w), min(b + extra, h)) for l, t, r, b in ordered]
    return tablets, packs


def detect_tablets(image: Image.Image) -> List[Box]:
    """Boxes of every tablet in the photo (loose or packaged), in reading order."""
    return detect(image)[0]


#: Tablets in one photo should be close to the same size. When they are not,
#: blobs have usually merged or fragmented, which silently changes the count --
#: the dangerous failure, because an undetected tablet is never inspected.
#: Measured over 450 correct detections the spread never exceeded 0.275, while
#: the severe undercounts sat at 0.95-1.06, so 0.40 separates them cleanly.
AREA_SPREAD_LIMIT = 0.40
OVERSIZE_FACTOR = 1.8


def detection_warnings(boxes: List[Box]) -> List[str]:
    """Reasons to distrust a detection, as sentences for the operator.

    Returns an empty list when the detection looks internally consistent. This
    cannot find a tablet the detector missed outright; it flags the signature a
    miscount leaves behind -- boxes of wildly differing size.
    """
    if len(boxes) < 2:
        return []
    areas = np.array([(r - l) * (b - t) for l, t, r, b in boxes], dtype=float)
    median = float(np.median(areas))
    if median <= 0:
        return []

    warnings = []
    if areas.std() / median > AREA_SPREAD_LIMIT:
        warnings.append(
            "The detected tablets vary a lot in size, which usually means some were "
            "merged together or split apart — the count may be wrong. Retake the photo "
            "on a plain, evenly lit background with a gap between tablets."
        )
    oversized = int((areas > OVERSIZE_FACTOR * median).sum())
    if oversized:
        warnings.append(
            f"{oversized} detected region{'s are' if oversized > 1 else ' is'} far larger than "
            "the others and may be touching tablets counted as one."
        )
    return warnings


def _silhouette(mask: np.ndarray) -> np.ndarray:
    """Rotation- and scale-invariant descriptors of a binary silhouette.

    A chip bites a notch out of the tablet's outline while barely changing its
    colour or texture, so it is almost invisible to the intensity and colour
    features. These give the classifier the outline itself:

    * ``fill``     mask area over its bounding box -- a notch lowers it
    * ``solidity`` mask area over its convex hull -- a notch is a concavity
    * the radial profile (furthest pixel per angular sector, normalised), whose
      minimum and spread spike where the outline is cut away

    The profile is sorted before the summary statistics are taken, so the
    result does not depend on how the tablet happens to be rotated.
    """
    if mask.sum() < 30:
        return np.zeros(SILHOUETTE_FEATURES, dtype=np.float32)

    mask = ndimage.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    height = int(ys.max() - ys.min()) + 1
    width = int(xs.max() - xs.min()) + 1

    cy, cx = ys.mean(), xs.mean()
    radius = np.hypot(ys - cy, xs - cx)
    angle = np.arctan2(ys - cy, xs - cx)
    sector = (((angle + np.pi) / (2 * np.pi)) * RADIAL_BINS).astype(int) % RADIAL_BINS
    profile = np.zeros(RADIAL_BINS, dtype=np.float64)
    np.maximum.at(profile, sector, radius)
    profile /= profile.max() or 1.0

    try:
        hull_area = float(ConvexHull(np.column_stack([xs, ys])).volume)
    except (QhullError, ValueError):
        hull_area = area  # degenerate silhouette (a line or a single point)

    ordered = np.sort(profile)
    return np.array([
        area / (height * width),
        area / max(hull_area, 1.0),
        profile.min(),
        profile.std(),
        ordered[:4].mean(),
        ordered[-4:].mean(),
    ], dtype=np.float32)


def silhouette_features(image: Image.Image) -> np.ndarray:
    """Silhouette descriptors at each background-distance threshold."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    distance = np.abs(arr - np.median(border, axis=0)).sum(axis=2)
    return np.concatenate([_silhouette(distance > t) for t in SILHOUETTE_THRESHOLDS])


def extract_features(image: Image.Image) -> np.ndarray:
    """Extract compact, dependency-light visual features (shape, edges and colour)."""
    tablet = crop_to_tablet(image)

    gray = np.asarray(tablet.convert("L").resize((32, 32)), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))

    hsv = np.asarray(tablet.convert("HSV").resize((32, 32)), dtype=np.float32) / 255.0
    color_stats = np.concatenate([hsv.mean(axis=(0, 1)), hsv.std(axis=(0, 1))])
    hue_hist = np.histogram(hsv[..., 0], bins=12, range=(0, 1))[0] / hsv[..., 0].size
    sat_hist = np.histogram(hsv[..., 1], bins=8, range=(0, 1))[0] / hsv[..., 1].size

    # Fraction of strong edges and dark pixels: cracks, chips and spots raise these.
    edges = np.hypot(gx, gy)
    summary = np.array([
        gray.mean(), gray.std(), gx.mean(), gy.mean(),
        (edges > 0.08).mean(), (edges > 0.15).mean(),
        (gray < gray.mean() - 2 * gray.std()).mean(),
        tablet.width / max(tablet.height, 1),
    ], dtype=np.float32)

    return np.concatenate([
        gray.flatten(), gx.flatten(), gy.flatten(),
        color_stats, hue_hist, sat_hist, summary,
        silhouette_features(tablet),
    ]).astype(np.float32)


def save_image(image: Image.Image, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
