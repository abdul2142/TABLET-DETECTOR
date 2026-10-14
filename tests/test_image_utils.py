"""Tests for synthetic image generation, tablet detection and feature extraction.

No trained model is needed here: these cover the deterministic vision layer that
everything else is built on.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import image_utils as iu
from image_utils import (DEFECT_TYPES, SILHOUETTE_FEATURES, SILHOUETTE_THRESHOLDS, crop_to_tablet,
                         detect, detect_tablets, extract_features, make_blister_pack_image,
                         make_multi_tablet_image, make_tablet_image, silhouette_features,
                         tablet_mask)


# --- Vocabulary -------------------------------------------------------------

def test_defect_types_start_with_none():
    """Index 0 must be the non-defect class; the whole app relies on `> 0`."""
    assert DEFECT_TYPES[0] == "none"
    assert len(DEFECT_TYPES) == len(set(DEFECT_TYPES))


# --- Single-tablet generation ----------------------------------------------

def test_make_tablet_image_is_deterministic():
    a, _, _ = make_tablet_image(defect_type="crack", seed=7)
    b, _, _ = make_tablet_image(defect_type="crack", seed=7)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_make_tablet_image_varies_with_the_seed():
    a, _, _ = make_tablet_image(defect_type="crack", seed=1)
    b, _, _ = make_tablet_image(defect_type="crack", seed=2)
    assert not np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("defect_type", DEFECT_TYPES)
def test_label_matches_the_requested_defect(defect_type):
    image, label, reported = make_tablet_image(defect_type=defect_type, seed=3)
    assert reported == defect_type
    assert label == ("Normal Tablet" if defect_type == "none" else "Defective Tablet")
    assert image.size == (iu.IMAGE_SIZE, iu.IMAGE_SIZE)


@pytest.mark.parametrize("packaged", [False, True])
def test_packaged_flag_is_honoured(packaged):
    image, _, _ = make_tablet_image(defect_type="none", seed=5, packaged=packaged)
    assert image.size == (iu.IMAGE_SIZE, iu.IMAGE_SIZE)


# --- Multi-tablet and pack generation --------------------------------------

@pytest.mark.parametrize("count", [1, 2, 6, 9, 12])
def test_loose_truth_length_matches_the_count(count):
    _, truth = make_multi_tablet_image(count=count, defect_rate=0.5, seed=4)
    assert len(truth) == count
    assert set(truth) <= set(DEFECT_TYPES)


@pytest.mark.parametrize("rows,cols", [(5, 2), (4, 3), (3, 4), (1, 4)])
def test_pack_truth_length_matches_the_layout(rows, cols):
    _, truth = make_blister_pack_image(rows=rows, cols=cols, defect_rate=0.4, seed=6)
    assert len(truth) == rows * cols


def test_defect_rate_zero_gives_no_defects():
    _, truth = make_blister_pack_image(5, 2, 0.0, seed=8)
    assert set(truth) == {"none"}


def test_defect_rate_one_gives_no_normals():
    _, truth = make_blister_pack_image(5, 2, 1.0, seed=8)
    assert "none" not in truth


# --- Detection --------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_loose_tablets_are_counted_exactly(seed):
    image, truth = make_multi_tablet_image(count=6, defect_rate=0.4, seed=200 + seed)
    assert len(detect_tablets(image)) == len(truth)


@pytest.mark.parametrize("seed", range(6))
def test_pack_pockets_are_counted_exactly(seed):
    image, truth = make_blister_pack_image(5, 2, 0.3, seed=300 + seed)
    boxes, packs = detect(image)
    assert len(boxes) == len(truth)
    assert len(packs) == 1


def test_detection_is_in_reading_order():
    """Box i must correspond to truth i: top-to-bottom, then left-to-right."""
    image, _ = make_blister_pack_image(5, 2, 0.3, seed=12)
    boxes, _ = detect(image)
    centres = [((b[1] + b[3]) / 2, (b[0] + b[2]) / 2) for b in boxes]
    rows = [c[0] for c in centres]
    # Row centres must be non-decreasing, and within a row x must increase.
    for (y1, x1), (y2, x2) in zip(centres, centres[1:]):
        height = np.median([b[3] - b[1] for b in boxes])
        assert y2 > y1 - 0.5 * height
        if abs(y2 - y1) < 0.5 * height:
            assert x2 > x1
    assert rows == sorted(rows) or True  # ordering asserted pairwise above


def test_a_blank_photo_detects_nothing():
    blank = Image.new("RGB", (200, 200), (220, 220, 220))
    boxes, packs = detect(blank)
    assert boxes == [] and packs == []


def test_detection_tolerates_a_tiny_image():
    boxes, packs = detect(Image.new("RGB", (8, 8), (200, 200, 200)))
    assert boxes == [] and packs == []


def test_boxes_stay_inside_the_photo():
    image, _ = make_multi_tablet_image(count=4, defect_rate=0.5, seed=21)
    w, h = image.size
    for left, top, right, bottom in detect_tablets(image):
        assert 0 <= left < right <= w
        assert 0 <= top < bottom <= h


# --- Detection quality warnings ---------------------------------------------

def test_consistent_boxes_raise_no_warning():
    boxes = [(0, 0, 50, 40), (60, 0, 110, 40), (0, 60, 50, 100), (60, 60, 110, 100)]
    assert iu.detection_warnings(boxes) == []


def test_wildly_uneven_boxes_are_flagged():
    """The signature a miscount leaves behind: merged or fragmented blobs."""
    boxes = [(0, 0, 100, 100), (0, 0, 30, 30), (0, 0, 32, 28), (0, 0, 120, 110)]
    warnings = iu.detection_warnings(boxes)
    assert warnings
    assert any("vary a lot in size" in w for w in warnings)


def test_an_oversized_region_is_called_out():
    boxes = [(0, 0, 40, 40), (0, 0, 42, 38), (0, 0, 41, 40), (0, 0, 140, 130)]
    assert any("touching tablets" in w for w in iu.detection_warnings(boxes))


@pytest.mark.parametrize("boxes", [[], [(0, 0, 10, 10)]])
def test_too_few_boxes_to_judge(boxes):
    assert iu.detection_warnings(boxes) == []


def test_degenerate_boxes_do_not_raise():
    assert iu.detection_warnings([(0, 0, 0, 0), (5, 5, 5, 5)]) == []


@pytest.mark.parametrize("seed,rows,cols", [(31256, 5, 2), (31264, 5, 2)])
def test_the_known_undercount_photos_are_flagged(seed, rows, cols):
    """These two seeds really do miscount; the operator must be told.

    31256 finds 5 tablets in a pack of 10 -- without a warning, five tablets
    would silently go uninspected.
    """
    image, truth = make_blister_pack_image(rows, cols, 0.3, seed=seed)
    boxes, _ = detect(image)
    assert len(boxes) != len(truth), "seed no longer miscounts; update this test"
    assert iu.detection_warnings(boxes), "a miscounted photo produced no warning"


@pytest.mark.parametrize("seed", range(8))
def test_good_photos_are_not_falsely_flagged(seed):
    image, truth = make_blister_pack_image(5, 2, 0.3, seed=400 + seed)
    boxes, _ = detect(image)
    if len(boxes) == len(truth):
        assert iu.detection_warnings(boxes) == []


# --- Masking and cropping ---------------------------------------------------

def test_tablet_mask_finds_the_tablet():
    image, _, _ = make_tablet_image(defect_type="none", seed=9)
    mask = tablet_mask(image)
    assert 0.05 < mask.mean() < 0.8


def test_crop_to_tablet_shrinks_a_padded_photo():
    tablet, _, _ = make_tablet_image(defect_type="none", seed=10)
    padded = Image.new("RGB", (400, 400), (225, 225, 225))
    padded.paste(tablet, (136, 136))
    assert max(crop_to_tablet(padded).size) < 300


def test_crop_to_tablet_returns_the_original_when_nothing_stands_out():
    blank = Image.new("RGB", (64, 64), (210, 210, 210))
    assert crop_to_tablet(blank).size == (64, 64)


# --- Silhouette features ----------------------------------------------------

def test_silhouette_feature_count_is_stable():
    image, _, _ = make_tablet_image(defect_type="none", seed=11)
    assert len(silhouette_features(image)) == SILHOUETTE_FEATURES * len(SILHOUETTE_THRESHOLDS)


@pytest.mark.parametrize("mask", [
    np.zeros((20, 20), dtype=bool),                  # empty
    np.eye(20, dtype=bool),                          # a diagonal line (degenerate hull)
    np.pad(np.ones((1, 1), dtype=bool), 9),          # a single pixel
])
def test_degenerate_silhouettes_do_not_raise(mask):
    out = iu._silhouette(mask)
    assert len(out) == SILHOUETTE_FEATURES
    assert np.isfinite(out).all()


def test_silhouette_is_scale_invariant():
    """A bigger photo of the same shape must give the same descriptors."""
    small = np.zeros((60, 60), dtype=bool)
    small[15:45, 10:50] = True
    big = np.kron(small, np.ones((3, 3), dtype=bool))
    assert np.allclose(iu._silhouette(small), iu._silhouette(big), atol=0.05)


def test_a_notched_silhouette_is_less_solid_than_a_full_one():
    """This is the cue that makes a chip detectable at all."""
    full = np.zeros((80, 80), dtype=bool)
    yy, xx = np.mgrid[:80, :80]
    full[((yy - 40) ** 2 / 25 ** 2 + (xx - 40) ** 2 / 35 ** 2) <= 1] = True
    notched = full.copy()
    notched[30:50, 66:] = False  # bite a chunk out of one edge

    fill_full, solid_full = iu._silhouette(full)[:2]
    fill_notch, solid_notch = iu._silhouette(notched)[:2]
    assert solid_notch < solid_full
    assert fill_notch < fill_full


def test_chips_differ_from_normals_on_silhouette_features():
    """The regression that mattered: packaged chips were called normal because
    nothing in the feature vector described the outline."""
    def sample(defect_type):
        rows = [silhouette_features(make_tablet_image(defect_type=defect_type, seed=s, packaged=True)[0])
                for s in range(40)]
        return np.asarray(rows)

    chip, none = sample("chip"), sample("none")
    # The highest threshold isolates the tablet from the clear pocket; its
    # minimum radius is the discriminative feature.
    radmin = SILHOUETTE_FEATURES * 2 + 2
    pooled = np.sqrt((chip[:, radmin].std() ** 2 + none[:, radmin].std() ** 2) / 2)
    effect = abs(chip[:, radmin].mean() - none[:, radmin].mean()) / max(pooled, 1e-9)
    assert effect > 1.0, f"chip silhouette cue too weak (Cohen's d={effect:.2f})"


# --- Feature vector ---------------------------------------------------------

def test_feature_vector_is_fixed_length_and_finite():
    lengths = set()
    for defect_type in DEFECT_TYPES:
        for packaged in (False, True):
            image, _, _ = make_tablet_image(defect_type=defect_type, seed=13, packaged=packaged)
            features = extract_features(image)
            lengths.add(len(features))
            assert np.isfinite(features).all()
    assert len(lengths) == 1, f"feature length varies: {lengths}"


def test_features_are_deterministic():
    image, _, _ = make_tablet_image(defect_type="chip", seed=14)
    assert np.array_equal(extract_features(image), extract_features(image))


def test_features_survive_an_awkward_input():
    """A blank or one-pixel crop must not crash the extractor."""
    for image in (Image.new("RGB", (1, 1), (0, 0, 0)),
                  Image.new("RGB", (5, 5), (255, 255, 255)),
                  Image.new("L", (40, 40), 128)):
        features = extract_features(image)
        assert np.isfinite(features).all()


def test_features_are_dtype_float32():
    image, _, _ = make_tablet_image(defect_type="none", seed=15)
    assert extract_features(image).dtype == np.float32
