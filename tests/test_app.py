"""Tests for the inspection console's logic: verdicts, positions and reports.

The classifier is stubbed for the threshold tests so they do not depend on a
trained model. The end-to-end tests are marked ``model`` and skip when
``artifacts/tablet_defect_model.joblib`` is absent.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import app
from image_utils import DEFECT_TYPES, make_blister_pack_image, make_multi_tablet_image

needs_model = pytest.mark.skipif(not app.MODEL_PATH.exists(),
                                 reason="trained model not present; run python train.py")


# --- Test doubles -----------------------------------------------------------

class StubModel:
    """Returns one fixed probability vector over DEFECT_TYPES."""

    def __init__(self, probs):
        self.classes_ = np.arange(len(DEFECT_TYPES))
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        return np.tile(self._probs, (len(X), 1))


@pytest.fixture
def stub_classifier(monkeypatch):
    def install(**by_type):
        probs = np.zeros(len(DEFECT_TYPES))
        for name, value in by_type.items():
            probs[DEFECT_TYPES.index(name.replace("_", " "))] = value
        monkeypatch.setattr(app, "load_model", lambda: StubModel(probs))

    return install


TILE = Image.new("RGB", (40, 40), (230, 225, 200))


# --- classify ---------------------------------------------------------------

def test_a_confident_normal_is_passed(stub_classifier):
    stub_classifier(none=0.95, chip=0.05)
    result = app.classify(TILE)
    assert result["verdict"] == "normal"
    assert result["defect_type"] == "—"
    assert result["confidence"] == pytest.approx(95.0)


def test_a_confident_defect_is_named(stub_classifier):
    stub_classifier(none=0.05, chip=0.80, crack=0.15)
    result = app.classify(TILE)
    assert result["verdict"] == "defective"
    assert result["defect_type"] == "chip"
    assert result["confidence"] == pytest.approx(95.0)


def test_a_borderline_call_is_sent_for_review(stub_classifier):
    """Between 50% and the uncertainty threshold, no verdict is given."""
    stub_classifier(none=0.45, crack=0.55)
    result = app.classify(TILE)
    assert result["verdict"] == "uncertain"
    assert result["confidence"] < app.UNCERTAIN_BELOW


def test_the_uncertainty_threshold_is_the_boundary(stub_classifier):
    stub_classifier(none=1 - app.UNCERTAIN_BELOW / 100 - 0.01, chip=app.UNCERTAIN_BELOW / 100 + 0.01)
    assert app.classify(TILE)["verdict"] == "defective"

    stub_classifier(none=1 - app.UNCERTAIN_BELOW / 100 + 0.02, chip=app.UNCERTAIN_BELOW / 100 - 0.02)
    assert app.classify(TILE)["verdict"] == "uncertain"


# --- Geometry helpers -------------------------------------------------------

def test_centre_of_a_box():
    assert app._centre((0, 0, 10, 20)) == (5.0, 10.0)


@pytest.mark.parametrize("box,expected", [
    ((0, 0, 10, 10), "top-left"),
    ((140, 140, 160, 160), "centre"),
    ((280, 280, 300, 300), "bottom-right"),
    ((140, 0, 160, 20), "top-centre"),
    ((0, 140, 20, 160), "middle-left"),
])
def test_image_area_names_the_region(box, expected):
    assert app.image_area(box, (300, 300)) == expected


def test_image_area_handles_a_box_on_the_far_edge():
    """Indices must stay in range when the centre lands exactly on the border."""
    assert app.image_area((299, 299, 300, 300), (300, 300)) == "bottom-right"


def test_grid_positions_assigns_rows_and_columns():
    # Two rows of two, deliberately out of order.
    boxes = [(100, 100, 150, 140), (0, 0, 50, 40), (100, 0, 150, 40), (0, 100, 50, 140)]
    grid = app.grid_positions(boxes, range(len(boxes)))
    assert grid[1] == (1, 1) and grid[2] == (1, 2)
    assert grid[3] == (2, 1) and grid[0] == (2, 2)


def test_location_of_distinguishes_pack_from_loose():
    packs = [(0, 0, 100, 100)]
    assert app.location_of((10, 10, 40, 40), packs) == "Blister pack"
    assert app.location_of((200, 200, 240, 240), packs) == "Loose"


def test_location_of_numbers_multiple_packs():
    packs = [(0, 0, 100, 100), (200, 0, 300, 100)]
    assert app.location_of((10, 10, 40, 40), packs) == "Pack 1"
    assert app.location_of((210, 10, 240, 40), packs) == "Pack 2"


def test_positions_describes_every_tablet():
    boxes = [(0, 0, 50, 40), (100, 0, 150, 40)]
    where = app.positions(boxes, ["Loose", "Loose"], (300, 300))
    assert [p["col"] for p in where] == [1, 2]
    assert all(p["row"] == 1 for p in where)
    assert where[0]["xy"] == (25, 20)


def test_where_text_can_omit_pixels():
    place = {"group": "Loose", "row": 1, "col": 2, "area": "top-left", "xy": (3, 4)}
    assert "x=3" in app.where_text(place)
    assert "x=3" not in app.where_text(place, pixels=False)


# --- Summary HTML -----------------------------------------------------------

NORMAL = {"verdict": "normal", "defect_type": "—", "confidence": 99.0}
DEFECTIVE = {"verdict": "defective", "defect_type": "chip", "confidence": 94.0}
UNCERTAIN = {"verdict": "uncertain", "defect_type": "crack", "confidence": 61.0}
PLACE = {"group": "Blister pack", "row": 1, "col": 1, "area": "centre", "xy": (10, 10)}


def test_idle_summary_awaits_an_image():
    markup = app.summary_html()
    assert "Awaiting image" in markup
    assert "count-card idle" in markup


def test_all_normal_is_a_pass():
    markup = app.summary_html([NORMAL, NORMAL], ["Loose"] * 2, 0, [PLACE] * 2)
    assert "PASS" in markup
    assert "verdict pass" in markup


def test_any_defect_is_a_reject():
    markup = app.summary_html([NORMAL, DEFECTIVE], ["Blister pack"] * 2, 1, [PLACE] * 2)
    assert "REJECT" in markup
    assert "Do not dispense this strip" in markup
    assert "#2" in markup


def test_uncertain_without_defects_is_a_review():
    markup = app.summary_html([NORMAL, UNCERTAIN], ["Loose"] * 2, 0, [PLACE] * 2)
    assert "MANUAL REVIEW" in markup
    assert "verdict review" in markup


def test_a_reject_also_mentions_uncertain_tablets():
    markup = app.summary_html([DEFECTIVE, UNCERTAIN], ["Loose"] * 2, 0, [PLACE] * 2)
    assert "REJECT" in markup
    assert "Also check" in markup


def test_a_count_warning_appears_above_the_verdict():
    """A wrong count makes every verdict below it incomplete."""
    markup = app.summary_html([NORMAL], ["Loose"], 0, [PLACE],
                              warnings=["Sizes vary a lot."])
    assert "Check the tablet count" in markup
    assert markup.index("detect-warn") < markup.index("verdict ")


def test_no_warning_strip_when_detection_is_sound():
    assert "detect-warn" not in app.summary_html([NORMAL], ["Loose"], 0, [PLACE])


# --- Report -----------------------------------------------------------------

def test_report_states_the_overall_result_and_counts():
    report = app.build_report([NORMAL, DEFECTIVE], ["Blister pack", "Loose"], 1, [PLACE] * 2)
    assert "TABLET VISUAL INSPECTION REPORT" in report
    assert "Tablets counted : 2" in report
    assert "REJECT" in report
    assert "DEFECTIVE" in report
    assert "#2" in report


def test_report_marks_a_clean_batch_as_visual_only():
    report = app.build_report([NORMAL], ["Loose"], 0, [PLACE])
    assert "PASS (visual only)" in report
    assert "Appear normal: #1" in report


def test_report_lists_uncertain_tablets_separately():
    report = app.build_report([UNCERTAIN], ["Loose"], 0, [PLACE])
    assert "MANUAL REVIEW" in report
    assert "UNCERTAIN" in report


def test_report_records_a_count_reliability_warning():
    report = app.build_report([NORMAL], ["Loose"], 0, [PLACE],
                              warnings=["Sizes vary a lot."])
    assert "COUNT RELIABILITY WARNING" in report
    assert "Sizes vary a lot." in report


def test_report_omits_the_warning_block_when_clean():
    assert "COUNT RELIABILITY" not in app.build_report([NORMAL], ["Loose"], 0, [PLACE])


# --- Upload handling --------------------------------------------------------

def test_a_small_image_is_left_untouched():
    image = Image.new("RGB", (800, 600))
    assert app.downscale(image, 1600) is image


@pytest.mark.parametrize("size,expected", [
    ((3200, 2400), (1600, 1200)),
    ((2400, 3200), (1200, 1600)),
    ((4800, 4800), (1600, 1600)),
])
def test_a_large_upload_is_downscaled_preserving_aspect(size, expected):
    assert app.downscale(Image.new("RGB", size), 1600).size == expected


def test_downscaling_never_produces_a_zero_dimension():
    assert min(app.downscale(Image.new("RGB", (5000, 3)), 1600).size) >= 1


@needs_model
@pytest.mark.model
def test_an_oversized_photo_still_counts_correctly():
    """Downscaling must not cost accuracy -- that is why it is safe to do."""
    image, truth = make_blister_pack_image(5, 2, 0.3, seed=11)
    huge = image.resize((image.width * 5, image.height * 5), Image.LANCZOS)
    _, _, _, table, _ = app.inspect(huge)
    assert len(table) == len(truth)


# --- Overlay drawing --------------------------------------------------------

def test_annotate_adds_a_count_strip_above_the_photo():
    image = Image.new("RGB", (200, 160), (220, 220, 220))
    out = app.annotate(image, [(10, 10, 60, 50)], [DEFECTIVE], [])
    assert out.width == image.width
    assert out.height > image.height  # the strip was added


def test_zoom_enlarges_and_frames_a_crop():
    out = app.zoom(Image.new("RGB", (30, 20), (255, 0, 0)), (0, 255, 0), size=100)
    assert max(out.size) == 110  # 100 plus a 5px frame each side


# --- End to end -------------------------------------------------------------

@needs_model
@pytest.mark.model
@pytest.mark.parametrize("seed", [11, 24])
def test_inspect_counts_a_blister_pack_correctly(seed):
    image, truth = make_blister_pack_image(5, 2, 0.3, seed=seed)
    summary, annotated, closeups, table, report = app.inspect(image)

    assert len(table) == len(truth)
    assert f'<div class="count-num">{len(truth)}</div>' in summary
    assert annotated.size[0] == image.size[0]
    assert f"Tablets counted : {len(truth)}" in report


@needs_model
@pytest.mark.model
def test_inspect_counts_loose_tablets_correctly():
    image, truth = make_multi_tablet_image(count=8, defect_rate=0.35, seed=3)
    _, _, _, table, _ = app.inspect(image)
    assert len(table) == len(truth)


@needs_model
@pytest.mark.model
def test_a_clean_pack_passes():
    image, truth = make_blister_pack_image(5, 2, 0.0, seed=77)
    summary, _, closeups, table, report = app.inspect(image)
    assert len(table) == len(truth)
    assert "PASS" in summary
    assert closeups == []


@needs_model
@pytest.mark.model
def test_inspect_rejects_an_empty_input():
    with pytest.raises(Exception):
        app.inspect(None)


@needs_model
@pytest.mark.model
def test_inspect_refuses_a_blank_photo():
    with pytest.raises(Exception):
        app.inspect(Image.new("RGB", (300, 300), (225, 225, 225)))


# --- Model metadata ---------------------------------------------------------

@needs_model
@pytest.mark.model
def test_model_summary_reports_accuracy():
    summary = app.model_summary()
    assert "Accuracy" in summary
    assert "%" in summary


@needs_model
@pytest.mark.model
def test_per_class_metrics_are_published():
    """An aggregate score hides a weak class, so each one must be reported."""
    meta = app.model_metadata()
    assert set(meta["recall_per_class"]) == set(DEFECT_TYPES)
    assert app.model_metrics_html().count('class="metric"') == len(DEFECT_TYPES)
