"""Tests for the presentation layer: escaping, status mapping, motion policy."""

from __future__ import annotations

import re

import ui

XSS = '<img src=x onerror="alert(1)">'

NORMAL = {"verdict": "normal", "defect_type": "—", "confidence": 99.0}
DEFECTIVE = {"verdict": "defective", "defect_type": "chip", "confidence": 94.0}
UNCERTAIN = {"verdict": "uncertain", "defect_type": "crack", "confidence": 61.0}
PLACE = {"group": "Blister pack", "row": 2, "col": 1, "area": "middle-left", "xy": (120, 240)}


# --- Escaping ---------------------------------------------------------------

def test_esc_neutralises_markup():
    assert ui.esc("<b>x</b>") == "&lt;b&gt;x&lt;/b&gt;"
    assert ui.esc('a "quote" & amp') == "a &quot;quote&quot; &amp; amp"


def test_builders_escape_untrusted_text():
    builders = [
        ui.header(XSS, XSS, XSS, [(XSS, True)]),
        ui.step(1, XSS, XSS),
        ui.verdict_banner("pass", XSS, XSS),
        ui.footer(XSS),
        ui.count_card([NORMAL], [XSS], 1),
        ui.locations_table([{**DEFECTIVE, "defect_type": XSS}], [PLACE]),
        ui.metrics_grid({XSS: 0.9}, {XSS: 0.1}),
    ]
    for markup in builders:
        assert "<img" not in markup, markup[:120]


# --- Colours ----------------------------------------------------------------

def test_colour_map_covers_every_verdict():
    assert set(ui.COLORS) == set(ui.VERDICT_LABELS)


def test_rgb_formats_a_hex_colour():
    assert ui.rgb((22, 163, 74)) == "#16a34a"
    assert ui.rgb((0, 0, 0)) == "#000000"


def test_every_verdict_has_a_stylesheet_rule():
    for verdict in ui.COLORS:
        assert f".dot.{verdict}" in ui.CSS
        assert f".seg.{verdict}" in ui.CSS


# --- Count card -------------------------------------------------------------

def test_idle_count_card_shows_a_placeholder():
    markup = ui.count_card()
    assert "count-card idle" in markup
    assert "Run an inspection" in markup


def test_count_card_totals_and_splits():
    results = [NORMAL, NORMAL, DEFECTIVE, UNCERTAIN]
    locations = ["Blister pack"] * 3 + ["Loose"]
    markup = ui.count_card(results, locations, packs=1)

    assert '<div class="count-num">4</div>' in markup
    assert "3 in 1 blister pack" in markup
    assert "1 loose" in markup
    # One segment per present verdict, widths summing to 100%.
    widths = [float(w) for w in re.findall(r'class="seg \w+" style="width:([\d.]+)%', markup)]
    assert len(widths) == 3
    assert abs(sum(widths) - 100) < 0.01


def test_count_card_singular_wording():
    markup = ui.count_card([NORMAL], ["Loose"], packs=0)
    assert "Tablet in photo" in markup
    assert "1 loose" in markup


def test_count_card_omits_absent_verdicts_from_the_bar():
    markup = ui.count_card([NORMAL, NORMAL], ["Loose", "Loose"], packs=0)
    assert markup.count('class="seg ') == 1
    # The legend still lists every verdict, at zero.
    assert markup.count('class="lg"') == 3


# --- Verdict banner ---------------------------------------------------------

def test_verdict_banner_carries_its_kind_and_icon():
    for kind in ("pass", "reject", "review", "idle"):
        markup = ui.verdict_banner(kind, "T", "S")
        assert f'class="verdict {kind}"' in markup
        assert "<svg" in markup


def test_an_unknown_verdict_kind_falls_back_without_raising():
    assert ui.verdict_banner("nonsense", "T", "S")


def test_only_a_pass_may_shimmer():
    """A REJECT must not read as a celebration."""
    assert ".verdict.pass::after" in ui.CSS
    assert ".verdict.reject::after" not in ui.CSS


# --- Locations table --------------------------------------------------------

def test_locations_table_is_empty_when_everything_is_normal():
    assert ui.locations_table([NORMAL, NORMAL], [PLACE, PLACE]) == ""


def test_locations_table_lists_only_flagged_tablets():
    markup = ui.locations_table([NORMAL, DEFECTIVE, UNCERTAIN], [PLACE] * 3)
    assert markup.count("<tr class=") == 2
    assert "#2" in markup and "#3" in markup
    assert "Defective" in markup and "Needs review" in markup


def test_locations_table_renders_a_confidence_bar():
    markup = ui.locations_table([DEFECTIVE], [PLACE])
    assert 'class="cbar"' in markup
    assert "width:94.0%" in markup
    assert "94%" in markup


def test_locations_table_header_count_matches_its_columns():
    markup = ui.locations_table([DEFECTIVE], [PLACE])
    assert markup.count("<th>") == len(ui.LOCATION_HEADERS)
    assert markup.count("<td>") + markup.count('<td class="mono">') == len(ui.LOCATION_HEADERS)


# --- Detection notice -------------------------------------------------------

def test_no_notice_when_the_detection_looks_sound():
    assert ui.detection_notice([]) == ""


def test_the_notice_is_announced_as_an_alert():
    markup = ui.detection_notice(["Sizes vary a lot."])
    assert 'role="alert"' in markup
    assert "Check the tablet count" in markup
    assert "Sizes vary a lot." in markup


def test_the_notice_escapes_and_lists_every_warning():
    markup = ui.detection_notice([XSS, "second"])
    assert "<img" not in markup
    assert markup.count("<li>") == 2


# --- Per-class metrics ------------------------------------------------------

def test_metrics_grid_renders_one_tile_per_class():
    recalls = {"none": 0.997, "crack": 0.905, "chip": 0.92}
    markup = ui.metrics_grid(recalls, {"crack": 0.034, "chip": 0.08})
    assert markup.count('class="metric"') == 3
    assert "99.7%" in markup and "90.5%" in markup
    assert "missed as normal 8.0%" in markup


def test_metrics_grid_colours_a_weak_class_differently():
    strong = ui.metrics_grid({"chip": 0.97}, {"chip": 0.01})
    weak = ui.metrics_grid({"chip": 0.58}, {"chip": 0.40})
    assert ui.rgb(ui.COLORS["normal"]) in strong
    assert ui.rgb(ui.COLORS["defective"]) in weak


def test_metrics_grid_handles_a_missing_missed_rate():
    assert "recall on held-out crops" in ui.metrics_grid({"crack": 0.9}, {})


# --- Bars -------------------------------------------------------------------

def test_confidence_bar_clamps_out_of_range_values():
    assert "width:0.0%" in ui._bar(-20, "#000")
    assert "width:100.0%" in ui._bar(250, "#000")


# --- Motion and print policy ------------------------------------------------

def rules() -> str:
    return re.sub(r"/\*.*?\*/", "", ui.CSS, flags=re.DOTALL)


def test_reduced_motion_is_honoured_and_comes_last():
    assert "@media (prefers-reduced-motion: reduce)" in ui.CSS
    assert ui.CSS.index("prefers-reduced-motion") > ui.CSS.rindex("@keyframes")

    block = ui.CSS.split("prefers-reduced-motion: reduce")[1]
    assert "animation-duration: .001ms !important" in block
    assert "animation: none !important" in block


def test_every_animation_used_is_defined():
    css = rules()
    defined = set(re.findall(r"@keyframes\s+(\w+)", css))
    used = {n for n in re.findall(r"animation:\s*([\w-]+)", css) if n != "none"}
    assert used, "no animations found"
    assert used <= defined, f"undefined: {used - defined}"


def test_no_animation_is_defined_but_unused():
    css = rules()
    defined = set(re.findall(r"@keyframes\s+(\w+)", css))
    used = set(re.findall(r"animation:\s*([\w-]+)", css))
    assert defined <= used, f"dead animations: {defined - used}"


def test_only_ambient_animations_loop():
    """Inspection results must not pulse forever."""
    looping = set(re.findall(r"animation:\s*([\w-]+)[^;]*infinite", rules()))
    assert looping <= {"ticDrift", "ticPulse"}, f"unexpected loop: {looping}"


def test_a_print_stylesheet_strips_the_interface():
    assert "@media print" in ui.CSS
    block = ui.CSS.split("@media print")[1]
    for hidden in ("app-header", "app-footer", ".step"):
        assert hidden in block
    assert "animation: none !important" in block
