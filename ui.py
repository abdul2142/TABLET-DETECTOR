"""Presentation layer for the Tablet Inspection Console.

This module owns every pixel of the Gradio interface: the stylesheet, the theme
and the HTML fragments. It holds no vision logic and loads no model -- it turns
inspection results into markup, escaping every interpolated value on the way.

Theming
-------
Colours come from Gradio's own CSS variables (``--body-text-color``,
``--block-background-fill``, ``--border-color-primary``) wherever the surface
must follow the user's light/dark preference, and from the ``--tic-*`` tokens
below where the brand colour should stay fixed. That keeps the console readable
in both themes without maintaining two palettes.

Animation
---------
Pure CSS keyframes, kept brief and purposeful: panels rise in, the tablet count
counts up, status bars grow from zero, and the "model ready" dot pulses. The
verdict banner never bounces -- a REJECT is information, not a celebration. A
``prefers-reduced-motion`` block disables all of it, and a print stylesheet
turns the page into a clean inspection record.
"""

from __future__ import annotations

from html import escape
from typing import Mapping, Sequence

# Verdict colours, shared with the image-overlay drawing in app.py so the boxes
# on the photo match the chips in the report.
COLORS = {"normal": (22, 163, 74), "defective": (220, 38, 38), "uncertain": (234, 138, 0)}
PACK_COLOR = (37, 99, 235)

VERDICT_LABELS = {"normal": "Normal", "defective": "Defective", "uncertain": "Needs review"}


def esc(value: object) -> str:
    """HTML-escape any value for safe interpolation."""
    return escape(str(value), quote=True)


def rgb(color: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % color


LOGO = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect x="3" y="8" width="18" height="8" rx="4"/><path d="M12 8v8"/>'
    '<path d="M3 3h4M3 3v4M21 3h-4M21 3v4M3 21h4M3 21v-4M21 21h-4M21 21v-4"/></svg>'
)

ICONS = {
    "pass": '<path d="M4 12.5l5 5L20 6.5"/>',
    "reject": '<path d="M6 6l12 12M18 6L6 18"/>',
    "review": '<path d="M12 4l9 16H3z"/><path d="M12 10v4.5M12 17.4v.2"/>',
    "idle": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.6v.2"/>',
}


def icon(kind: str, size: int = 22) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS.get(kind, ICONS["idle"])}</svg>')


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1340px !important; margin: 0 auto; }
footer { display: none !important; }

.gradio-container {
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

/* ---------- Motion ---------- */
@keyframes ticRise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes ticFade { from { opacity: 0; } to { opacity: 1; } }
@keyframes ticBar { from { width: 0 !important; } }
@keyframes ticCount {
  from { opacity: 0; transform: translateY(8px) scale(.9); filter: blur(4px); }
  to   { opacity: 1; transform: none; filter: blur(0); }
}
@keyframes ticPulse {
  0%   { box-shadow: 0 0 0 0 rgba(34, 197, 94, .55); }
  70%  { box-shadow: 0 0 0 9px rgba(34, 197, 94, 0); }
  100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}
@keyframes ticSweep {
  0%   { transform: translateX(-130%) skewX(-16deg); opacity: 0; }
  25%  { opacity: .55; }
  100% { transform: translateX(320%) skewX(-16deg); opacity: 0; }
}
@keyframes ticDrift {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  50%      { transform: translate3d(-2%, 2%, 0) scale(1.08); }
}

/* ---------- Header ---------- */
.app-header {
  position: relative; overflow: hidden;
  background: linear-gradient(135deg, #0b1120 0%, #15264a 48%, #0e7490 100%);
  color: #fff; border-radius: 18px; padding: 20px 26px;
  display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap;
  box-shadow: 0 2px 4px rgba(15, 23, 42, .18), 0 18px 40px -22px rgba(14, 116, 144, .75);
  animation: ticFade .45s ease-out both;
}
.app-header::before {
  content: ""; position: absolute; inset: -40%; pointer-events: none;
  background: radial-gradient(circle at 78% 20%, rgba(56, 189, 248, .42), transparent 46%),
              radial-gradient(circle at 14% 92%, rgba(37, 99, 235, .34), transparent 50%);
  animation: ticDrift 26s var(--tic-ease) infinite;
}
.app-header::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
  background: repeating-linear-gradient(118deg, rgba(255, 255, 255, .05) 0 1px, transparent 1px 8px);
}
.app-header > * { position: relative; z-index: 2; }
.brand { display: flex; align-items: center; gap: 15px; }
.logo {
  width: 46px; height: 46px; border-radius: 13px; display: grid; place-items: center; flex: none;
  background: linear-gradient(135deg, var(--tic-brand), var(--tic-brand-2));
  box-shadow: 0 8px 20px -8px rgba(14, 165, 233, .8);
}
.app-header h1 {
  margin: 0; font-size: 1.42rem; font-weight: 750; letter-spacing: -.025em; color: #fff; line-height: 1.15;
}
.app-header h1 em {
  font-style: normal;
  background: linear-gradient(96deg, #7dd3fc, #67e8f9 55%, #a5f3fc);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
.app-header p { margin: 3px 0 0; font-size: .87rem; color: #9fb3cd; max-width: 62ch; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  border: 1px solid rgba(148, 183, 226, .32); border-radius: 999px; padding: 6px 13px;
  font-size: .71rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase;
  color: #d7e6f7; white-space: nowrap; display: inline-flex; align-items: center; gap: 7px;
  background: rgba(255, 255, 255, .08);
  -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
  transition: background .25s var(--tic-ease), transform .25s var(--tic-ease);
}
.chip:hover { background: rgba(255, 255, 255, .17); transform: translateY(-2px); }
.chip .live {
  width: 7px; height: 7px; border-radius: 50%; background: #22c55e;
  animation: ticPulse 2.4s ease-out infinite;
}

/* ---------- Step headings ---------- */
.step {
  display: flex; align-items: center; gap: 11px; font-weight: 650; font-size: .96rem;
  margin: 6px 0 -4px; animation: ticFade .35s ease-out both;
}
.step b {
  width: 23px; height: 23px; border-radius: 50%; display: grid; place-items: center; font-size: .74rem;
  background: linear-gradient(135deg, var(--tic-brand), var(--tic-brand-2)); color: #fff; flex: none;
  box-shadow: 0 4px 10px -4px rgba(37, 99, 235, .7);
}
.step span { opacity: .58; font-weight: 400; font-size: .85rem; }

/* ---------- Tablet count card ---------- */
.count-card {
  position: relative; overflow: hidden;
  border-radius: var(--tic-radius); padding: 20px 22px;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  box-shadow: var(--tic-shadow);
  animation: ticRise .4s var(--tic-ease) both;
}
.count-card::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--tic-brand), var(--tic-accent), transparent);
}
.count-main { display: flex; align-items: center; gap: 20px; }
.count-num {
  font-size: 3.6rem; font-weight: 800; line-height: 1; letter-spacing: -.04em;
  min-width: 1.4ch; font-variant-numeric: tabular-nums;
  background: linear-gradient(140deg, var(--tic-brand), var(--tic-accent));
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: var(--tic-brand);
  animation: ticCount .5s var(--tic-ease) .05s both;
}
.count-card.idle .count-num {
  background: none; -webkit-text-fill-color: currentColor; color: var(--body-text-color-subdued);
}
.count-label { font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; opacity: .62; }
.count-sub { font-size: 1.04rem; font-weight: 650; margin-top: 3px; }
.bar {
  display: flex; height: 11px; border-radius: 999px; overflow: hidden; margin-top: 17px; gap: 2px;
  background: var(--border-color-primary);
}
.seg { animation: ticBar .8s var(--tic-ease) both; }
.seg.normal { background: linear-gradient(90deg, #16a34a, #22c55e); }
.seg.defective { background: linear-gradient(90deg, #b91c1c, #ef4444); }
.seg.uncertain { background: linear-gradient(90deg, #c2710c, #f59e0b); }
.seg.empty { background: transparent; animation: none; }
.legend { display: flex; flex-wrap: wrap; gap: 7px 20px; margin-top: 12px; font-size: .85rem; }
.lg { display: inline-flex; align-items: center; gap: 7px; }
.lg em { opacity: .55; font-style: normal; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
.dot.normal { background: #16a34a; } .dot.defective { background: #dc2626; } .dot.uncertain { background: #ea8a00; }

/* ---------- Verdict banner ---------- */
.verdict {
  position: relative; overflow: hidden;
  border-radius: var(--tic-radius); padding: 16px 20px; border: 1px solid; margin-top: 12px;
  display: flex; gap: 14px; align-items: flex-start;
  animation: ticRise .42s var(--tic-ease) .06s both;
}
.verdict .v-ico {
  flex: none; width: 38px; height: 38px; border-radius: 50%; display: grid; place-items: center; color: #fff;
}
.verdict-title { font-size: 1.08rem; font-weight: 750; letter-spacing: -.01em; }
.verdict-sub { margin-top: 4px; font-size: .92rem; opacity: .92; line-height: 1.55; }
.verdict.idle { border-color: var(--border-color-primary); background: transparent; }
.verdict.idle .v-ico { background: var(--border-color-primary); color: var(--body-text-color-subdued); }
.verdict.pass { border-color: rgba(22, 163, 74, .45); background: rgba(22, 163, 74, .08); }
.verdict.pass .verdict-title { color: var(--tic-pass); }
.verdict.pass .v-ico { background: var(--tic-pass); }
.verdict.reject { border-color: rgba(220, 38, 38, .48); background: rgba(220, 38, 38, .09); }
.verdict.reject .verdict-title { color: var(--tic-reject); }
.verdict.reject .v-ico { background: var(--tic-reject); }
.verdict.review { border-color: rgba(234, 138, 0, .5); background: rgba(234, 138, 0, .09); }
.verdict.review .verdict-title { color: #d97706; }
.verdict.review .v-ico { background: var(--tic-review); }
/* A pass may shimmer once; a reject must not look celebratory. */
.verdict.pass::after {
  content: ""; position: absolute; top: 0; bottom: 0; left: 0; width: 30%; pointer-events: none;
  background: linear-gradient(100deg, transparent, rgba(255, 255, 255, .5), transparent);
  animation: ticSweep 2.4s var(--tic-ease) .3s 1;
}

/* ---------- Detection warning ---------- */
.detect-warn {
  display: flex; gap: 13px; align-items: flex-start; margin-top: 12px;
  border-radius: var(--tic-radius); padding: 14px 18px;
  border: 1px solid rgba(234, 138, 0, .5); background: rgba(234, 138, 0, .1);
  animation: ticRise .4s var(--tic-ease) both;
}
.detect-warn .w-ico {
  flex: none; width: 32px; height: 32px; border-radius: 50%; display: grid; place-items: center;
  background: var(--tic-review); color: #fff;
}
.detect-warn .w-title { font-weight: 700; color: #d97706; font-size: .98rem; }
.detect-warn ul { margin: 5px 0 0; padding-left: 18px; font-size: .89rem; line-height: 1.55; }
.detect-warn li { margin: 2px 0; }

/* ---------- Tables ---------- */
.loc-card, .metrics-card {
  margin-top: 12px; border-radius: var(--tic-radius); border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill); padding: 14px 16px; overflow-x: auto;
  box-shadow: var(--tic-shadow); animation: ticRise .4s var(--tic-ease) .1s both;
}
.loc-title {
  font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em;
  opacity: .62; margin-bottom: 8px;
}
table.loc { width: 100%; border-collapse: collapse; font-size: .86rem; }
table.loc th {
  text-align: left; font-weight: 650; opacity: .6; font-size: .73rem; padding: 5px 9px;
  border-bottom: 1px solid var(--border-color-primary); white-space: nowrap;
  text-transform: uppercase; letter-spacing: .05em;
}
table.loc td { padding: 7px 9px; border-bottom: 1px solid var(--border-color-primary); white-space: nowrap; }
table.loc tbody tr { transition: background .2s var(--tic-ease); }
table.loc tbody tr:hover { background: rgba(37, 99, 235, .06); }
table.loc tr:last-child td { border-bottom: none; }
table.loc tr td:first-child { border-left: 3px solid transparent; font-variant-numeric: tabular-nums; }
table.loc tr.defective td:first-child { border-left-color: var(--tic-reject); }
table.loc tr.uncertain td:first-child { border-left-color: var(--tic-review); }
table.loc tr.defective td:nth-child(2) { color: var(--tic-reject); font-weight: 650; }
table.loc tr.uncertain td:nth-child(2) { color: #d97706; font-weight: 650; }
.mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

/* Confidence micro-bar inside a table cell */
.cbar { display: inline-block; width: 46px; height: 5px; border-radius: 999px;
  background: var(--border-color-primary); overflow: hidden; vertical-align: middle; margin-right: 7px; }
.cbar > i { display: block; height: 100%; border-radius: 999px; animation: ticBar .7s var(--tic-ease) both; }

/* ---------- Per-class metrics ---------- */
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.metric {
  border-radius: 12px; border: 1px solid var(--border-color-primary); padding: 12px 14px;
  background: var(--block-background-fill);
  animation: ticRise .36s var(--tic-ease) both; animation-delay: var(--d, 0s);
  transition: transform .24s var(--tic-ease), border-color .24s var(--tic-ease);
}
.metric:hover { transform: translateY(-3px); border-color: rgba(37, 99, 235, .45); }
.metric .m-k { font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; opacity: .6; font-weight: 700; }
.metric .m-v { font-size: 1.5rem; font-weight: 780; letter-spacing: -.03em; margin-top: 4px;
  font-variant-numeric: tabular-nums; }
.metric .m-s { font-size: .78rem; opacity: .62; margin-top: 2px; }
.metric .cbar { width: 100%; margin: 7px 0 0; height: 6px; }

/* ---------- Footer ---------- */
.app-footer {
  text-align: center; font-size: .79rem; opacity: .62; padding: 12px 0 4px;
  border-top: 1px solid var(--border-color-primary); margin-top: 10px; line-height: 1.6;
}

/* ---------- Gradio widgets ---------- */
.gradio-container button.lg { border-radius: 11px !important; }
.gradio-container .gr-button, .gradio-container button {
  transition: transform .18s var(--tic-ease), box-shadow .22s ease, filter .2s ease;
}
.gradio-container button.primary:hover, .gradio-container button[variant="primary"]:hover {
  transform: translateY(-2px); box-shadow: var(--tic-shadow-lg); filter: brightness(1.06);
}
.gradio-container button:active { transform: translateY(0) scale(.99); }

/* ---------- Narrow viewports ---------- */
@media (max-width: 700px) {
  .app-header { padding: 16px 18px; border-radius: 14px; }
  .app-header h1 { font-size: 1.16rem; }
  .count-num { font-size: 2.7rem; }
  .count-main { gap: 14px; }
}

/* ---------- Reduced motion ---------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .001ms !important; animation-iteration-count: 1 !important;
    animation-delay: 0ms !important; transition-duration: .001ms !important;
  }
  .app-header::before, .chip .live, .verdict.pass::after { animation: none !important; }
}

/* ---------- Print: a clean inspection record ---------- */
@media print {
  .app-header, .step, .app-footer, .gradio-container button,
  .gradio-container .gr-button { display: none !important; }
  .count-card, .verdict, .loc-card, .metrics-card {
    box-shadow: none !important; break-inside: avoid; border: 1px solid #cbd5e1 !important;
  }
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .seg, .cbar > i { animation: none !important; }
}
"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def header(title: str, emphasis: str, tagline: str, chips: Sequence[tuple[str, bool]]) -> str:
    """The masthead. Each chip is ``(label, show_live_dot)``."""
    live_dot = '<i class="live"></i>'
    chip_html = "".join(
        f'<span class="chip">{live_dot if live else ""}{esc(label)}</span>'
        for label, live in chips
    )
    return (
        f'<div class="app-header"><div class="brand"><div class="logo">{LOGO}</div>'
        f'<div><h1>{esc(title)}<em>{esc(emphasis)}</em></h1>'
        f'<p>{esc(tagline)}</p></div></div>'
        f'<div class="chips">{chip_html}</div></div>'
    )


def step(number: int, title: str, hint: str = "") -> str:
    tail = f'<span>· {esc(hint)}</span>' if hint else ""
    return f'<div class="step"><b>{number}</b>{esc(title)}{tail}</div>'


def _bar(percent: float, color: str) -> str:
    width = max(0.0, min(100.0, percent))
    return f'<span class="cbar"><i style="width:{width:.1f}%;background:{color}"></i></span>'


def count_card(results: Sequence[Mapping] | None = None,
               locations: Sequence[str] | None = None, packs: int = 0) -> str:
    """Headline card: how many tablets are in the photo, and where."""
    if not results:
        return (
            '<div class="count-card idle"><div class="count-main"><div class="count-num">—</div>'
            '<div><div class="count-label">Tablets in photo</div>'
            '<div class="count-sub">Run an inspection to count tablets</div></div></div>'
            '<div class="bar"><span class="seg empty" style="width:100%"></span></div></div>'
        )

    total = len(results)
    locations = list(locations or [])
    loose = sum(loc == "Loose" for loc in locations)
    where = []
    if total - loose:
        where.append(f"{total - loose} in {packs} blister pack{'s' if packs != 1 else ''}")
    if loose:
        where.append(f"{loose} loose")

    segments, legend = [], []
    for verdict, label in VERDICT_LABELS.items():
        found = sum(r["verdict"] == verdict for r in results)
        if found:
            segments.append(f'<span class="seg {verdict}" style="width:{found / total * 100:.2f}%"></span>')
        legend.append(f'<span class="lg"><i class="dot {verdict}"></i>{label} <b>{found}</b> '
                      f'<em>({found / total:.0%})</em></span>')

    return (
        f'<div class="count-card"><div class="count-main"><div class="count-num">{total}</div>'
        f'<div><div class="count-label">Tablet{"s" if total != 1 else ""} in photo</div>'
        f'<div class="count-sub">{esc(" · ".join(where))}</div></div></div>'
        f'<div class="bar">{"".join(segments)}</div>'
        f'<div class="legend">{"".join(legend)}</div></div>'
    )


def verdict_banner(kind: str, title: str, subtitle: str) -> str:
    """``kind`` is ``pass``, ``reject``, ``review`` or ``idle``."""
    return (
        f'<div class="verdict {esc(kind)}"><span class="v-ico">{icon(kind)}</span>'
        f'<div><div class="verdict-title">{esc(title)}</div>'
        f'<div class="verdict-sub">{esc(subtitle)}</div></div></div>'
    )


LOCATION_HEADERS = ("Tablet", "Status", "Defect", "In", "Position", "Photo area",
                    "Confidence", "Centre x, y (px)")


def locations_table(results: Sequence[Mapping], where: Sequence[Mapping]) -> str:
    """Where each defective or uncertain tablet is. Empty string if all normal."""
    rows = []
    for index, (result, place) in enumerate(zip(results, where), start=1):
        if result["verdict"] == "normal":
            continue
        color = rgb(COLORS[result["verdict"]])
        rows.append(
            f'<tr class="{esc(result["verdict"])}"><td><b>#{index}</b></td>'
            f'<td>{esc(VERDICT_LABELS[result["verdict"]])}</td>'
            f'<td>{esc(result["defect_type"])}</td><td>{esc(place["group"])}</td>'
            f'<td>Row {place["row"]}, Col {place["col"]}</td><td>{esc(place["area"])}</td>'
            f'<td>{_bar(result["confidence"], color)}'
            f'<span class="mono">{result["confidence"]:.0f}%</span></td>'
            f'<td class="mono">{place["xy"][0]}, {place["xy"][1]}</td></tr>'
        )
    if not rows:
        return ""
    head = "".join(f"<th>{esc(h)}</th>" for h in LOCATION_HEADERS)
    return (
        '<div class="loc-card"><div class="loc-title">Defect locations</div>'
        f'<table class="loc"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def metrics_grid(recalls: Mapping[str, float], missed: Mapping[str, float]) -> str:
    """Per-class recall tiles.

    An aggregate accuracy figure hides a weak class, and in an inspection tool a
    missed defect is the expensive error -- so each defect type reports both its
    recall and how often it was called normal.
    """
    tiles = []
    for index, (name, recall) in enumerate(recalls.items()):
        if name == "none":
            sub = "correctly passed"
            color = rgb(COLORS["normal"])
        else:
            as_normal = missed.get(name)
            sub = (f"missed as normal {as_normal:.1%}" if as_normal is not None
                   else "recall on held-out crops")
            color = rgb(COLORS["normal"]) if recall >= 0.9 else (
                rgb(COLORS["uncertain"]) if recall >= 0.75 else rgb(COLORS["defective"]))
        tiles.append(
            f'<div class="metric" style="--d:{min(index * 0.06, 0.4):.2f}s">'
            f'<div class="m-k">{esc(name)}</div>'
            f'<div class="m-v">{recall:.1%}</div>'
            f'{_bar(recall * 100, color)}'
            f'<div class="m-s">{esc(sub)}</div></div>'
        )
    return (
        '<div class="metrics-card"><div class="loc-title">Recall by class '
        '(held-out synthetic crops)</div>'
        f'<div class="metrics">{"".join(tiles)}</div></div>'
    )


def detection_notice(warnings: Sequence[str]) -> str:
    """A caution strip shown when the tablet count may be unreliable.

    Placed above the verdict, because a wrong count makes every verdict below it
    incomplete -- an undetected tablet is never inspected at all.
    """
    if not warnings:
        return ""
    items = "".join(f"<li>{esc(w)}</li>" for w in warnings)
    return (
        '<div class="detect-warn" role="alert">'
        f'<span class="w-ico">{icon("review", 20)}</span>'
        '<div><div class="w-title">Check the tablet count</div>'
        f'<ul>{items}</ul></div></div>'
    )


def footer(text: str) -> str:
    return f'<div class="app-footer">{esc(text)}</div>'
