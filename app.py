"""
JioTV+ Xtras — AI Video Content Classifier (Prototype)

A simplified proof-of-concept of the JioTV+ Xtras contextual overlay feature.
Detects song sequences, action scenes, dialogue, credits, and transitions in
video content using multi-modal AI — the core detection pipeline that powers
an Amazon X-Ray-style viewing experience.

Pipeline:
  1. Video is cut into overlapping clips
  2. Audio analysis (librosa) scores musical / dialogue / action
  3. Optional vision-language model (Ollama) labels sampled frames
  4. Hybrid fusion of both signals, then adjacent clips with the same label merge
  5. Confidence-based routing (auto-tag / QA review / human annotator)

The page opens on a pre-computed demo run (demo_results.json) so it needs no
video processing; "Analyse your own clip" runs the live pipeline.
"""

import html
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

CASE_STUDY_URL = "https://akhilrajs.com/portfolio/jiotv-xtras"
SOURCE_URL = "https://github.com/Akhil743/jiotv-xtras-prototype"
SITE_URL = "https://akhilrajs.com"


def _page_icon() -> Image.Image:
    """A small drawn favicon (three timeline bars) so the browser tab carries no emoji."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, 63, 63), radius=14, fill="#a8441e")
    for x0, height in ((13, 22), (27, 34), (41, 26)):
        top = 32 - height // 2
        draw.rounded_rectangle((x0, top, x0 + 9, top + height), radius=3, fill="#fbf9f4")
    return img


st.set_page_config(
    page_title="JioTV+ Xtras · AI content classifier",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Design tokens -----------------------------------------------------------

# label, fill colour, text colour on the fill (all fills pass 4.5:1 with their text)
CATEGORIES = {
    "musical": ("Musical", "#a8441e", "#fbf9f4"),
    "dialogue": ("Dialogue", "#4b4840", "#fbf9f4"),
    "action": ("Action", "#3f6d4e", "#fbf9f4"),
    "credits": ("Credits", "#8a6a1f", "#fbf9f4"),
    "transition": ("Transition", "#5a6b7b", "#fbf9f4"),
    "silence": ("Silence", "#cfc9bc", "#1c1b18"),
    "unknown": ("Unknown", "#9b968b", "#1c1b18"),
}

# Routing tiers: same thresholds as before (>=90% auto, 70-90% QA, <70% human)
TIERS = {
    "auto": {"label": "Auto-tag", "rule": "90% or more",
             "action": "published without review"},
    "qa": {"label": "QA review", "rule": "70–90%",
           "action": "a reviewer confirms or fixes it"},
    "human": {"label": "Human annotator", "rule": "under 70%",
              "action": "a person labels it"},
}

MODE_LABELS = {
    "audio": "Audio only",
    "vlm": "Vision only",
    "hybrid": "Hybrid (audio + vision)",
}

HYBRID_RE = re.compile(
    r"\[Hybrid\]\s*VLM:\s*(\w+)\s*\((\d+)%\)\s*\+\s*Audio:\s*(\w+)\s*\((\d+)%\)"
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500&family=Inter:wght@400;500;600&display=swap');
:root {
  --xt-bg: #f6f3ec; --xt-sunk: #eeeae0; --xt-surface: #fbf9f4;
  --xt-ink: #1c1b18; --xt-ink-2: #4b4840; --xt-muted: #6f6b62;
  --xt-rule: #dcd6c9; --xt-rule-strong: #c4bdac;
  --xt-accent: #a8441e; --xt-live: #2f7a55;
  --xt-auto-fg: #276a49; --xt-auto-bg: #e3efe7; --xt-auto-dot: #2f7a55;
  --xt-qa-fg: #8a5a00; --xt-qa-bg: #f6ead2; --xt-qa-dot: #b7791f;
  --xt-human-fg: #a8441e; --xt-human-bg: #f3e1d6; --xt-human-dot: #a8441e;
  --xt-display: 'Fraunces', Georgia, 'Times New Roman', serif;
  --xt-body: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  --xt-radius: 14px;
}
.stApp { font-family: var(--xt-body); }
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stExpander"] summary,
.stApp button, .stApp input, .stApp textarea { font-family: var(--xt-body); }
.stApp h1, .stApp h2, .stApp h3 { font-family: var(--xt-display); font-weight: 400; letter-spacing: -0.01em; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 1120px; padding-top: 4rem; padding-bottom: 4rem; }

/* Streamlit gives markdown blocks a -1rem bottom margin (for paragraph spacing); restore it */
.xt { color: var(--xt-ink); font-family: var(--xt-body); padding-bottom: 1rem; }
.xt p { margin: 0; }
.xt a { color: var(--xt-ink); text-decoration-color: rgba(168, 68, 30, .55); text-decoration-thickness: 1.5px; text-underline-offset: 3px; }
.xt a:hover { color: var(--xt-accent); text-decoration-color: var(--xt-accent); }

/* Header */
.xt-head { padding: 0 0 6px; }
.xt-eyebrow { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; font-size: .8rem; letter-spacing: .06em; text-transform: uppercase; color: var(--xt-muted); }
.xt-pill { display: inline-block; padding: 2px 9px; border: 1px solid var(--xt-rule-strong); border-radius: 999px; font-size: .7rem; letter-spacing: .08em; color: var(--xt-ink-2); background: var(--xt-surface); }
.xt-title { font-family: var(--xt-display); font-weight: 400; font-size: clamp(2.25rem, 5vw, 3.25rem); line-height: 1.05; letter-spacing: -0.02em; margin: 14px 0 12px; color: var(--xt-ink); }
.xt .xt-lede { font-size: 1.08rem; line-height: 1.6; color: var(--xt-ink-2); max-width: 46rem; }
.xt-links { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 18px; margin-top: 14px; font-size: .92rem; }
.xt-links .xt-sep { color: var(--xt-rule-strong); }

/* Section headings and notes */
.xt .xt-h { font-family: var(--xt-display); font-weight: 400; font-size: 1.5rem; line-height: 1.2; letter-spacing: -0.01em; color: var(--xt-ink); margin: 18px 0 2px; }
.xt .xt-sub { font-size: .92rem; color: var(--xt-muted); margin: 2px 0 12px; }
.xt-note { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; padding: 12px 16px; border: 1px solid var(--xt-rule); border-radius: 12px; background: var(--xt-surface); font-size: .93rem; color: var(--xt-ink-2); }
.xt-note strong { color: var(--xt-ink); font-weight: 600; }
.xt-note .xt-note-meta { color: var(--xt-muted); }

/* Stat tiles */
.xt-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.xt-stat { background: var(--xt-surface); border: 1px solid var(--xt-rule); border-radius: 12px; padding: 14px 16px 13px; min-width: 0; }
.xt-stat-val { font-family: var(--xt-display); font-size: 2.1rem; line-height: 1.1; color: var(--xt-ink); font-variant-numeric: tabular-nums; }
.xt-stat-label { display: flex; align-items: center; gap: 7px; margin-top: 6px; font-size: .88rem; font-weight: 500; color: var(--xt-ink-2); }
.xt-stat-rule { font-size: .8rem; color: var(--xt-muted); margin-top: 1px; }
.xt-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; display: inline-block; }

/* Timeline */
.xt-panel { background: var(--xt-surface); border: 1px solid var(--xt-rule); border-radius: var(--xt-radius); padding: 18px 20px 16px; }
.xt-track { position: relative; height: 46px; border-radius: 8px; overflow: hidden; background: var(--xt-sunk); }
.xt-piece { position: absolute; top: 0; bottom: 0; display: flex; align-items: center; gap: 7px; padding: 0 10px; overflow: hidden; white-space: nowrap; font-size: .8rem; font-weight: 500; container-type: inline-size; }
.xt-piece + .xt-piece { box-shadow: inset 2px 0 0 var(--xt-surface); }
.xt-piece-num { font-weight: 600; font-variant-numeric: tabular-nums; }
.xt-piece-name { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
@container (max-width: 64px) { .xt-piece-name { display: none; } }
.xt-axis { position: relative; height: 24px; margin-top: 4px; }
.xt-tick { position: absolute; top: 0; height: 100%; }
.xt-tick::before { content: ""; position: absolute; top: 0; left: 0; width: 1px; height: 5px; background: var(--xt-rule-strong); }
.xt-tick span { position: absolute; top: 7px; left: 0; transform: translateX(-50%); font-size: .75rem; color: var(--xt-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.xt-tick:first-child span { transform: none; }
.xt-tick.xt-tick-end::before { left: -1px; }
.xt-tick.xt-tick-end span { transform: translateX(-100%); }
.xt-legend { display: flex; flex-wrap: wrap; gap: 8px 22px; margin-top: 10px; font-size: .88rem; color: var(--xt-ink-2); }
.xt-legend-title { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--xt-muted); align-self: center; }
.xt-key { display: inline-flex; align-items: center; gap: 8px; }
.xt-swatch { width: 12px; height: 12px; border-radius: 3px; flex: none; display: inline-block; }
.xt-key b { font-weight: 600; color: var(--xt-ink); font-variant-numeric: tabular-nums; }
.xt-key .xt-key-meta { color: var(--xt-muted); }
.xt .xt-foot { margin-top: 10px; font-size: .8rem; color: var(--xt-muted); line-height: 1.5; }

/* Routing callout */
.xt .xt-callout { border-left: 3px solid var(--xt-accent); margin: 8px 0 6px; padding: 4px 0 4px 16px; font-size: 1rem; line-height: 1.65; color: var(--xt-ink-2); max-width: 52rem; }

/* Segment cards */
.xt-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr)); gap: 14px; }
.xt-card { background: var(--xt-surface); border: 1px solid var(--xt-rule); border-radius: var(--xt-radius); padding: 16px 18px 16px; min-width: 0; }
.xt-card-top { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; font-size: .85rem; color: var(--xt-muted); }
.xt-card-top .xt-time { color: var(--xt-ink); font-weight: 500; font-variant-numeric: tabular-nums; }
.xt-card-main { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 12px; }
.xt-chip { display: inline-flex; align-items: center; padding: 3px 11px; border-radius: 999px; font-size: .82rem; font-weight: 600; letter-spacing: .01em; }
.xt-conf { font-family: var(--xt-display); font-size: 1.9rem; line-height: 1; color: var(--xt-ink); font-variant-numeric: tabular-nums; }
.xt-conf small { font-family: var(--xt-body); font-size: .72rem; color: var(--xt-muted); margin-left: 4px; letter-spacing: .02em; }
.xt-meter { position: relative; height: 6px; margin-top: 12px; border-radius: 3px; background: var(--xt-sunk); }
.xt-meter-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; }
.xt-mark { position: absolute; top: -3px; width: 1px; height: 12px; background: var(--xt-ink-2); }
.xt-marks { position: relative; height: 14px; margin-top: 3px; font-size: .68rem; color: var(--xt-muted); font-variant-numeric: tabular-nums; }
.xt-marks span { position: absolute; transform: translateX(-50%); }
.xt-route { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; margin-top: 8px; font-size: .84rem; color: var(--xt-muted); }
.xt-tier { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.xt-tier-auto { color: var(--xt-auto-fg); background: var(--xt-auto-bg); }
.xt-tier-qa { color: var(--xt-qa-fg); background: var(--xt-qa-bg); }
.xt-tier-human { color: var(--xt-human-fg); background: var(--xt-human-bg); }
.xt-flag { display: inline-flex; align-items: center; gap: 6px; margin-top: 10px; padding: 3px 10px; border: 1px solid var(--xt-qa-dot); border-radius: 999px; font-size: .78rem; font-weight: 600; color: var(--xt-qa-fg); background: var(--xt-qa-bg); }
.xt-block { margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--xt-rule); }
.xt-block-title { display: flex; justify-content: space-between; gap: 8px; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--xt-muted); margin-bottom: 8px; }
.xt-block-title span { font-weight: 400; letter-spacing: .02em; text-transform: none; }
.xt-sig { display: grid; grid-template-columns: 4.2rem 1fr auto; align-items: center; gap: 4px 10px; font-size: .9rem; }
.xt-sig-src { color: var(--xt-muted); }
.xt-sig-cat { display: inline-flex; align-items: center; gap: 7px; color: var(--xt-ink); }
.xt-sig-val { font-variant-numeric: tabular-nums; color: var(--xt-ink); font-weight: 500; }
.xt-sig-na { color: var(--xt-muted); font-style: italic; }
.xt .xt-verdict { margin-top: 8px; font-size: .84rem; line-height: 1.5; color: var(--xt-ink-2); }
.xt .xt-quote { font-size: .84rem; line-height: 1.5; color: var(--xt-ink-2); }
.xt-evidence { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 12px; }
.xt-evidence div { font-size: .8rem; color: var(--xt-muted); }
.xt-evidence b { display: block; font-size: .92rem; font-weight: 500; color: var(--xt-ink); font-variant-numeric: tabular-nums; }

/* How it works */
.xt-steps { margin: 4px 0 0; padding: 0; list-style: none; counter-reset: xt; }
.xt-steps li { counter-increment: xt; position: relative; padding: 0 0 12px 34px; font-size: .95rem; line-height: 1.6; color: var(--xt-ink-2); }
.xt-steps li::before { content: counter(xt); position: absolute; left: 0; top: 1px; width: 22px; height: 22px; border-radius: 50%; border: 1px solid var(--xt-rule-strong); font-size: .72rem; font-weight: 600; color: var(--xt-ink-2); display: flex; align-items: center; justify-content: center; }
.xt-steps strong { color: var(--xt-ink); font-weight: 600; }
.xt-tiers { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 6px 0 14px; }
.xt-tierbox { border: 1px solid var(--xt-rule); border-radius: 12px; padding: 12px 14px; background: var(--xt-surface); font-size: .86rem; line-height: 1.5; color: var(--xt-ink-2); }
.xt-tierbox .xt-tier { margin-bottom: 6px; }
.xt .xt-prose { font-size: .95rem; line-height: 1.65; color: var(--xt-ink-2); max-width: 50rem; }
.xt .xt-prose + .xt-prose { margin-top: 10px; }
.xt .xt-kicker { font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--xt-muted); margin: 14px 0 8px; }

.xt .xt-try { margin: 26px 0 10px; }
.xt .xt-try .xt-h { margin: 0; }
.xt .xt-try .xt-sub { margin: 4px 0 0; }

/* The two ways in (demo or your own clip) are big buttons, not quiet tab labels, so the
   live pipeline is as easy to find as the demo. Unselected "own clip" stays outlined in the accent. */
.stTabs [role="tablist"] { gap: 12px; padding: 0; border: 0; box-shadow: none; margin-bottom: 18px; }
.stTabs [role="tablist"]::after, .stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [role="tab"] {
  flex: 1 1 0; height: auto; min-height: 60px; margin: 0; padding: 12px 18px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  border: 1.5px solid var(--xt-rule-strong); border-radius: 12px; background: var(--xt-surface);
  color: var(--xt-ink-2); transition: background .15s, border-color .15s, color .15s;
}
.stTabs [role="tab"] p { font-size: 1.05rem; font-weight: 600; color: inherit; white-space: normal; text-align: center; }
.stTabs [role="tab"] span[role="img"], .stTabs [role="tab"] [data-testid="stIconMaterial"] { color: inherit; font-size: 1.35rem; vertical-align: -0.28em; }
.stTabs [role="tab"]:nth-of-type(2) { border-color: var(--xt-accent); color: var(--xt-accent); }
.stTabs [role="tab"]:hover { border-color: var(--xt-accent); color: var(--xt-accent); background: var(--xt-human-bg); }
.stTabs [role="tab"][aria-selected="true"],
.stTabs [role="tab"][aria-selected="true"]:hover { background: var(--xt-accent); border-color: var(--xt-accent); color: #fff; }
.stTabs [role="tab"]:focus-visible { outline: 2px solid var(--xt-accent); outline-offset: 2px; }

.xt-footer { margin-top: 28px; padding-top: 16px; border-top: 1px solid var(--xt-rule); font-size: .85rem; color: var(--xt-muted); }

@media (max-width: 720px) {
  .xt-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .xt-stat-val { font-size: 1.8rem; }
  .xt-tiers { grid-template-columns: 1fr; }
  .xt-panel { padding: 16px 14px 14px; }
  .xt-legend { gap: 6px 16px; }
}
@media (max-width: 520px) {
  .xt-piece-name { display: none; }
  .xt-legend-title { flex-basis: 100%; }
  .xt-piece { justify-content: center; padding: 0 4px; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --- Helpers -----------------------------------------------------------------

def esc(value) -> str:
    return html.escape(str(value), quote=True)


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def category_style(category: str) -> tuple[str, str, str]:
    return CATEGORIES.get(category, (str(category).title(), "#9b968b", "#1c1b18"))


def get_confidence_tier(conf: float) -> str:
    if conf >= 0.9:
        return "auto"
    elif conf >= 0.7:
        return "qa"
    return "human"


def tier_badge(tier: str) -> str:
    return f'<span class="xt-tier xt-tier-{tier}">{TIERS[tier]["label"]}</span>'


def md(markup: str) -> None:
    """Render one block of custom HTML. Kept on one line so Markdown never re-parses it."""
    st.markdown(f'<div class="xt">{markup}</div>', unsafe_allow_html=True)


def sorted_segments(segments: list[dict]) -> list[dict]:
    return sorted(segments, key=lambda s: (s["start_time"], s["end_time"]))


def runtime_share(segments: list[dict]) -> dict[str, float]:
    """Seconds per category. Adjacent clips overlap, so each overlap is split at its midpoint."""
    seconds: dict[str, float] = {}
    for i, seg in enumerate(segments):
        start, end = seg["start_time"], seg["end_time"]
        if i > 0 and segments[i - 1]["end_time"] > start:
            start = (start + segments[i - 1]["end_time"]) / 2
        if i + 1 < len(segments) and segments[i + 1]["start_time"] < end:
            end = (end + segments[i + 1]["start_time"]) / 2
        seconds[seg["category"]] = seconds.get(seg["category"], 0.0) + max(0.0, end - start)
    return dict(sorted(seconds.items(), key=lambda kv: -kv[1]))


def explain(seg: dict, mode: str) -> dict:
    """Recover the per-signal story for a segment from what the pipeline recorded."""
    desc = seg.get("description", "") or ""
    scores = seg.get("scores") or {}
    audio = None
    if scores:
        top = max(scores, key=scores.get)
        audio = (top, round(scores[top] * 100))

    match = HYBRID_RE.search(desc)
    if match:
        vision = (match.group(1).lower(), int(match.group(2)))
        audio = (match.group(3).lower(), int(match.group(4)))
        return {"kind": "fused", "vision": vision, "audio": audio, "agree": vision[0] == audio[0]}
    if mode == "hybrid":
        return {"kind": "audio_fallback", "vision": None,
                "audio": audio or (seg["category"], round(seg["confidence"] * 100))}
    if mode == "vlm" or desc.startswith("VLM:"):
        text = desc[4:].strip() if desc.startswith("VLM:") else desc
        return {"kind": "vision", "vision": (seg["category"], round(seg["confidence"] * 100)),
                "audio": None, "quote": text}
    return {"kind": "audio", "vision": None,
            "audio": audio or (seg["category"], round(seg["confidence"] * 100))}


# --- Rendering ---------------------------------------------------------------

def render_header() -> None:
    md(
        '<header class="xt-head">'
        '<div class="xt-eyebrow"><span class="xt-pill">Prototype</span><span>AI content classifier</span></div>'
        '<div class="xt-title" role="heading" aria-level="1">JioTV+ Xtras</div>'
        '<p class="xt-lede">Splits a film clip into segments, labels each one as musical, dialogue, '
        'action, credits or transition from what it hears and sees, then routes every label by how '
        'confident the model is: auto-tag, QA review or a human annotator.</p>'
        '<nav class="xt-links" aria-label="Project links">'
        f'<a href="{CASE_STUDY_URL}" target="_blank" rel="noopener">Read the case study</a>'
        '<span class="xt-sep" aria-hidden="true">/</span>'
        f'<a href="{SOURCE_URL}" target="_blank" rel="noopener">Source on GitHub</a>'
        '</nav>'
        '</header>'
    )


def render_how_it_works() -> None:
    with st.expander("How it works"):
        tiers_html = "".join(
            f'<div class="xt-tierbox">{tier_badge(key)}<div><strong>{t["rule"]}</strong>: '
            f'{t["action"]}.</div></div>'
            for key, t in TIERS.items()
        )
        md(
            '<p class="xt-kicker">Pipeline</p>'
            '<ol class="xt-steps">'
            '<li><strong>Cut.</strong> The video is split into short overlapping clips (10 seconds '
            'with 2 seconds of overlap by default), so a scene change never falls blind between two windows.</li>'
            '<li><strong>Listen.</strong> librosa measures each clip&#8217;s audio: tempo, beat strength, '
            'harmonic ratio, spectral centroid, zero-crossing rate and loudness. Weighted rules turn those '
            'into musical, dialogue and action scores.</li>'
            '<li><strong>Look</strong> (vision and hybrid modes). Sampled frames go to a small '
            'vision-language model on a local Ollama server. It describes each frame, keywords map the '
            'description to one of the five labels, and the frames vote.</li>'
            '<li><strong>Fuse.</strong> In hybrid mode, if vision is more than 40% sure its label wins and '
            'confidence blends 60% vision with 40% audio; otherwise audio decides. Neighbouring clips with '
            'the same label merge into one segment.</li>'
            '<li><strong>Route.</strong> Every segment goes to one of three queues by confidence.</li>'
            '</ol>'
            f'<div class="xt-tiers">{tiers_html}</div>'
            '<p class="xt-kicker">Why route by confidence</p>'
            '<p class="xt-prose">A wrong label on screen, like a song marker over a fight scene, costs more '
            'viewer trust than a missing one. Routing lets automation handle what the models are sure of '
            'while people check the ambiguous stretches, so quality holds without labelling every hour of '
            'content by hand.</p>'
            '<p class="xt-prose">Prototype limits: the audio rules can only tell musical, dialogue and '
            'action apart, so credits and transitions need the vision model; thresholds are hand-set, not '
            'learned.</p>'
        )


def render_stats(segments: list[dict]) -> None:
    counts = {key: 0 for key in TIERS}
    for seg in segments:
        counts[get_confidence_tier(seg["confidence"])] += 1
    tiles = [
        f'<div class="xt-stat"><div class="xt-stat-val">{len(segments)}</div>'
        f'<div class="xt-stat-label">Segments</div>'
        f'<div class="xt-stat-rule">in this run</div></div>'
    ]
    for key, tier in TIERS.items():
        tiles.append(
            f'<div class="xt-stat"><div class="xt-stat-val">{counts[key]}</div>'
            f'<div class="xt-stat-label"><span class="xt-dot" style="background:var(--xt-{key}-dot)"></span>'
            f'{tier["label"]}</div><div class="xt-stat-rule">{tier["rule"]}</div></div>'
        )
    md(f'<div class="xt-stats" role="list">{"".join(tiles)}</div>')


def axis_ticks(total: float) -> str:
    step = 3600
    for candidate in (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600):
        if total / candidate <= 6:
            step = candidate
            break
    ticks, t = [], 0
    while t < total * 0.88:
        ticks.append(f'<span class="xt-tick" style="left:{t / total * 100:.3f}%"><span>{format_time(t)}</span></span>')
        t += step
    ticks.append(f'<span class="xt-tick xt-tick-end" style="left:100%"><span>{format_time(total)}</span></span>')
    return "".join(ticks)


def render_timeline(segments: list[dict], total_duration: float, clip_overlap=None, intro: str = "") -> None:
    if not segments or total_duration <= 0:
        return

    def pct(t: float) -> float:
        return max(0.0, min(100.0, t / total_duration * 100))

    pieces, aria, has_overlap = [], [], False
    n = len(segments)
    for i, seg in enumerate(segments):
        start, end = seg["start_time"], seg["end_time"]
        label, fill, on = category_style(seg["category"])
        prev_end = segments[i - 1]["end_time"] if i > 0 else None
        next_start = segments[i + 1]["start_time"] if i + 1 < n else None
        solo_start = max(start, prev_end) if prev_end is not None and prev_end > start else start
        solo_end = min(end, next_start) if next_start is not None and next_start < end else end
        conf = f'{seg["confidence"] * 100:.0f}%'
        tip = f'{i + 1}. {label}, {format_time(start)}–{format_time(end)}, {conf} confidence'
        aria.append(f'{label} {format_time(start)} to {format_time(end)}')
        if solo_end > solo_start:
            left, width = pct(solo_start), pct(solo_end) - pct(solo_start)
            pieces.append(
                f'<div class="xt-piece" style="left:{left:.3f}%;width:{width:.3f}%;background:{fill};color:{on}" '
                f'title="{esc(tip)}"><span class="xt-piece-num">{i + 1}</span>'
                f'<span class="xt-piece-name">{label}</span></div>'
            )
        if next_start is not None and next_start < end:
            has_overlap = True
            ov_end = min(end, segments[i + 1]["end_time"])
            _, fill_next, _ = category_style(segments[i + 1]["category"])
            left, width = pct(next_start), pct(ov_end) - pct(next_start)
            pieces.append(
                f'<div class="xt-piece" style="left:{left:.3f}%;width:{width:.3f}%;'
                f'background:repeating-linear-gradient(135deg,{fill} 0 4px,{fill_next} 4px 8px)" '
                f'title="Clip overlap, {format_time(next_start)}–{format_time(ov_end)}"></div>'
            )

    legend = []
    for cat, secs in runtime_share(segments).items():
        label, fill, _ = category_style(cat)
        legend.append(
            f'<span class="xt-key"><span class="xt-swatch" style="background:{fill}"></span>{label} '
            f'<b>{secs / total_duration * 100:.0f}%</b><span class="xt-key-meta">{format_time(secs)}</span></span>'
        )
    if has_overlap:
        _, a, _ = category_style(segments[0]["category"])
        legend.append(
            '<span class="xt-key"><span class="xt-swatch" style="background:repeating-linear-gradient('
            f'135deg,{a} 0 3px,var(--xt-surface) 3px 6px)"></span>Clip overlap</span>'
        )

    foot = ""
    if has_overlap:
        overlap_txt = f"{clip_overlap:g} seconds" if isinstance(clip_overlap, (int, float)) else "a little"
        foot = (f'<p class="xt-foot">Neighbouring clips overlap by {overlap_txt} (hatched); '
                'runtime shares split each overlap at its midpoint.</p>')

    md(
        f'{intro}<div class="xt-panel">'
        f'<div class="xt-track" role="img" aria-label="Timeline: {esc("; ".join(aria))}">{"".join(pieces)}</div>'
        f'<div class="xt-axis" aria-hidden="true">{axis_ticks(total_duration)}</div>'
        f'<div class="xt-legend"><span class="xt-legend-title">Share of runtime</span>{"".join(legend)}</div>'
        f'{foot}'
        '</div>'
    )


def render_routing_summary(segments: list[dict], mode: str, is_demo: bool) -> None:
    n = len(segments)
    counts = {key: 0 for key in TIERS}
    for seg in segments:
        counts[get_confidence_tier(seg["confidence"])] += 1

    if counts["human"] == n:
        lead = (f"Every segment scored under 70%, so all {n} go to a human annotator instead of being auto-tagged."
                if n > 1 else "The only segment scored under 70%, so it goes to a human annotator.")
    elif counts["auto"] == n:
        lead = "Every segment cleared 90%, so all of them can be auto-tagged without review."
    else:
        lead = (f"{counts['auto']} of {n} segments cleared 90% and can be auto-tagged; "
                f"{counts['qa']} {'goes' if counts['qa'] == 1 else 'go'} to QA review and "
                f"{counts['human']} to a human annotator.")

    sentences = [lead]
    disagreements = []
    for i, seg in enumerate(segments):
        info = explain(seg, mode)
        if info["kind"] == "fused" and not info["agree"]:
            disagreements.append((i, seg, info))
    if len(disagreements) == 1:
        i, seg, info = disagreements[0]
        sentences.append(
            f"In segment {i + 1} ({format_time(seg['start_time'])}–{format_time(seg['end_time'])}) "
            f"vision said {info['vision'][0]} while audio said {info['audio'][0]}."
        )
        if is_demo:
            sentences.append("The clip is made entirely of songs, so that label is probably wrong, and it is "
                             "exactly the kind of call the routing keeps off the screen.")
    elif disagreements:
        nums = ", ".join(str(i + 1) for i, _, _ in disagreements)
        sentences.append(f"Vision and audio disagreed in segments {nums}.")
    if counts["auto"] < n and not (is_demo and len(disagreements) == 1):
        sentences.append("That is what routing is for: tag what the models are sure of and send the uncertain "
                         "stretches to a person before a wrong label reaches viewers.")
    md(f'<p class="xt-callout">{esc(" ".join(sentences))}</p>')


def render_segment_cards(segments: list[dict], mode: str, clip_duration=None, intro: str = "") -> None:
    cards = []
    for i, seg in enumerate(segments):
        label, fill, on = category_style(seg["category"])
        conf = seg["confidence"]
        tier = get_confidence_tier(conf)
        dur = seg.get("duration", seg["end_time"] - seg["start_time"])
        info = explain(seg, mode)

        flag = ""
        if info["kind"] == "fused" and not info["agree"]:
            flag = ('<div class="xt-flag"><span class="xt-dot" style="background:var(--xt-qa-dot)"></span>'
                    'Audio and vision disagree</div>')

        def sig_row(source: str, value) -> str:
            if value is None:
                return (f'<span class="xt-sig-src">{source}</span><span class="xt-sig-na">inconclusive</span>'
                        '<span class="xt-sig-val"></span>')
            cat, p = value
            c_label, c_fill, _ = category_style(cat)
            return (f'<span class="xt-sig-src">{source}</span>'
                    f'<span class="xt-sig-cat"><span class="xt-swatch" style="background:{c_fill}"></span>{c_label}</span>'
                    f'<span class="xt-sig-val">{p}%</span>')

        rows, verdict = [], ""
        if info["kind"] == "fused":
            rows = [sig_row("Vision", info["vision"]), sig_row("Audio", info["audio"])]
            verdict = ("Both signals agree." if info["agree"] else
                       "Vision cleared the 40% bar, so its label wins; confidence blends 60% vision, 40% audio.")
        elif info["kind"] == "audio_fallback":
            rows = [sig_row("Vision", None), sig_row("Audio", info["audio"])]
            verdict = "Vision was 40% sure or less on this clip, so the label comes from audio alone."
        elif info["kind"] == "vision":
            rows = [sig_row("Vision", info["vision"])]
            quote = info.get("quote", "")
            if quote:
                verdict = f'Frame description: &#8220;{esc(quote)}&#8221;'
        else:
            rows = [sig_row("Audio", info["audio"])]

        first_clip = ""
        if isinstance(clip_duration, (int, float)) and dur > clip_duration + 0.01:
            first_clip = "<span>first clip in segment</span>"
        signals = (f'<div class="xt-block"><div class="xt-block-title">How it was labelled{first_clip}</div>'
                   f'<div class="xt-sig">{"".join(rows)}</div>'
                   + (f'<p class="xt-verdict">{verdict}</p>' if verdict else "") + '</div>')

        evidence = ""
        feats = seg.get("features") or {}
        if "tempo" in feats:
            items = [
                ("Tempo", f'{feats["tempo"]:.0f} BPM'),
                ("Beat strength", f'{feats.get("beat_strength", 0):.1f} per s'),
                ("Harmonic ratio", f'{feats.get("harmonic_ratio", 0):.2f}'),
                ("Spectral centroid", f'{feats.get("centroid_mean", 0):,.0f} Hz'),
            ]
            evidence = ('<div class="xt-block"><div class="xt-block-title">Audio evidence</div>'
                        '<div class="xt-evidence">'
                        + "".join(f'<div><b>{v}</b>{k}</div>' for k, v in items)
                        + '</div></div>')

        cards.append(
            '<article class="xt-card">'
            f'<div class="xt-card-top"><span>Segment {i + 1}</span>'
            f'<span><span class="xt-time">{format_time(seg["start_time"])}–{format_time(seg["end_time"])}</span>'
            f' · {dur:.0f} s</span></div>'
            '<div class="xt-card-main">'
            f'<span class="xt-chip" style="background:{fill};color:{on}">{label}</span>'
            f'<span class="xt-conf">{conf * 100:.0f}%<small>confidence</small></span></div>'
            f'<div class="xt-meter" aria-hidden="true"><div class="xt-meter-fill" '
            f'style="width:{conf * 100:.1f}%;background:var(--xt-{tier}-dot)"></div>'
            '<span class="xt-mark" style="left:70%"></span><span class="xt-mark" style="left:90%"></span></div>'
            '<div class="xt-marks" aria-hidden="true"><span style="left:70%">70</span><span style="left:90%">90</span></div>'
            f'<div class="xt-route"><span>Routed to</span>{tier_badge(tier)}</div>'
            f'{flag}{signals}{evidence}'
            '</article>'
        )
    md(f'{intro}<div class="xt-cards">{"".join(cards)}</div>')


def render_result(data: dict, *, is_demo: bool) -> None:
    segments = sorted_segments(data.get("segments", []))
    total = float(data.get("total_duration") or 0)
    mode = data.get("analysis_mode", "audio")
    clip_duration = data.get("clip_duration")
    clip_overlap = data.get("clip_overlap")

    if not segments:
        st.warning("No segments were detected. Try a different clip, or a different analysis mode.")
        return

    render_stats(segments)
    render_timeline(segments, total, clip_overlap, intro=(
        '<div class="xt-h">Timeline</div>'
        '<p class="xt-sub">Each bar is one labelled segment; the numbers match the cards below.</p>'))
    render_routing_summary(segments, mode, is_demo)
    render_segment_cards(segments, mode, clip_duration, intro=(
        '<div class="xt-h">Segments</div>'
        '<p class="xt-sub">Label, confidence, routing, and the signals behind each call.</p>'))
    with st.expander("Raw output (JSON)"):
        st.json(data, expanded=False)


def load_precached_results() -> dict | None:
    cache_path = Path(__file__).parent / "demo_results.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=30, show_spinner=False)
def ollama_status() -> tuple[bool, list[str]]:
    from vlm_classifier import check_ollama_available
    return check_ollama_available()


# --- Live pipeline (logic unchanged from the original app) --------------------

class PipelineError(Exception):
    pass


def run_pipeline(video_path: str, video_name: str, mode_key: str, vlm_model, clip_duration: int,
                 clip_overlap: int, progress) -> tuple[dict, str | None]:
    """Run the analysis. Returns (result, warning). Raises PipelineError on fatal steps."""
    from video_processor import get_video_duration, extract_audio
    from audio_analyzer import analyze_full_audio

    warning = None
    progress.progress(5, text="Reading video...")
    try:
        total_duration = get_video_duration(video_path)
    except Exception:
        raise PipelineError("Couldn't read this file as a video. Try an .mp4 or .mov export of the clip.")

    progress.progress(10, text="Extracting audio...")
    audio_path = video_path.replace(".mp4", "_audio.wav")
    try:
        extract_audio(video_path, audio_path)
    except Exception:
        raise PipelineError("Couldn't extract an audio track from this video. Check that the clip has sound.")

    try:
        progress.progress(30, text="Analysing audio features (librosa)...")
        skip_merge = mode_key in ("vlm", "hybrid")
        try:
            audio_segments = analyze_full_audio(
                audio_path,
                segment_duration=float(clip_duration),
                overlap=float(clip_overlap),
                merge=not skip_merge
            )
        except Exception as e:
            warning = f"Audio analysis failed ({type(e).__name__}), so audio signals are missing from this run."
            audio_segments = []

        segments = audio_segments

        if mode_key in ("vlm", "hybrid") and vlm_model:
            progress.progress(50, text="Running VLM classification on frames...")
            from video_processor import segment_video_clips, extract_clip_frames
            from vlm_classifier import classify_frames_batch

            clips = segment_video_clips(video_path, clip_duration, clip_overlap)

            for i, clip in enumerate(clips):
                pct = 50 + int(40 * i / len(clips))
                progress.progress(pct, text=f"VLM analysing clip {i + 1} of {len(clips)}...")

                with tempfile.TemporaryDirectory() as frame_dir:
                    try:
                        frames = extract_clip_frames(
                            video_path, clip["start_time"], clip["end_time"],
                            frame_dir, sample_every_n=3
                        )
                        if frames:
                            vlm_result = classify_frames_batch(frames[:3], vlm_model)
                            clip["vlm_category"] = vlm_result["category"]
                            clip["vlm_confidence"] = vlm_result["confidence"]
                            clip["vlm_description"] = vlm_result["description"]
                    except Exception:
                        clip["vlm_category"] = "unknown"
                        clip["vlm_confidence"] = 0.0

            if mode_key == "hybrid":
                from audio_analyzer import merge_adjacent_segments
                fused = []
                for clip in clips:
                    audio_match = next(
                        (a for a in audio_segments
                         if a["start_time"] == clip["start_time"]),
                        None
                    )
                    audio_cat = audio_match["category"] if audio_match else "unknown"
                    audio_conf = audio_match["confidence"] if audio_match else 0.0
                    audio_scores = audio_match.get("scores", {}) if audio_match else {}
                    audio_features = audio_match.get("features", {}) if audio_match else {}

                    vlm_cat = clip.get("vlm_category", "unknown")
                    vlm_conf = clip.get("vlm_confidence", 0.0)

                    if vlm_cat != "unknown" and vlm_conf > 0.4:
                        final_cat = vlm_cat
                        final_conf = round(vlm_conf * 0.6 + audio_conf * 0.4, 3)
                        desc = f"[Hybrid] VLM: {vlm_cat} ({vlm_conf:.0%}) + Audio: {audio_cat} ({audio_conf:.0%})"
                    else:
                        final_cat = audio_cat
                        final_conf = audio_conf
                        desc = audio_match.get("description", "") if audio_match else ""

                    fused.append({
                        "clip_id": clip["clip_id"],
                        "start_time": clip["start_time"],
                        "end_time": clip["end_time"],
                        "duration": clip["duration"],
                        "category": final_cat,
                        "confidence": final_conf,
                        "scores": audio_scores,
                        "features": audio_features,
                        "description": desc,
                    })
                segments = merge_adjacent_segments(fused)

            elif mode_key == "vlm":
                from audio_analyzer import merge_adjacent_segments
                segments = []
                for clip in clips:
                    segments.append({
                        "clip_id": clip["clip_id"],
                        "start_time": clip["start_time"],
                        "end_time": clip["end_time"],
                        "duration": clip["duration"],
                        "category": clip.get("vlm_category", "unknown"),
                        "confidence": clip.get("vlm_confidence", 0.0),
                        "description": clip.get("vlm_description", ""),
                    })
                segments = merge_adjacent_segments(segments)
    finally:
        if os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    progress.progress(95, text="Building the timeline...")
    result = {
        "video_name": video_name,
        "total_duration": total_duration,
        "analysis_mode": mode_key,
        "clip_duration": clip_duration,
        "clip_overlap": clip_overlap,
        "segments": segments,
    }
    progress.progress(100, text="Analysis complete")
    time.sleep(0.3)
    return result, warning


# --- Page --------------------------------------------------------------------

render_header()
render_how_it_works()

md('<div class="xt-try"><p class="xt-h">Try it</p>'
   '<p class="xt-sub">See a finished run on a film clip, or upload your own and run the pipeline live.</p></div>')

tab_demo, tab_own = st.tabs([":material/play_circle: See the demo result", ":material/upload: Analyse your own clip"])

with tab_demo:
    demo_data = load_precached_results()
    if demo_data:
        md(
            '<div class="xt-note"><strong>Demo run: a 63-second clip stitched from five film songs, '
            'hybrid mode.</strong>'
            f'<span class="xt-note-meta">Pre-computed with {demo_data.get("clip_duration", 10)}-second clips and '
            f'{demo_data.get("clip_overlap", 2)} seconds of overlap, so nothing is processed when the page loads.</span></div>'
        )
        render_result(demo_data, is_demo=True)
    else:
        st.info("The demo result (demo_results.json) isn't available. Use “Analyse your own clip” above to run one.")

with tab_own:
    md('<p class="xt-prose">Run the live pipeline on a short clip; 1 to 5 minutes works best. '
       'The upload is saved to a temporary file and deleted once the analysis finishes.</p>')

    with st.container(border=True):
        mode_options = ["audio", "vlm", "hybrid"]
        mode_key = st.radio(
            "Analysis mode",
            mode_options,
            format_func=lambda k: {"audio": "Audio only (librosa)", "vlm": "Vision (VLM via Ollama)",
                                   "hybrid": "Hybrid: audio + vision"}[k],
            horizontal=True,
        )
        st.caption({
            "audio": "Reads tempo, beats, harmony and speech-like spectra. Runs anywhere ffmpeg is installed, "
                     "including this hosted demo. Labels: musical, dialogue, action.",
            "vlm": "A vision-language model describes sampled frames. Needs a local Ollama server, "
                   "so it won't run on the hosted demo.",
            "hybrid": "Both signals, fused per clip. Needs a local Ollama server for the vision half, "
                      "so it won't run on the hosted demo.",
        }[mode_key])

        c1, c2 = st.columns(2)
        with c1:
            clip_duration = st.slider("Clip length (seconds)", 5, 30, 10,
                                      help="Length of each analysis window.")
        with c2:
            # Overlap must stay below the clip length or the windowing loop never advances.
            max_overlap = min(5, clip_duration - 1)
            clip_overlap = st.slider("Overlap (seconds)", 0, max_overlap, min(2, max_overlap),
                                     help="How much neighbouring windows share, so boundaries aren't missed.")

        vlm_model = None
        vision_ready = True
        if mode_key in ("vlm", "hybrid"):
            available, vision_models = ollama_status()
            if available and vision_models:
                vlm_model = st.selectbox("Vision model", vision_models)
            elif available:
                vision_ready = False
                st.warning("Ollama is running but has no vision model. Pull one with `ollama pull moondream`, "
                           "then reload this page.")
            else:
                vision_ready = False
                st.info("Vision and hybrid modes call a vision-language model on a local Ollama server "
                        "(localhost:11434), and none is reachable here, which is expected on the hosted demo. "
                        "To use them, run the app on your machine with `ollama serve` and a vision model "
                        "(for example `ollama pull moondream`). Audio only works here.")

    ffmpeg_ready = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    if not ffmpeg_ready:
        st.warning("This environment has no ffmpeg, which the pipeline needs to read video. Install it "
                   "(`brew install ffmpeg` or `apt install ffmpeg`) to analyse a clip. The demo result "
                   "doesn't need it.")

    uploaded_file = st.file_uploader(
        "Video clip (.mp4, .mov, .avi, .mkv)",
        type=["mp4", "mov", "avi", "mkv"],
        help="A short clip from any film. Song-and-dance and action scenes are the easiest to detect.",
    )

    if uploaded_file is not None:
        preview_col, _ = st.columns([3, 2])
        with preview_col:
            st.video(uploaded_file)

        can_run = ffmpeg_ready and vision_ready
        run = st.button("Analyse clip", type="primary", disabled=not can_run)
        if not can_run:
            st.caption("Analysis is unavailable with the current settings; see the note above.")

        if run:
            progress = st.progress(0, text="Starting...")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(uploaded_file.getvalue())
                video_path = tmp.name
            try:
                result, run_warning = run_pipeline(video_path, uploaded_file.name, mode_key, vlm_model,
                                                   clip_duration, clip_overlap, progress)
                st.session_state["xt_result"] = result
                st.session_state["xt_warning"] = run_warning
            except PipelineError as e:
                st.session_state.pop("xt_result", None)
                st.error(str(e))
            except Exception as e:
                st.session_state.pop("xt_result", None)
                st.error(f"The analysis stopped unexpectedly ({type(e).__name__}). Try a shorter clip or "
                         "Audio only mode.")
            finally:
                progress.empty()
                try:
                    os.unlink(video_path)
                except OSError:
                    pass

        result = st.session_state.get("xt_result")
        if result and result.get("video_name") == uploaded_file.name:
            if st.session_state.get("xt_warning"):
                st.warning(st.session_state["xt_warning"])
            md(
                f'<div class="xt-note"><strong>{esc(result["video_name"])}</strong>'
                f'<span class="xt-note-meta">{format_time(result["total_duration"])} · '
                f'{MODE_LABELS.get(result["analysis_mode"], result["analysis_mode"])} · '
                f'{result["clip_duration"]}-second clips, {result["clip_overlap"]} s overlap</span></div>'
            )
            render_result(result, is_demo=False)
            st.download_button(
                "Download results (JSON)",
                data=json.dumps(result, indent=2),
                file_name=f"{Path(result['video_name']).stem}_segments.json",
                mime="application/json",
            )

md(
    '<footer class="xt-footer">A prototype of the detection layer behind JioTV+ Xtras, built by '
    f'<a href="{SITE_URL}" target="_blank" rel="noopener">Akhil S</a>.</footer>'
)
