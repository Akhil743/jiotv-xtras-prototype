"""
JioTV+ Xtras — AI Video Content Classifier (Prototype)

A simplified proof-of-concept of the JioTV+ Xtras contextual overlay feature.
Detects song sequences, action scenes, dialogue, credits, and transitions in
video content using multi-modal AI — the core detection pipeline that powers
an Amazon X-Ray-style viewing experience.

Pipeline:
  1. VLM-first architecture for multi-modal temporal segmentation
  2. Audio analysis pipeline (librosa) for music/action/dialogue detection
  3. Confidence-based routing (auto-tag / QA review / human annotator)
  4. Color-coded timeline with segment navigation
"""

import streamlit as st
import tempfile
import os
import json
import time
from pathlib import Path

st.set_page_config(
    page_title="JioTV+ Xtras — AI Content Classifier",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

CATEGORY_COLORS = {
    "dialogue": "#3B82F6",
    "musical": "#A855F7",
    "action": "#EF4444",
    "transition": "#F59E0B",
    "credits": "#6B7280",
    "silence": "#374151",
    "unknown": "#9CA3AF",
}

CATEGORY_ICONS = {
    "dialogue": "💬",
    "musical": "🎵",
    "action": "⚡",
    "transition": "🎞️",
    "credits": "📋",
    "silence": "🔇",
    "unknown": "❓",
}

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.7; font-size: 0.9rem; }

    .timeline-container {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 16px;
        margin: 12px 0;
    }
    .timeline-bar {
        display: flex;
        height: 40px;
        border-radius: 6px;
        overflow: hidden;
        cursor: pointer;
    }
    .timeline-segment {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        color: white;
        font-weight: 600;
        transition: opacity 0.2s;
        position: relative;
    }
    .timeline-segment:hover { opacity: 0.8; }
    .timeline-legend {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: #ccc;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    .segment-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 6px 0;
        border-left: 4px solid;
        transition: transform 0.1s;
    }
    .segment-card:hover { transform: translateX(4px); }
    .segment-card .cat-label {
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .segment-card .time-label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 2px;
    }
    .segment-card .desc {
        font-size: 12px;
        color: #cbd5e1;
        margin-top: 6px;
    }
    .segment-card .confidence-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 8px;
        background: #334155;
    }
    .segment-card .confidence-fill {
        height: 100%;
        border-radius: 2px;
    }

    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #e2e8f0;
    }
    .metric-card .metric-label {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .confidence-tier {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    .tier-auto { background: #065f46; color: #6ee7b7; }
    .tier-qa { background: #713f12; color: #fcd34d; }
    .tier-human { background: #7f1d1d; color: #fca5a5; }

    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .mode-audio { background: #1e3a5f; color: #60a5fa; }
    .mode-vlm { background: #3b0764; color: #c084fc; }
    .mode-hybrid { background: #14532d; color: #86efac; }

    div[data-testid="stSidebar"] { background: #0f172a; }
    .stApp { background: #0f172a; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🎬 JioTV+ Xtras — AI Content Classifier</h1>
    <p>Contextual AI overlay prototype • Detect songs, action, dialogue & credits in video content</p>
</div>
""", unsafe_allow_html=True)


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def get_confidence_tier(conf: float) -> tuple[str, str]:
    if conf >= 0.9:
        return "AUTO-TAG", "tier-auto"
    elif conf >= 0.7:
        return "QA REVIEW", "tier-qa"
    return "HUMAN ANNOTATOR", "tier-human"


def render_timeline(segments: list[dict], total_duration: float):
    if not segments or total_duration <= 0:
        return

    html_parts = []
    for seg in segments:
        width_pct = max(1, (seg["end_time"] - seg["start_time"]) / total_duration * 100)
        color = CATEGORY_COLORS.get(seg["category"], "#666")
        icon = CATEGORY_ICONS.get(seg["category"], "")
        label = f'{icon} {seg["category"][:3].upper()}' if width_pct > 5 else icon
        html_parts.append(
            f'<div class="timeline-segment" style="width:{width_pct:.1f}%;background:{color};" '
            f'title="{seg["category"].title()}: {format_time(seg["start_time"])} - {format_time(seg["end_time"])}">'
            f'{label}</div>'
        )

    legend_html = ""
    seen = set()
    for seg in segments:
        cat = seg["category"]
        if cat not in seen:
            seen.add(cat)
            color = CATEGORY_COLORS.get(cat, "#666")
            icon = CATEGORY_ICONS.get(cat, "")
            legend_html += f'<div class="legend-item"><div class="legend-dot" style="background:{color};"></div>{icon} {cat.title()}</div>'

    st.markdown(f"""
    <div class="timeline-container">
        <div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:11px;margin-bottom:6px;">
            <span>0:00</span>
            <span>Color-coded AI-detected segments</span>
            <span>{format_time(total_duration)}</span>
        </div>
        <div class="timeline-bar">{''.join(html_parts)}</div>
        <div class="timeline-legend">{legend_html}</div>
    </div>
    """, unsafe_allow_html=True)


def render_segment_cards(segments: list[dict]):
    for seg in segments:
        color = CATEGORY_COLORS.get(seg["category"], "#666")
        icon = CATEGORY_ICONS.get(seg["category"], "")
        tier_label, tier_class = get_confidence_tier(seg["confidence"])
        conf_pct = seg["confidence"] * 100

        st.markdown(f"""
        <div class="segment-card" style="border-left-color:{color};">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div class="cat-label" style="color:{color};">{icon} {seg["category"]}</div>
                    <div class="time-label">{format_time(seg["start_time"])} → {format_time(seg["end_time"])} ({seg.get("duration", seg["end_time"]-seg["start_time"]):.0f}s)</div>
                </div>
                <div style="text-align:right;">
                    <div class="confidence-tier {tier_class}">{tier_label}</div>
                    <div style="font-size:18px;font-weight:700;color:{color};margin-top:2px;">{conf_pct:.0f}%</div>
                </div>
            </div>
            <div class="desc">{seg.get("description", "")}</div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:{conf_pct}%;background:{color};"></div></div>
        </div>
        """, unsafe_allow_html=True)


def render_metrics(segments: list[dict], total_duration: float):
    if not segments:
        return

    category_counts = {}
    category_durations = {}
    confidences = []
    for seg in segments:
        cat = seg["category"]
        dur = seg["end_time"] - seg["start_time"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        category_durations[cat] = category_durations.get(cat, 0) + dur
        confidences.append(seg["confidence"])

    auto_pct = sum(1 for c in confidences if c >= 0.9) / len(confidences) * 100 if confidences else 0
    avg_conf = sum(confidences) / len(confidences) * 100 if confidences else 0

    cols = st.columns(4)
    metrics = [
        (f"{len(segments)}", "Segments Detected"),
        (f"{avg_conf:.0f}%", "Avg Confidence"),
        (f"{auto_pct:.0f}%", "Auto-Tag Rate"),
        (f"{len(category_counts)}", "Categories Found"),
    ]
    for col, (value, label) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Category Breakdown")
    for cat, dur in sorted(category_durations.items(), key=lambda x: -x[1]):
        pct = dur / total_duration * 100
        color = CATEGORY_COLORS.get(cat, "#666")
        icon = CATEGORY_ICONS.get(cat, "")
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
            <div style="width:100px;font-size:13px;color:{color};font-weight:600;">{icon} {cat.title()}</div>
            <div style="flex:1;background:#1e293b;border-radius:4px;height:20px;">
                <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;display:flex;align-items:center;padding-left:8px;">
                    <span style="font-size:10px;color:white;font-weight:600;">{format_time(dur)} ({pct:.0f}%)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def load_precached_results() -> list[dict] | None:
    cache_path = Path(__file__).parent / "demo_results.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return None


# --- Sidebar ---
with st.sidebar:
    st.markdown("### Configuration")

    analysis_mode = st.radio(
        "Analysis Mode",
        ["Audio Analysis (librosa)", "VLM (Ollama)", "Hybrid (Audio + VLM)"],
        index=0,
        help="Audio mode works offline. VLM requires Ollama running with a vision model."
    )

    mode_key = {"Audio Analysis (librosa)": "audio", "VLM (Ollama)": "vlm", "Hybrid (Audio + VLM)": "hybrid"}[analysis_mode]
    mode_class = {"audio": "mode-audio", "vlm": "mode-vlm", "hybrid": "mode-hybrid"}[mode_key]
    st.markdown(f'<div class="mode-badge {mode_class}">{mode_key.upper()} MODE</div>', unsafe_allow_html=True)

    if mode_key in ("vlm", "hybrid"):
        from vlm_classifier import check_ollama_available
        available, vision_models = check_ollama_available()
        if available and vision_models:
            vlm_model = st.selectbox("Vision Model", vision_models)
            st.success(f"Ollama connected — {len(vision_models)} vision model(s)")
        elif available:
            st.warning("Ollama running but no vision models found. Pull one: `ollama pull moondream`")
            vlm_model = None
        else:
            st.error("Ollama not running. Start it: `ollama serve`")
            vlm_model = None
    else:
        vlm_model = None

    st.markdown("---")
    st.markdown("### Clip Parameters")
    clip_duration = st.slider("Clip Duration (s)", 5, 30, 10)
    clip_overlap = st.slider("Overlap (s)", 0, 5, 2)

    st.markdown("---")
    st.markdown("### Confidence Tiers")
    st.markdown("""
    <div style="font-size:12px;line-height:1.8;">
        <span class="confidence-tier tier-auto">≥90% AUTO-TAG</span><br>
        <span class="confidence-tier tier-qa">70-90% QA REVIEW</span><br>
        <span class="confidence-tier tier-human">&lt;70% HUMAN ANNOTATOR</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Architecture")
    st.markdown("""
    <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
    <strong>Pipeline:</strong><br>
    1. Video → 10s clips (2s overlap)<br>
    2. Audio → librosa features<br>
    3. Frames → VLM classification<br>
    4. Fusion + post-processing<br>
    5. Confidence-based routing<br><br>
    <strong>JioTV+ Xtras Concept</strong><br>
    Amazon X-Ray-style contextual overlay
    </div>
    """, unsafe_allow_html=True)


# --- Main Content ---
uploaded_file = st.file_uploader(
    "Upload a video clip (.mp4, .mov, .avi, .mkv)",
    type=["mp4", "mov", "avi", "mkv"],
    help="Upload a short clip (1-5 min recommended) from any Indian film to analyze"
)

use_demo = st.checkbox("Use pre-cached demo results (for reliable demo)", value=False)

if use_demo:
    demo_data = load_precached_results()
    if demo_data:
        segments = demo_data["segments"]
        total_duration = demo_data["total_duration"]
        st.info(f"Loaded pre-cached results: {demo_data.get('video_name', 'Demo clip')} ({format_time(total_duration)})")

        render_timeline(segments, total_duration)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### Detected Segments")
            render_segment_cards(segments)
        with col2:
            st.markdown("### Analytics")
            render_metrics(segments, total_duration)
    else:
        st.warning("No pre-cached results found. Run analysis first or add `demo_results.json`.")

elif uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(uploaded_file)

    if st.button("🔬 Analyze Video", type="primary", use_container_width=True):
        from video_processor import get_video_duration, extract_audio, segment_video_clips
        from audio_analyzer import analyze_full_audio

        progress = st.progress(0, text="Initializing pipeline...")

        try:
            total_duration = get_video_duration(video_path)
            st.info(f"Video duration: {format_time(total_duration)}")
        except Exception as e:
            st.error(f"Could not read video: {e}")
            st.stop()

        progress.progress(10, text="Extracting audio...")

        audio_path = video_path.replace(".mp4", "_audio.wav")
        try:
            from video_processor import extract_audio
            extract_audio(video_path, audio_path)
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            st.stop()

        progress.progress(30, text="Analyzing audio features (librosa)...")

        skip_merge = mode_key in ("vlm", "hybrid")
        try:
            audio_segments = analyze_full_audio(
                audio_path,
                segment_duration=float(clip_duration),
                overlap=float(clip_overlap),
                merge=not skip_merge
            )
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            audio_segments = []

        segments = audio_segments

        if mode_key in ("vlm", "hybrid") and vlm_model:
            progress.progress(50, text="Running VLM classification on frames...")
            from video_processor import segment_video_clips, extract_clip_frames
            from vlm_classifier import classify_frames_batch

            clips = segment_video_clips(video_path, clip_duration, clip_overlap)

            for i, clip in enumerate(clips):
                pct = 50 + int(40 * i / len(clips))
                progress.progress(pct, text=f"VLM analyzing clip {i+1}/{len(clips)}...")

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

        progress.progress(95, text="Generating timeline...")

        cache_data = {
            "video_name": uploaded_file.name,
            "total_duration": total_duration,
            "analysis_mode": mode_key,
            "clip_duration": clip_duration,
            "clip_overlap": clip_overlap,
            "segments": segments,
        }
        cache_path = Path(__file__).parent / "demo_results.json"
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        progress.progress(100, text="Analysis complete!")
        time.sleep(0.5)

        if segments:
            render_timeline(segments, total_duration)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.markdown("### Detected Segments")
                render_segment_cards(segments)
            with col2:
                st.markdown("### Analytics")
                render_metrics(segments, total_duration)

            with st.expander("Raw JSON Output (API response format)"):
                st.json(cache_data)
        else:
            st.warning("No segments detected. Try a different clip or analysis mode.")

        try:
            os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
        except Exception:
            pass

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;background:#1e293b;border-radius:12px;margin:20px 0;">
        <div style="font-size:48px;margin-bottom:16px;">🎬</div>
        <h3 style="color:#e2e8f0;margin-bottom:8px;">Upload a video clip to begin</h3>
        <p style="color:#94a3b8;font-size:14px;max-width:500px;margin:0 auto;">
            A simplified prototype of the JioTV+ Xtras contextual overlay — detecting song
            sequences, action scenes, dialogue, and credits in video content using
            multi-modal AI (audio analysis + vision language models).
        </p>
        <div style="margin-top:24px;display:flex;gap:12px;justify-content:center;flex-wrap:wrap;">
            <div style="background:#1a1a2e;padding:8px 16px;border-radius:6px;font-size:12px;color:#94a3b8;">
                💡 Tip: Use a 1-5 minute clip for best results
            </div>
            <div style="background:#1a1a2e;padding:8px 16px;border-radius:6px;font-size:12px;color:#94a3b8;">
                🎵 Songs with choreography are easiest to detect
            </div>
            <div style="background:#1a1a2e;padding:8px 16px;border-radius:6px;font-size:12px;color:#94a3b8;">
                ⚡ Action scenes with high energy are well-classified
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How It Works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1️⃣", "Clip Segmentation", "Split video into 10s clips with 2s overlap for boundary handling"),
        ("2️⃣", "Audio Analysis", "librosa extracts tempo, harmonic ratio, spectral features, beat strength"),
        ("3️⃣", "VLM Classification", "Vision model analyzes sampled frames — choreography, combat, conversation"),
        ("4️⃣", "Fusion + Routing", "Merge signals, assign confidence tiers, route to auto-tag or human review"),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
        col.markdown(f"""
        <div style="background:#1e293b;border-radius:8px;padding:16px;text-align:center;height:180px;">
            <div style="font-size:28px;margin-bottom:8px;">{num}</div>
            <div style="font-weight:700;font-size:14px;color:#e2e8f0;margin-bottom:6px;">{title}</div>
            <div style="font-size:11px;color:#94a3b8;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
