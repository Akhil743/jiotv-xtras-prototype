"""
Audio-based segment classification using librosa.

Music detection uses spectral features + beat strength + harmonic ratio.
Dialogue detection uses speech-like spectral centroids + zero-crossing rates.
Action detection uses high-energy transients + low speech indicators.
"""

import numpy as np
import librosa


def analyze_audio_segment(audio_path: str, sr: int = 22050) -> dict:
    """Analyze an audio file and return classification features."""
    y, sr = librosa.load(audio_path, sr=sr)

    if len(y) < sr * 0.5:
        return {
            "category": "silence",
            "confidence": 0.95,
            "features": {"rms_mean": 0.0},
            "description": "Segment too short or silent"
        }

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    if rms_mean < 0.005:
        return {
            "category": "silence",
            "confidence": 0.95,
            "features": {"rms_mean": rms_mean},
            "description": "Very low audio energy — likely silence or transition"
        }

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(spectral_centroid))
    centroid_std = float(np.std(spectral_centroid))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = float(np.mean(zcr))

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_ratio = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(y)) + 1e-10))

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)
    beat_strength = len(beat_frames) / (len(y) / sr) if len(y) > 0 else 0

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    bandwidth_mean = float(np.mean(spectral_bandwidth))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rolloff_mean = float(np.mean(spectral_rolloff))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

    features = {
        "rms_mean": round(rms_mean, 5),
        "rms_std": round(rms_std, 5),
        "centroid_mean": round(centroid_mean, 1),
        "centroid_std": round(centroid_std, 1),
        "zcr_mean": round(zcr_mean, 5),
        "harmonic_ratio": round(harmonic_ratio, 3),
        "tempo": round(tempo, 1),
        "beat_strength": round(beat_strength, 2),
        "bandwidth_mean": round(bandwidth_mean, 1),
        "rolloff_mean": round(rolloff_mean, 1),
        "mfcc_variance": round(mfcc_var, 3),
    }

    music_score = 0.0
    dialogue_score = 0.0
    action_score = 0.0

    # --- Music indicators ---
    if harmonic_ratio > 0.6:
        music_score += 0.3
    elif harmonic_ratio > 0.4:
        music_score += 0.15

    if 60 < tempo < 180 and beat_strength > 1.5:
        music_score += 0.3
    elif beat_strength > 1.0:
        music_score += 0.15

    if centroid_mean > 1500 and centroid_std < 1000:
        music_score += 0.2

    if bandwidth_mean > 1500:
        music_score += 0.1

    if mfcc_var > 50:
        music_score += 0.1

    # --- Dialogue indicators ---
    if 800 < centroid_mean < 3000 and centroid_std < 800:
        dialogue_score += 0.25

    if 0.02 < zcr_mean < 0.12:
        dialogue_score += 0.2

    if rms_std / (rms_mean + 1e-10) > 0.3:
        dialogue_score += 0.15

    if harmonic_ratio > 0.3 and harmonic_ratio < 0.65:
        dialogue_score += 0.15

    if mfcc_var < 80:
        dialogue_score += 0.1

    if bandwidth_mean < 2000:
        dialogue_score += 0.15

    # --- Action indicators ---
    if rms_mean > 0.08:
        action_score += 0.2
    elif rms_mean > 0.04:
        action_score += 0.1

    if rms_std > 0.03:
        action_score += 0.2

    if centroid_mean > 2500:
        action_score += 0.15

    if zcr_mean > 0.1:
        action_score += 0.15

    if harmonic_ratio < 0.35:
        action_score += 0.2

    if bandwidth_mean > 2500:
        action_score += 0.1

    total = music_score + dialogue_score + action_score + 0.01
    music_norm = music_score / total
    dialogue_norm = dialogue_score / total
    action_norm = action_score / total

    scores = {
        "musical": round(music_norm, 3),
        "dialogue": round(dialogue_norm, 3),
        "action": round(action_norm, 3),
    }

    category = max(scores, key=scores.get)
    confidence = scores[category]

    descriptions = {
        "musical": f"Music detected — tempo {tempo:.0f} BPM, harmonic ratio {harmonic_ratio:.2f}, beat strength {beat_strength:.1f}/s",
        "dialogue": f"Speech/dialogue — centroid {centroid_mean:.0f} Hz, moderate energy variation, harmonic ratio {harmonic_ratio:.2f}",
        "action": f"Action/high-energy — RMS {rms_mean:.4f}, high spectral centroid {centroid_mean:.0f} Hz, low harmonic ratio {harmonic_ratio:.2f}",
    }

    return {
        "category": category,
        "confidence": round(confidence, 3),
        "scores": scores,
        "features": features,
        "description": descriptions[category],
    }


def analyze_full_audio(audio_path: str, segment_duration: float = 10.0,
                       overlap: float = 2.0, sr: int = 22050,
                       merge: bool = True) -> list[dict]:
    """Analyze a full audio file in overlapping segments."""
    y, sr = librosa.load(audio_path, sr=sr)
    total_duration = len(y) / sr
    results = []
    start = 0.0
    idx = 0

    while start < total_duration:
        end = min(start + segment_duration, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment = y[start_sample:end_sample]

        if len(segment) < sr * 0.5:
            start += segment_duration - overlap
            continue

        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, segment, sr)
            result = analyze_audio_segment(tmp.name, sr=sr)
            import os
            os.unlink(tmp.name)

        result["clip_id"] = idx
        result["start_time"] = round(start, 2)
        result["end_time"] = round(end, 2)
        results.append(result)

        idx += 1
        start += segment_duration - overlap
        if end >= total_duration:
            break

    if merge:
        return merge_adjacent_segments(results)
    return results


def merge_adjacent_segments(segments: list[dict]) -> list[dict]:
    """Merge adjacent segments with the same category."""
    if not segments:
        return []

    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["category"] == merged[-1]["category"]:
            merged[-1]["end_time"] = seg["end_time"]
            merged[-1]["confidence"] = round(
                (merged[-1]["confidence"] + seg["confidence"]) / 2, 3
            )
        else:
            merged.append(seg.copy())

    for i, seg in enumerate(merged):
        seg["clip_id"] = i
        seg["duration"] = round(seg["end_time"] - seg["start_time"], 2)

    return merged
