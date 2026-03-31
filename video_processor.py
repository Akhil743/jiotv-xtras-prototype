"""
Video processing pipeline: extract frames and audio from video files using ffmpeg.
"""

import subprocess
import os
import json
import tempfile
from pathlib import Path


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def extract_audio(video_path: str, output_path: str) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
         "-ar", "22050", "-ac", "1", output_path],
        capture_output=True, text=True, check=True
    )
    return output_path


def extract_frames(video_path: str, output_dir: str, fps: float = 1.0) -> list[str]:
    """Extract frames at given FPS. Default 1 FPS for analysis."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
         "-q:v", "2", os.path.join(output_dir, "frame_%06d.jpg")],
        capture_output=True, text=True, check=True
    )
    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )
    return frames


def segment_video_clips(video_path: str, clip_duration: float = 10.0,
                        overlap: float = 2.0) -> list[dict]:
    """Generate clip metadata with overlap for boundary handling."""
    duration = get_video_duration(video_path)
    clips = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + clip_duration, duration)
        clips.append({
            "clip_id": idx,
            "start_time": round(start, 2),
            "end_time": round(end, 2),
            "duration": round(end - start, 2),
        })
        idx += 1
        start += clip_duration - overlap
        if end >= duration:
            break
    return clips


def extract_clip_audio(video_path: str, start: float, end: float,
                       output_path: str) -> str:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-to", str(end),
         "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", output_path],
        capture_output=True, text=True, check=True
    )
    return output_path


def extract_clip_frames(video_path: str, start: float, end: float,
                        output_dir: str, sample_every_n: int = 5) -> list[str]:
    """Extract frames from a clip, sampling every Nth frame (default every 5th = 5x reduction)."""
    os.makedirs(output_dir, exist_ok=True)
    fps_filter = f"fps=1"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ss", str(start), "-to", str(end),
         "-vf", fps_filter, "-q:v", "2",
         os.path.join(output_dir, "frame_%04d.jpg")],
        capture_output=True, text=True, check=True
    )
    frames = sorted(
        [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
    )
    sampled = frames[::sample_every_n] if sample_every_n > 1 else frames
    return sampled
