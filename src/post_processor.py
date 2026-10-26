"""Ferme la boucle et pose le rendu lofi, sur CPU, via FFmpeg.

La jointure est un fondu de la fin vers le début. La première image utile
et la dernière image utile sont alors la même image source (celle à t = fondu).
Enchaîner la lecture ne montre plus de coupure. Le grain et la chaleur
viennent ensuite, dans le même encodage.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.errors import PipelineError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostProcessConfig:
    crossfade_seconds: float = 0.35
    grain: int = 4
    warm_grade: bool = True
    crf: int = 18
    grain_seed: int = 1
    audio_path: Path | None = None
    audio_volume: float = 0.22


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    frames: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class PostProcessResult:
    output_path: Path
    source_duration: float
    output_duration: float
    loop_fade_seconds: float
    endpoint_mae_before: float
    endpoint_mae_after: float


def post_process(source: Path, destination: Path, config: PostProcessConfig) -> PostProcessResult:
    if not source.is_file():
        raise PipelineError(f"Vidéo source introuvable: {source}.")
    if source.resolve() == destination.resolve():
        raise PipelineError("La sortie de post-traitement doit être un autre fichier que la source.")
    ffmpeg = _tool("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)

    info = probe_video(source)
    before = endpoint_mae(source, info.frames)
    fade = _fade_seconds(info.duration, config.crossfade_seconds)
    video_target = destination
    temp_video: Path | None = None
    if config.audio_path is not None:
        temp_video = destination.with_suffix(".video.mp4")
        video_target = temp_video

    filtergraph = _filtergraph(info.duration, fade, config)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter_complex",
        filtergraph,
        "-map",
        "[vout]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(config.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_target),
    ]
    _run(command, "Le filtre de boucle a échoué.")
    if config.audio_path is not None and temp_video is not None:
        _mux_audio(temp_video, destination, config)
        temp_video.unlink(missing_ok=True)

    output_info = probe_video(destination)
    after = endpoint_mae(destination, output_info.frames)
    logger.info(
        "Boucle : écart première/dernière image %.2f → %.2f (fondu %.2f s).",
        before,
        after,
        fade,
    )
    return PostProcessResult(
        output_path=destination,
        source_duration=info.duration,
        output_duration=output_info.duration,
        loop_fade_seconds=fade,
        endpoint_mae_before=before,
        endpoint_mae_after=after,
    )


def probe_video(path: Path) -> VideoInfo:
    ffprobe = _tool("ffprobe")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise PipelineError(completed.stderr.strip() or f"ffprobe a échoué sur {path}.")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise PipelineError(f"Aucune piste vidéo dans {path}.")
    stream = streams[0]
    frames = int(stream.get("nb_read_frames") or 0)
    duration = float((payload.get("format") or {}).get("duration") or stream.get("duration") or 0)
    if frames < 2 or duration <= 0:
        raise PipelineError(f"Vidéo trop courte pour une boucle: {path}.")
    fps = _fps(str(stream.get("r_frame_rate") or "0/1"))
    return VideoInfo(
        duration=duration,
        frames=frames,
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
    )


def endpoint_mae(path: Path, frames: int) -> float:
    """Écart moyen absolu entre la première et la dernière image, sur 24×24."""
    first = _sample_frame(path, 0)
    last = _sample_frame(path, frames - 1)
    if len(first) != len(last) or not first:
        raise PipelineError(f"Impossible de comparer les images de {path}.")
    return sum(abs(left - right) for left, right in zip(first, last)) / len(first)


def _filtergraph(duration: float, fade: float, config: PostProcessConfig) -> str:
    # offset = durée du corps après le début du fondu.
    # Le corps commence à t=fade et le fondu ramène vers ce même instant :
    # la dernière image rejoint donc la première.
    offset = max(0.0, duration - (2 * fade))
    parts = [
        "[0:v]split[va][vb]",
        f"[va]trim=duration={fade:.6f},setpts=PTS-STARTPTS,format=yuv420p[head]",
        f"[vb]trim=start={fade:.6f},setpts=PTS-STARTPTS,format=yuv420p[main]",
        (
            f"[main][head]xfade=transition=fade:duration={fade:.6f}:offset={offset:.6f}"
        ),
    ]
    grade: list[str] = []
    if config.warm_grade:
        grade.extend(
            (
                "eq=saturation=0.88:contrast=0.96:brightness=0.02:gamma=1.04",
                "colorbalance=rs=0.05:gs=0.01:bs=-0.05:rm=0.03:bm=-0.03:rh=0.02:bh=-0.04:pl=1",
            )
        )
    if config.grain > 0:
        strength = max(0, min(100, config.grain))
        seed = config.grain_seed if config.grain_seed >= 0 else 1
        grade.append(f"noise=alls={strength}:allf=t+u:all_seed={seed}")
    tail = ",".join(grade)
    suffix = f",{tail}[vout]" if tail else "[vout]"
    return ";".join(parts[:-1] + [parts[-1] + suffix])


def _mux_audio(video: Path, destination: Path, config: PostProcessConfig) -> None:
    if config.audio_path is None or not config.audio_path.is_file():
        raise PipelineError(f"Piste audio introuvable: {config.audio_path}.")
    info = probe_video(video)
    fade = min(0.4, max(0.05, info.duration / 5))
    start = max(0.0, info.duration - fade)
    volume = min(1.0, max(0.0, config.audio_volume))
    audio_filter = (
        f"volume={volume:.3f},"
        f"afade=t=in:st=0:d={fade:.3f},"
        f"afade=t=out:st={start:.3f}:d={fade:.3f}"
    )
    command = [
        _tool("ffmpeg"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-stream_loop",
        "-1",
        "-i",
        str(config.audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-af",
        audio_filter,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    _run(command, "Le mixage audio a échoué.")


def _sample_frame(path: Path, index: int) -> bytes:
    completed = subprocess.run(
        [
            _tool("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{index}),scale=24:24",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or not completed.stdout:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PipelineError(message or f"Image {index} illisible dans {path}.")
    return completed.stdout


def _fade_seconds(duration: float, requested: float) -> float:
    # Le fondu doit tenir deux fois dans le plan, sinon la jointure mange le plan.
    upper = max(0.05, duration / 3)
    return min(max(0.05, requested), upper)


def _fps(value: str) -> float:
    numerator, _, denominator = value.partition("/")
    if not denominator:
        return float(numerator)
    return float(numerator) / float(denominator)


def _tool(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise PipelineError(f"{name} est absent du PATH. Installer FFmpeg.")
    return found


def _run(command: list[str], failure: str) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PipelineError(f"{failure} {detail}".strip())
