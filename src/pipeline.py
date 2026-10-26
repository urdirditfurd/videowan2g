"""Enchaîne thème, prompt, Wan2GP et finition."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from src.config_loader import Aspect, AudioSettings, PipelineConfig, load_pipeline_config
from src.errors import PipelineError, ThemeError
from src.post_processor import PostProcessConfig, post_process
from src.prompt_builder import BuiltPrompt, build_prompt, load_catalog
from src.wan2gp_runner import GenerationRequest, generate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunOptions:
    theme_id: str
    seed: int | None = None
    dry_run: bool = False
    aspect: Aspect | None = None
    frames: int | None = None
    resolution: str | None = None
    image_path: Path | None = None
    audio_path: Path | None = None
    config_path: Path | None = None
    themes_path: Path | None = None


@dataclass(frozen=True)
class RunResult:
    stem: str
    seed: int
    prompt: BuiltPrompt
    manifest_path: Path
    video_path: Path | None
    dry_run: bool


def run(options: RunOptions, config: PipelineConfig | None = None) -> RunResult:
    loaded = config or load_pipeline_config(options.config_path)
    themes_path = options.themes_path or (Path(__file__).resolve().parents[1] / "config" / "themes.yaml")
    catalog = load_catalog(themes_path)
    theme = catalog.get(options.theme_id)
    if theme is None:
        known = ", ".join(sorted(catalog))
        raise ThemeError(f"Thème inconnu: {options.theme_id}. Thèmes: {known}.")

    # Seed explicite et borné : la même commande refait la même vidéo.
    seed = options.seed if options.seed is not None else random.SystemRandom().randrange(1, 2**31 - 1)
    built = build_prompt(theme, seed)
    aspect: Aspect = options.aspect or loaded.wan.aspect
    generated = generate(
        GenerationRequest(
            prompt=built,
            settings=loaded.wan,
            output_dir=loaded.output_dir,
            aspect=aspect,
            frames=options.frames,
            resolution=options.resolution,
            image_path=options.image_path,
            dry_run=options.dry_run,
        )
    )

    if not options.dry_run and (
        generated.raw_video is None or generated.raw_video.suffix.lower() != ".mp4"
    ):
        raise PipelineError(
            "Aucun MP4 après Wan2GP. La génération est un échec, même si le processus s'est terminé."
        )

    video_path: Path | None = None
    mae_before: float | None = None
    mae_after: float | None = None
    if generated.raw_video is not None:
        destination = loaded.output_dir / f"{generated.stem}.mp4"
        audio = _audio_source(options.audio_path, loaded.audio)
        finished = post_process(
            generated.raw_video,
            destination,
            PostProcessConfig(
                crossfade_seconds=loaded.post.crossfade_seconds,
                grain=loaded.post.grain,
                warm_grade=loaded.post.warm_grade,
                crf=loaded.post.crf,
                grain_seed=seed,
                audio_path=audio,
                audio_volume=loaded.audio.volume,
            ),
        )
        video_path = finished.output_path
        mae_before = finished.endpoint_mae_before
        mae_after = finished.endpoint_mae_after

    manifest_path = loaded.output_dir / f"{generated.stem}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "theme": built.theme_id,
        "title": built.title,
        "seed": seed,
        "variation": built.variation,
        "prompt": built.prompt,
        "negative_prompt": built.negative_prompt,
        "wan_settings": generated.settings,
        "command": list(generated.command),
        "video": str(video_path) if video_path else None,
        "endpoint_mae_before": mae_before,
        "endpoint_mae_after": mae_after,
        "dry_run": options.dry_run,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Manifeste : %s", manifest_path)
    if video_path is not None:
        logger.info("Vidéo : %s", video_path)
    return RunResult(
        stem=generated.stem,
        seed=seed,
        prompt=built,
        manifest_path=manifest_path,
        video_path=video_path,
        dry_run=options.dry_run,
    )


def list_themes(themes_path: Path | None = None) -> list[tuple[str, str]]:
    path = themes_path or (Path(__file__).resolve().parents[1] / "config" / "themes.yaml")
    catalog = load_catalog(path)
    return [(theme_id, theme.title) for theme_id, theme in catalog.items()]


def _audio_source(override: Path | None, audio: AudioSettings) -> Path | None:
    chosen = override if override is not None else (audio.path if audio.enabled else None)
    if chosen is None:
        return None
    if not chosen.is_file():
        raise PipelineError(
            f"Audio introuvable: {chosen}. Déposer un fichier libre de droits, ou laisser audio.enabled à false."
        )
    return chosen
