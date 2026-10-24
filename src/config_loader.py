"""Charge pipeline.yaml et applique les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from src.errors import PipelineError

Backend = Literal["subprocess", "api"]
Aspect = Literal["9:16", "1:1"]

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WanSettings:
    root: Path | None
    python: str
    backend: Backend
    model_type: str
    i2v_model_type: str
    profile: int
    attention: str
    fp16: bool
    frames: int
    fps: int
    steps: int
    guidance_scale: float
    aspect: Aspect
    resolutions: dict[Aspect, str]
    timeout_seconds: int
    extra_settings: dict[str, object]


@dataclass(frozen=True)
class PostSettings:
    crossfade_seconds: float
    grain: int
    warm_grade: bool
    crf: int


@dataclass(frozen=True)
class AudioSettings:
    enabled: bool
    path: Path | None
    volume: float


@dataclass(frozen=True)
class PipelineConfig:
    wan: WanSettings
    output_dir: Path
    post: PostSettings
    audio: AudioSettings


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    config_path = path or (REPO_ROOT / "config" / "pipeline.yaml")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PipelineError(f"Impossible de lire {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PipelineError(f"YAML invalide dans {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineError(f"{config_path} est vide.")

    wan_raw = _section(raw, "wan2gp")
    post_raw = _section(raw, "post")
    audio_raw = _section(raw, "audio")
    output_raw = raw.get("output_dir", "output")
    if not isinstance(output_raw, str) or not output_raw.strip():
        raise PipelineError("output_dir doit être un chemin.")

    root = _optional_path(
        os.environ.get("WAN2GP_ROOT", "").strip() or _optional_str(wan_raw.get("root"))
    )
    python = os.environ.get("WAN2GP_PYTHON", "").strip() or _optional_str(wan_raw.get("python"))
    resolutions = _resolutions(wan_raw.get("resolutions"))
    return PipelineConfig(
        wan=WanSettings(
            root=root,
            python=python,
            backend=_backend(wan_raw.get("backend", "subprocess")),
            model_type=_required_str(wan_raw, "model_type"),
            i2v_model_type=_required_str(wan_raw, "i2v_model_type"),
            profile=_int(wan_raw.get("profile"), "profile"),
            attention=_required_str(wan_raw, "attention"),
            fp16=_bool(wan_raw.get("fp16"), "fp16"),
            frames=_int(wan_raw.get("frames"), "frames"),
            fps=_int(wan_raw.get("fps"), "fps"),
            steps=_int(wan_raw.get("steps"), "steps"),
            guidance_scale=_float(wan_raw.get("guidance_scale"), "guidance_scale"),
            aspect=_aspect(wan_raw.get("aspect", "9:16")),
            resolutions=resolutions,
            timeout_seconds=_int(wan_raw.get("timeout_seconds"), "timeout_seconds"),
            extra_settings=_extra(wan_raw.get("extra_settings")),
        ),
        output_dir=_resolve_output(output_raw),
        post=PostSettings(
            crossfade_seconds=_float(post_raw.get("crossfade_seconds"), "crossfade_seconds"),
            grain=_int(post_raw.get("grain"), "grain"),
            warm_grade=_bool(post_raw.get("warm_grade"), "warm_grade"),
            crf=_int(post_raw.get("crf"), "crf"),
        ),
        audio=AudioSettings(
            enabled=_bool(audio_raw.get("enabled"), "audio.enabled"),
            path=_optional_path(_optional_str(audio_raw.get("path"))),
            volume=_float(audio_raw.get("volume"), "audio.volume"),
        ),
    )


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise PipelineError(f"Section {name} manquante dans pipeline.yaml.")
    return value


def _resolutions(value: object) -> dict[Aspect, str]:
    if not isinstance(value, dict):
        raise PipelineError("wan2gp.resolutions doit être un dictionnaire.")
    portrait = value.get("9:16")
    square = value.get("1:1")
    if isinstance(portrait, int) or isinstance(square, int):
        raise PipelineError(
            "Les clés 9:16 et 1:1 doivent être entre guillemets dans le YAML."
        )
    if not isinstance(portrait, str) or not isinstance(square, str):
        raise PipelineError("Les résolutions 9:16 et 1:1 sont requises.")
    return {"9:16": portrait.strip(), "1:1": square.strip()}


def _backend(value: object) -> Backend:
    if value == "subprocess":
        return "subprocess"
    if value == "api":
        return "api"
    raise PipelineError("wan2gp.backend vaut subprocess ou api.")


def _aspect(value: object) -> Aspect:
    if value == "9:16":
        return "9:16"
    if value == "1:1":
        return "1:1"
    raise PipelineError("wan2gp.aspect vaut 9:16 ou 1:1.")


def _extra(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PipelineError("wan2gp.extra_settings doit être un dictionnaire.")
    return dict(value)


def _required_str(section: dict[str, object], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PipelineError(f"{key} doit être une chaîne non vide.")
    return value.strip()


def _optional_str(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PipelineError("Un chemin optionnel doit être une chaîne.")
    return value.strip()


def _optional_path(value: str) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _resolve_output(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(f"{label} doit être un entier.")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"{label} doit être un nombre.")
    return float(value)


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise PipelineError(f"{label} doit être un booléen.")
    return value
