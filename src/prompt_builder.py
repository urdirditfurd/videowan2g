"""Construit le prompt Wan2GP d'une boucle lofi, mascotte comprise.

Le texte de style, le mouvement lent et la consigne de boucle sont ajoutés
ici, pas dans le YAML, pour qu'un thème ne puisse pas les oublier.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.errors import ThemeError

# Description fixe : le seed change le décor, pas l'identité de la mascotte.
MASCOT = (
    "a single calm capybara mascot with rounded chestnut-brown fur, "
    "a soft cream muzzle, small round ears, rosy cheeks, half-lidded peaceful eyes, "
    "and a simple oversized cream sweater"
)

# Plan fixe, micro-mouvements, retour visuel vers la première image.
MOTION_CONSTRAINT = (
    "Camera: locked tripod, perfectly stable, no shake, no zoom, no whip pan, "
    "at most an extremely slow drift. "
    "Motion: slow gentle natural motion, seamless looping animation, "
    "the last moment returns to the first frame"
)

STYLE = (
    "Visual style: Studio Ghibli background, Hayao Miyazaki hand-painted animation, "
    "lofi aesthetic, soft pastel colors, cozy peaceful relaxing atmosphere, "
    "delicate painted backgrounds"
)

NEGATIVE_PROMPT = (
    "camera shake, handheld camera, dutch angle, fast zoom, whip pan, crash zoom, "
    "sudden movement, running, jumping, energetic dancing, flicker, strobe, "
    "harsh flash, oversaturated neon, horror, gore, text, subtitles, watermark, "
    "logo, signature, extra limbs, deformed paws, distorted face, multiple capybaras, "
    "lowres, jpeg artifacts, uncanny plastic, photorealistic, 3d render"
)


@dataclass(frozen=True)
class ThemeMotion:
    speed: str
    ambience: str


@dataclass(frozen=True)
class Theme:
    id: str
    title: str
    base_prompt: str
    background_motion: str
    mascot_action: str
    lighting: str
    palette: str
    motion: ThemeMotion
    variations: tuple[str, ...]


@dataclass(frozen=True)
class BuiltPrompt:
    theme_id: str
    title: str
    seed: int
    variation: str
    prompt: str
    negative_prompt: str


def load_catalog(path: Path) -> dict[str, Theme]:
    """Charge themes.yaml. La clé de chaque entrée est l'identifiant stable."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ThemeError(f"Impossible de lire {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ThemeError(f"YAML invalide dans {path}: {exc}") from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("themes"), dict):
        raise ThemeError(f"{path} doit contenir un dictionnaire 'themes'.")

    catalog: dict[str, Theme] = {}
    for theme_id, body in raw["themes"].items():
        if not isinstance(theme_id, str) or not theme_id.replace("_", "").isalnum():
            raise ThemeError(f"Identifiant de thème invalide: {theme_id!r}.")
        if not isinstance(body, dict):
            raise ThemeError(f"Thème {theme_id}: contenu attendu sous forme de dictionnaire.")
        catalog[theme_id] = _parse_theme(theme_id, body)
    if not catalog:
        raise ThemeError(f"Aucun thème dans {path}.")
    return catalog


def build_prompt(theme: Theme, seed: int) -> BuiltPrompt:
    """Assemble un prompt reproductible : le même seed redonne la même variation."""
    if seed < 0:
        raise ThemeError("Le seed doit être un entier positif ou nul.")
    variation = theme.variations[seed % len(theme.variations)]
    prompt = " ".join(
        (
            theme.base_prompt,
            f"In the foreground, {MASCOT}, {theme.mascot_action}.",
            f"The background moves on its own: {theme.background_motion}.",
            f"Mascot micro-motion: {theme.motion.ambience}.",
            f"Small unique detail: {variation}.",
            f"{STYLE}, palette of {theme.palette}, {theme.lighting}.",
            MOTION_CONSTRAINT + ".",
        )
    )
    return BuiltPrompt(
        theme_id=theme.id,
        title=theme.title,
        seed=seed,
        variation=variation,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
    )


def _parse_theme(theme_id: str, body: dict[str, object]) -> Theme:
    motion_raw = body.get("motion")
    if not isinstance(motion_raw, dict):
        raise ThemeError(f"Thème {theme_id}: bloc motion manquant.")
    speed = _text(motion_raw.get("speed"), "motion.speed", theme_id)
    # Le catalogue entier reste sur un mouvement lent, même si un thème est édité vite.
    if speed != "slow":
        raise ThemeError(
            f"Thème {theme_id}: motion.speed vaut {speed!r}. La valeur attendue est 'slow'."
        )
    variations_raw = body.get("variations")
    if not isinstance(variations_raw, list) or len(variations_raw) < 3:
        raise ThemeError(f"Thème {theme_id}: au moins trois variations sont requises.")
    variations = tuple(
        _text(item, f"variations[{index}]", theme_id)
        for index, item in enumerate(variations_raw)
    )
    return Theme(
        id=theme_id,
        title=_text(body.get("title"), "title", theme_id),
        base_prompt=_text(body.get("base_prompt"), "base_prompt", theme_id),
        background_motion=_text(body.get("background_motion"), "background_motion", theme_id),
        mascot_action=_text(body.get("mascot_action"), "mascot_action", theme_id),
        lighting=_text(body.get("lighting"), "lighting", theme_id),
        palette=_text(body.get("palette"), "palette", theme_id),
        motion=ThemeMotion(
            speed=speed,
            ambience=_text(motion_raw.get("ambience"), "motion.ambience", theme_id),
        ),
        variations=variations,
    )


def _text(value: object, field: str, theme_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ThemeError(f"Thème {theme_id}: champ {field} manquant.")
    return " ".join(value.split())
