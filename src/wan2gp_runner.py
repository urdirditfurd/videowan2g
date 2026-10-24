"""Lance Wan2GP en headless, avec un seed explicite par vidéo.

Wan2GP (octobre 2026) se pilote de deux façons :
- ``python wgp.py --process reglages.json --output-dir ...`` libère la VRAM
  quand le processus se termine ;
- ``shared.api.init(...).submit_task(reglages)`` garde le modèle chargé.

Le modèle par défaut est ``t2v_1.3B`` (environ 6 Go, profil mémoire 4).
Le handler Wan annonce 16 i/s, un minimum de 5 images et un pas de 4.
81 images (4*20+1) durent donc 5,06 s et restent dans une seule fenêtre,
ce qui évite d'allonger la VRAM avec une sliding window absente du 1.3B.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Never, TextIO

from src.config_loader import Aspect, Backend, WanSettings
from src.errors import Wan2GPError
from src.prompt_builder import BuiltPrompt

logger = logging.getLogger(__name__)

# Grille lue dans models/wan/wan_handler.py pour la famille non-VACE.
FRAME_MINIMUM = 5
FRAME_STEP = 4
# Au-delà, le 1.3B sort de la longueur sur laquelle il est à l'aise, et la VRAM monte.
COMFORT_FRAMES = 81
VAE_MULTIPLE = 16

STARTUP_FLAGS: dict[str, str] = {
    "t2v_1.3B": "--t2v-1-3B",
    "fun_inp_1.3B": "--i2v-1-3B",
    "t2v": "--t2v-14B",
    "i2v": "--i2v-14B",
}

HEAVY_MODELS = frozenset({"t2v", "i2v", "t2v_2_2", "i2v_2_2"})
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov"}


@dataclass(frozen=True)
class GenerationRequest:
    prompt: BuiltPrompt
    settings: WanSettings
    output_dir: Path
    aspect: Aspect
    frames: int | None = None
    resolution: str | None = None
    image_path: Path | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class GenerationResult:
    stem: str
    seed: int
    settings_path: Path
    settings: dict[str, object]
    command: tuple[str, ...]
    raw_video: Path | None
    dry_run: bool


def snap_frame_count(requested: int, minimum: int = FRAME_MINIMUM, step: int = FRAME_STEP) -> int:
    """Ramène la durée sur la grille Wan, jamais en dessous de la demande."""
    if requested < minimum:
        return minimum
    remainder = (requested - minimum) % step
    if remainder == 0:
        return requested
    return requested + (step - remainder)


def parse_resolution(value: str) -> str:
    """Vérifie WIDTHxHEIGHT et l'alignement VAE (16 px sur le Wan 1.3B)."""
    width_text, separator, height_text = value.partition("x")
    if separator != "x" or not width_text.isdigit() or not height_text.isdigit():
        raise Wan2GPError(f"Résolution illisible: {value!r}. Forme attendue: 480x832.")
    width, height = int(width_text), int(height_text)
    if width % VAE_MULTIPLE or height % VAE_MULTIPLE or width < 256 or height < 256:
        raise Wan2GPError(
            f"Résolution {value} hors grille. Largeur et hauteur multiples de {VAE_MULTIPLE}, au moins 256."
        )
    return f"{width}x{height}"


def generate(request: GenerationRequest) -> GenerationResult:
    wan = request.settings
    requested_frames = request.frames if request.frames is not None else wan.frames
    if requested_frames < 1:
        raise Wan2GPError("Le nombre d'images doit être au moins 1.")
    frames = snap_frame_count(requested_frames)
    if frames > COMFORT_FRAMES and "1.3B" in _model_type(request):
        logger.warning(
            "Durée de %s images : le 1.3B est à l'aise jusqu'à %s images à 480p. "
            "La VRAM et le temps montent au-delà.",
            frames,
            COMFORT_FRAMES,
        )
    resolution = parse_resolution(request.resolution or wan.resolutions[request.aspect])
    model_type = _model_type(request)
    if model_type in HEAVY_MODELS:
        logger.warning(
            "Modèle %s : prévoir au moins 12 Go de VRAM, ou revenir à t2v_1.3B.",
            model_type,
        )

    if request.image_path is not None and not request.image_path.is_file():
        raise Wan2GPError(f"Image de départ introuvable: {request.image_path}.")

    stem = f"capybara_{request.prompt.theme_id}_{request.prompt.seed}"
    job_dir = request.output_dir / "jobs"
    raw_dir = request.output_dir / "raw" / stem
    job_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    payload = _settings_payload(
        request=request,
        model_type=model_type,
        frames=frames,
        resolution=resolution,
        stem=stem,
    )
    settings_path = job_dir / f"{stem}.settings.json"
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    command = build_command(
        wan,
        settings_path,
        raw_dir,
        model_type,
        seed=request.prompt.seed,
        frames=frames,
    )
    if request.dry_run:
        logger.info("Simulation : %s", " ".join(command))
        return GenerationResult(
            stem=stem,
            seed=request.prompt.seed,
            settings_path=settings_path,
            settings=payload,
            command=command,
            raw_video=None,
            dry_run=True,
        )

    root = _require_root(wan)
    wgp = root / "wgp.py"
    if not wgp.is_file():
        raise Wan2GPError(f"wgp.py introuvable dans {root}.")

    before = _video_index(raw_dir)
    match wan.backend:
        case "subprocess":
            _run_subprocess(command, root, request.output_dir / "logs" / f"{stem}.log", wan.timeout_seconds)
        case "api":
            _run_api(root, payload, raw_dir, _cli_args(wan, model_type))
        case unreachable:
            _assert_never(unreachable)

    raw_video = _newest_video(raw_dir, before)
    if raw_video is None:
        raise Wan2GPError(
            f"Wan2GP s'est terminé sans vidéo dans {raw_dir}. "
            f"Journal : {request.output_dir / 'logs' / f'{stem}.log'}."
        )
    return GenerationResult(
        stem=stem,
        seed=request.prompt.seed,
        settings_path=settings_path,
        settings=payload,
        command=command,
        raw_video=raw_video,
        dry_run=False,
    )


def build_command(
    wan: WanSettings,
    settings_path: Path,
    raw_dir: Path,
    model_type: str,
    seed: int,
    frames: int,
) -> tuple[str, ...]:
    """Commande headless. Le profil et l'attention sont des options de processus, pas du JSON."""
    python = wan.python or sys.executable
    command = [
        python,
        "wgp.py",
        "--process",
        str(settings_path),
        "--output-dir",
        str(raw_dir),
        "--profile",
        str(wan.profile),
        "--attention",
        wan.attention,
        "--frames",
        str(frames),
        "--steps",
        str(wan.steps),
        "--seed",
        str(seed),
        "--verbose",
        "1",
    ]
    startup = STARTUP_FLAGS.get(model_type)
    if startup:
        command.append(startup)
    if wan.fp16:
        command.append("--fp16")
    return tuple(command)


def _settings_payload(
    request: GenerationRequest,
    model_type: str,
    frames: int,
    resolution: str,
    stem: str,
) -> dict[str, object]:
    prompt = request.prompt
    payload: dict[str, object] = {
        "model_type": model_type,
        "prompt": prompt.prompt,
        "negative_prompt": prompt.negative_prompt,
        "image_mode": 0,
        "resolution": resolution,
        "video_length": frames,
        "force_fps": str(request.settings.fps),
        "num_inference_steps": request.settings.steps,
        "guidance_scale": request.settings.guidance_scale,
        "seed": prompt.seed,
        "repeat_generation": 1,
        "output_filename": stem,
        "override_profile": request.settings.profile,
        "prompt_enhancer": "",
    }
    if request.image_path is not None:
        # "S" : image de début, convention image_prompt_type de la famille i2v.
        payload["image_start"] = str(request.image_path.resolve())
        payload["image_prompt_type"] = "S"
    payload.update(request.settings.extra_settings)
    return payload


def _run_subprocess(command: tuple[str, ...], root: Path, log_path: Path, timeout: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Wan2GP : %s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            process.kill()
            raise Wan2GPError("Sortie Wan2GP indisponible.")
        relay = threading.Thread(target=_relay_output, args=(process.stdout, log), daemon=True)
        relay.start()
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            relay.join(timeout=5)
            raise Wan2GPError(
                f"Wan2GP a dépassé {timeout} s. Réduire les images ou les steps, "
                "ou passer au modèle 1.3B avec le profil 4."
            ) from None
        relay.join(timeout=5)
    if code != 0:
        tail = _tail(log_path)
        raise Wan2GPError(f"Wan2GP s'est arrêté avec le code {code}.\n{tail}")


def _run_api(
    root: Path,
    payload: dict[str, object],
    raw_dir: Path,
    cli_args: list[str],
) -> None:
    """Charge shared.api depuis le clone. Le module n'est pas un paquet pip."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        api = importlib.import_module("shared.api")
    except ImportError as exc:
        raise Wan2GPError(
            "L'API Wan2GP est introuvable. Lancer main.py avec le Python du conda wan2gp, "
            "ou utiliser backend: subprocess."
        ) from exc
    init = getattr(api, "init", None)
    if init is None:
        raise Wan2GPError("shared.api.init est absent de cette version de Wan2GP.")
    session = init(root=root, output_dir=raw_dir, cli_args=cli_args, console_output=True)
    job = session.submit_task(payload)
    result = job.result()
    if not getattr(result, "success", False):
        errors = getattr(result, "errors", ())
        detail = "; ".join(str(getattr(item, "message", item)) for item in errors)
        raise Wan2GPError(detail or "La génération API Wan2GP a échoué.")


def _cli_args(wan: WanSettings, model_type: str) -> list[str]:
    args = ["--profile", str(wan.profile), "--attention", wan.attention]
    startup = STARTUP_FLAGS.get(model_type)
    if startup:
        args.append(startup)
    if wan.fp16:
        args.append("--fp16")
    return args


def _relay_output(stream: TextIO, log: TextIO) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        sys.stdout.write(line)


def _video_index(directory: Path) -> dict[Path, int]:
    return {
        path: path.stat().st_mtime_ns
        for path in directory.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file()
    }


def _newest_video(directory: Path, before: dict[Path, int]) -> Path | None:
    found = [
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
        and path.is_file()
        and before.get(path) != path.stat().st_mtime_ns
    ]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime_ns)


def _model_type(request: GenerationRequest) -> str:
    if request.image_path is not None:
        return request.settings.i2v_model_type
    return request.settings.model_type


def _require_root(wan: WanSettings) -> Path:
    if wan.root is None:
        raise Wan2GPError(
            "WAN2GP_ROOT est vide. Renseigner wan2gp.root ou la variable WAN2GP_ROOT."
        )
    return wan.root


def _tail(path: Path, lines: int = 40) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _assert_never(value: Never) -> Never:
    raise Wan2GPError(f"Backend non géré: {value}.")
