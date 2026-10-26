"""Lance Wan2GP en headless et exige un vrai MP4.

La commande retenue est celle du mode file d'attente :

    python wgp.py --process <file.zip> --output-dir <dossier> --verbose 2

Le ZIP contient ``queue.json``. Le modèle est toujours ``t2v_1.3B``.
``fun_inp_1.3B`` (I2V) est refusé : cette famille peut terminer en
écrivant un JPEG, et Wan2GP quitte alors avec un code succès.
Un MP4 absent, illisible, ou réduit à une image fixe est une erreur.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Never, TextIO

from src.config_loader import Aspect, WanSettings
from src.errors import Wan2GPError
from src.prompt_builder import BuiltPrompt

logger = logging.getLogger(__name__)

# Grille lue dans models/wan/wan_handler.py pour la famille non-VACE.
FRAME_MINIMUM = 5
FRAME_STEP = 4
# Au-delà, le 1.3B sort de la longueur sur laquelle il est à l'aise, et la VRAM monte.
COMFORT_FRAMES = 81
VAE_MULTIPLE = 16
T2V_MODEL = "t2v_1.3B"
T2V_FLAG = "--t2v-1-3B"
SETTINGS_VERSION = 2.81
STILL_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STILL_CODECS = {"mjpeg", "png", "jpeg", "bmp", "gif", "webp"}
# Un MP4 d'une seule image (ou un JPEG renommé) reste sous ce seuil.
MIN_VIDEO_SECONDS = 0.15
MIN_VIDEO_BYTES = 256

_ffmpeg_cache: tuple[str, str] | None = None


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
    queue_path: Path
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
    model_type = _model_type(request)
    if frames > COMFORT_FRAMES and "1.3B" in model_type:
        logger.warning(
            "Durée de %s images : le 1.3B est à l'aise jusqu'à %s images à 480p. "
            "La VRAM et le temps montent au-delà.",
            frames,
            COMFORT_FRAMES,
        )
    resolution = parse_resolution(request.resolution or wan.resolutions[request.aspect])

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
    queue_path = job_dir / f"{stem}.queue.zip"
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_queue_zip(queue_path, payload)
    command = build_command(
        wan,
        queue_path,
        raw_dir,
        seed=request.prompt.seed,
        frames=frames,
    )
    if request.dry_run:
        logger.info("Simulation : %s", " ".join(command))
        return GenerationResult(
            stem=stem,
            seed=request.prompt.seed,
            settings_path=settings_path,
            queue_path=queue_path,
            settings=payload,
            command=command,
            raw_video=None,
            dry_run=True,
        )

    root = _require_root(wan)
    wgp = root / "wgp.py"
    if not wgp.is_file():
        raise Wan2GPError(f"wgp.py introuvable dans {root}.")

    log_path = request.output_dir / "logs" / f"{stem}.log"
    env = prepare_ffmpeg_env()
    before_mp4 = _index(raw_dir, {".mp4"})
    before_stills = _index(raw_dir, STILL_SUFFIXES)
    match wan.backend:
        case "subprocess":
            _run_subprocess(command, root, log_path, wan.timeout_seconds, env)
        case "api":
            _run_api(root, payload, raw_dir, _cli_args(wan))
        case unreachable:
            _assert_never(unreachable)

    raw_video = require_produced_mp4(
        raw_dir,
        before_mp4=before_mp4,
        before_stills=before_stills,
        log_path=log_path,
        ffprobe=env["FFPROBE_BINARY"],
    )
    return GenerationResult(
        stem=stem,
        seed=request.prompt.seed,
        settings_path=settings_path,
        queue_path=queue_path,
        settings=payload,
        command=command,
        raw_video=raw_video,
        dry_run=False,
    )


def build_command(
    wan: WanSettings,
    queue_path: Path,
    raw_dir: Path,
    seed: int,
    frames: int,
) -> tuple[str, ...]:
    """File ZIP headless. Le profil et l'attention restent des options de processus."""
    if queue_path.suffix.lower() != ".zip":
        raise Wan2GPError(f"La file Wan2GP doit être un ZIP, pas {queue_path.name}.")
    python = wan.python or sys.executable
    command = [
        python,
        "wgp.py",
        "--process",
        str(queue_path),
        "--output-dir",
        str(raw_dir),
        "--verbose",
        "2",
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
        T2V_FLAG,
    ]
    if wan.fp16:
        command.append("--fp16")
    return tuple(command)


def write_queue_zip(path: Path, payload: dict[str, object], task_id: int = 1) -> None:
    """ZIP de file d'attente : ``queue.json`` est le manifeste lu par ``--process``."""
    manifest = [{"id": task_id, "params": payload, "plugin_data": {}}]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("queue.json", json.dumps(manifest, ensure_ascii=False, indent=4))


def prepare_ffmpeg_env() -> dict[str, str]:
    """Rend ffmpeg et ffprobe visibles pour Wan2GP (PATH, FFMPEG_BINARY, imageio)."""
    ffmpeg, ffprobe = ensure_ffmpeg()
    env = os.environ.copy()
    bin_dir = str(Path(ffmpeg).resolve().parent)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["FFMPEG_BINARY"] = ffmpeg
    env["FFPROBE_BINARY"] = ffprobe
    env["IMAGEIO_FFMPEG_EXE"] = ffmpeg
    os.environ["FFMPEG_BINARY"] = ffmpeg
    os.environ["FFPROBE_BINARY"] = ffprobe
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg
    return env


def ensure_ffmpeg() -> tuple[str, str]:
    """Échoue si FFmpeg ne peut pas produire un MP4 lisible par ffprobe."""
    global _ffmpeg_cache
    if _ffmpeg_cache is not None:
        return _ffmpeg_cache
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        missing = "ffmpeg" if ffmpeg is None else "ffprobe"
        raise Wan2GPError(
            f"{missing} est absent du PATH. Sans FFmpeg, Wan2GP ne peut pas muxer de MP4 "
            "et enregistre parfois un JPEG à la place. Installer FFmpeg, puis relancer."
        )
    with tempfile.TemporaryDirectory(prefix="wan2gp-ffmpeg-") as directory:
        sample = Path(directory) / "probe.mp4"
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=16x16:r=8:d=0.25",
                "-frames:v",
                "2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                str(sample),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        detail = (completed.stderr or completed.stdout).strip()
        if completed.returncode != 0 or not sample.is_file():
            raise Wan2GPError(
                "FFmpeg est installé mais n'encode pas de MP4 (libx264). "
                f"{detail}".strip()
            )
        try:
            _assert_real_mp4(sample, ffprobe)
        except Wan2GPError as exc:
            raise Wan2GPError(f"FFmpeg a écrit un fichier que ffprobe refuse. {exc}") from exc
    _ffmpeg_cache = (ffmpeg, ffprobe)
    return _ffmpeg_cache


def require_produced_mp4(
    directory: Path,
    before_mp4: dict[Path, int],
    before_stills: dict[Path, int],
    log_path: Path | None = None,
    ffprobe: str | None = None,
) -> Path:
    """Échoue si le dossier de sortie n'a pas gagné un MP4 vidéo."""
    probe = ffprobe or shutil.which("ffprobe")
    if probe is None:
        raise Wan2GPError("ffprobe est absent du PATH. Impossible de vérifier le MP4.")
    fresh_mp4 = _new_files(directory, {".mp4"}, before_mp4)
    fresh_stills = _new_files(directory, STILL_SUFFIXES, before_stills)
    errors: list[str] = []
    for path in sorted(fresh_mp4, key=lambda item: item.stat().st_mtime_ns, reverse=True):
        try:
            _assert_real_mp4(path, probe)
        except Wan2GPError as exc:
            errors.append(str(exc))
            continue
        return path
    _fail_without_mp4(directory, fresh_stills, errors, log_path)


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
        "base_model_type": model_type,
        "settings_version": SETTINGS_VERSION,
        "prompt": prompt.prompt,
        "negative_prompt": prompt.negative_prompt,
        "multi_prompts_gen_type": "FG",
        "image_mode": 0,
        "image_prompt_type": "",
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
    payload.update(request.settings.extra_settings)
    # Les réglages libres ne doivent pas ramener l'I2V ni le mode image.
    payload["model_type"] = model_type
    payload["base_model_type"] = model_type
    payload["settings_version"] = SETTINGS_VERSION
    payload["image_mode"] = 0
    payload["image_prompt_type"] = ""
    payload["multi_prompts_gen_type"] = "FG"
    payload["video_length"] = frames
    payload["resolution"] = resolution
    payload["seed"] = prompt.seed
    payload["output_filename"] = stem
    payload.pop("image_start", None)
    payload.pop("image_end", None)
    payload.pop("image_refs", None)
    return payload


def _run_subprocess(
    command: tuple[str, ...],
    root: Path,
    log_path: Path,
    timeout: int,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Wan2GP : %s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"FFmpeg: {env.get('FFMPEG_BINARY', '')}\n")
        log.write("Commande: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
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
                "ou rester sur t2v_1.3B avec le profil 4."
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


def _cli_args(wan: WanSettings) -> list[str]:
    args = [
        "--profile",
        str(wan.profile),
        "--attention",
        wan.attention,
        "--verbose",
        "2",
        T2V_FLAG,
    ]
    if wan.fp16:
        args.append("--fp16")
    return args


def _relay_output(stream: TextIO, log: TextIO) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        sys.stdout.write(line)


def _index(directory: Path, suffixes: set[str]) -> dict[Path, int]:
    return {
        path: path.stat().st_mtime_ns
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def _new_files(directory: Path, suffixes: set[str], before: dict[Path, int]) -> list[Path]:
    found: list[Path] = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if before.get(path) != path.stat().st_mtime_ns:
            found.append(path)
    return found


def _assert_real_mp4(path: Path, ffprobe: str) -> None:
    if path.suffix.lower() != ".mp4":
        raise Wan2GPError(f"{path.name} n'est pas un MP4.")
    if not path.is_file() or path.stat().st_size < MIN_VIDEO_BYTES:
        raise Wan2GPError(f"{path.name} est vide ou trop petit pour être une vidéo.")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,nb_frames,duration",
            "-show_entries",
            "format=duration,format_name",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise Wan2GPError(f"{path.name} n'est pas un MP4 lisible. {detail}".strip())
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Wan2GPError(f"ffprobe n'a pas renvoyé de JSON pour {path.name}.") from exc
    format_info = payload.get("format") if isinstance(payload, dict) else None
    format_name = ""
    if isinstance(format_info, dict):
        format_name = str(format_info.get("format_name") or "")
    if "mp4" not in format_name:
        raise Wan2GPError(f"{path.name} n'est pas un conteneur MP4 ({format_name or 'inconnu'}).")
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise Wan2GPError(f"{path.name} ne contient pas de piste vidéo.")
    stream = streams[0]
    codec = str(stream.get("codec_name") or "")
    if codec in STILL_CODECS:
        raise Wan2GPError(f"{path.name} est une image {codec} dans un MP4, pas une vidéo.")
    duration = _duration(format_info if isinstance(format_info, dict) else {}, stream)
    frames = _frame_count(stream)
    if frames < 2 and duration < MIN_VIDEO_SECONDS:
        raise Wan2GPError(
            f"{path.name} ne contient qu'une image fixe (durée {duration:.3f} s)."
        )


def _duration(format_info: dict[str, object], stream: dict[str, object]) -> float:
    for source in (stream, format_info):
        raw = source.get("duration")
        if raw in (None, "", "N/A"):
            continue
        try:
            return float(str(raw))
        except ValueError:
            continue
    return 0.0


def _frame_count(stream: dict[str, object]) -> int:
    raw = stream.get("nb_frames")
    if raw in (None, "", "N/A"):
        return 0
    try:
        return int(str(raw))
    except ValueError:
        return 0


def _fail_without_mp4(
    directory: Path,
    stills: list[Path],
    mp4_errors: list[str],
    log_path: Path | None,
) -> Never:
    parts = [f"Aucun MP4 produit dans {directory}."]
    if stills:
        names = ", ".join(path.name for path in stills[:5])
        parts.append(
            f"Image fixe à la place : {names}. "
            "Wan2GP écrit un JPEG quand la tâche est en mode image "
            "(Fun_InP_1.3B / I2V) ou quand FFmpeg ne peut pas muxer la vidéo. "
            "Ce pipeline force t2v_1.3B et image_mode 0."
        )
    if mp4_errors:
        parts.append("Fichiers MP4 refusés : " + " | ".join(mp4_errors))
    if log_path is not None and log_path.is_file():
        parts.append(f"Journal : {log_path}\n{_tail(log_path)}")
    raise Wan2GPError(" ".join(parts))


def _model_type(request: GenerationRequest) -> str:
    if request.image_path is not None:
        raise Wan2GPError(
            "Image de départ refusée. Le modèle Fun_InP_1.3B (I2V) peut enregistrer un JPEG "
            "à la place d'un MP4. Ce pipeline ne lance que le text-to-video t2v_1.3B, sans image."
        )
    if request.settings.model_type != T2V_MODEL:
        raise Wan2GPError(
            f"Modèle {request.settings.model_type} refusé. "
            "Seul t2v_1.3B est utilisé : Fun_InP_1.3B en I2V produit parfois un JPEG."
        )
    return T2V_MODEL


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
