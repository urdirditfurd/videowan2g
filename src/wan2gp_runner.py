"""Lance Wan2GP en headless et exige un MP4 en sortie.

La commande documentée pour vider une file sans interface est :

    python wgp.py --process <file.zip> --output-dir <dossier> --verbose 2

Le zip contient ``queue.json``. Le modèle est ``t2v_1.3B`` en ``image_mode`` 0.
``Fun_InP_1.3B`` (I2V) est refusé : cette famille peut écrire un JPEG et
quitter avec le code 0, sans vidéo.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Never, TextIO

from src.config_loader import Aspect, WanSettings
from src.errors import Wan2GPError
from src.prompt_builder import BuiltPrompt

logger = logging.getLogger(__name__)

# Grille lue dans le handler Wan pour la famille non-VACE.
FRAME_MINIMUM = 5
FRAME_STEP = 4
# Au-delà, le 1.3B sort de la longueur sur laquelle il est à l'aise, et la VRAM monte.
COMFORT_FRAMES = 81
VAE_MULTIPLE = 16
# Affectation de module dans wgp.py sur main (WanGP 18.83). Le clone installé est relu.
SETTINGS_VERSION = 2.81
# WanGP 2.81 a remplacé override_profile par le texte overrides (memory_profile=…).
_OVERRIDES_VERSION = 2.81
_SETTINGS_VERSION_RE = re.compile(
    r"(?m)^settings_version\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*(?:#.*)?$"
)
_OVERRIDE_FIELDS = (
    "memory_profile",
    "compilation",
    "attention_mode",
    "transformer_quantization",
    "text_encoder_quantization",
)
# Un plan d'une image, ou un JPEG renommé, reste sous ce seuil.
MIN_VIDEO_SECONDS = 0.15
MIN_VIDEO_FRAMES = 2

HEAVY_MODELS = frozenset({"t2v", "t2v_14B", "t2v_2_2"})
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
IMAGE_CODECS = frozenset(
    {
        "mjpeg",
        "jpeg",
        "png",
        "bmp",
        "gif",
        "webp",
        "ppm",
        "tiff",
        "ljpeg",
        "jpeg2000",
        "mjpeg_2000",
    }
)
# Conteneurs que WanGP peut écrire à la place d'un MP4 (wgp_config.json).
OTHER_VIDEO_SUFFIXES = {".mkv", ".mov", ".webm", ".avi", ".m4v"}
# video_output_codec de WanGP 18.84 -> encodeur ffmpeg, quand le conteneur est mp4.
_MP4_ENCODER_FOR_CODEC = {
    "libx264_8": "libx264",
    "libx264_10": "libx264",
    "libx264_lossless": "libx264",
    "libx265_28": "libx265",
    "libx265_18": "libx265",
    "libx265_8": "libx265",
}
_NON_MP4_CODECS = {
    "prores_422": "ProRes 422 s'écrit dans un MOV ou un MKV, pas dans un MP4.",
    "prores_4444": "ProRes 4444 s'écrit dans un MOV, pas dans un MP4.",
    "dnxhr_hq": "DNxHR HQ s'écrit dans un MOV ou un MKV, pas dans un MP4.",
}
ATTACHMENT_KEYS = (
    "image_start",
    "image_end",
    "image_refs",
    "image_guide",
    "image_mask",
    "video_guide",
    "video_source",
    "custom_guide",
)


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


@dataclass(frozen=True)
class FfmpegBinaries:
    ffmpeg: str
    ffprobe: str


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
    _reject_image_input(request)
    requested_frames = request.frames if request.frames is not None else wan.frames
    if requested_frames < 1:
        raise Wan2GPError("Le nombre d'images doit être au moins 1.")
    frames = snap_frame_count(requested_frames)
    model_type = _model_type(wan.model_type)
    if frames > COMFORT_FRAMES and "1.3B" in model_type:
        logger.warning(
            "Durée de %s images : le 1.3B est à l'aise jusqu'à %s images à 480p. "
            "La VRAM et le temps montent au-delà.",
            frames,
            COMFORT_FRAMES,
        )
    resolution = parse_resolution(request.resolution or wan.resolutions[request.aspect])
    if model_type in HEAVY_MODELS:
        logger.warning(
            "Modèle %s : prévoir au moins 12 Go de VRAM, ou revenir à t2v_1.3B.",
            model_type,
        )

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
        settings_version=resolve_settings_version(wan.root),
    )
    settings_path = job_dir / f"{stem}.settings.json"
    queue_path = job_dir / f"{stem}.queue.zip"
    settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_queue(queue_path, payload)
    command = build_command(wan, queue_path, raw_dir)
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

    binaries = require_ffmpeg(root)
    before = _file_index(raw_dir)
    log_path = request.output_dir / "logs" / f"{stem}.log"
    match wan.backend:
        case "subprocess":
            _run_subprocess(command, root, log_path, wan.timeout_seconds, binaries)
        case "api":
            logger.warning(
                "Le backend api est ignoré. Seul wgp.py --process écrit un fichier que l'on peut refuser s'il n'est pas un MP4."
            )
            _run_subprocess(command, root, log_path, wan.timeout_seconds, binaries)
        case unreachable:
            _assert_never(unreachable)

    raw_video = _require_mp4(raw_dir, before, binaries.ffprobe, log_path)
    return GenerationResult(
        stem=stem,
        seed=request.prompt.seed,
        settings_path=settings_path,
        settings=payload,
        command=command,
        raw_video=raw_video,
        dry_run=False,
    )


def build_command(wan: WanSettings, queue_path: Path, raw_dir: Path) -> tuple[str, ...]:
    """Headless WanGP : file zip, dossier de sortie, journal détaillé.

    Le profil et l'attention restent des options de processus. Le modèle,
    la durée et le seed sont dans la file, pas sur la ligne de commande.
    Les chemins sont absolus : le processus a pour répertoire le clone WanGP,
    et un chemin relatif y écrirait le MP4 hors du dossier de sortie.
    """
    python = wan.python or sys.executable
    command = [
        python,
        "wgp.py",
        "--process",
        str(queue_path.resolve()),
        "--output-dir",
        str(raw_dir.resolve()),
        "--verbose",
        "2",
        "--profile",
        str(wan.profile),
        "--attention",
        wan.attention,
    ]
    if wan.fp16:
        command.append("--fp16")
    return tuple(command)


def resolve_settings_version(root: Path | None) -> float:
    """Reprend ``settings_version`` du ``wgp.py`` installé, sinon la valeur de main.

    Sur WanGP 18.84 l'affectation est après les imports, au-delà des 8 premiers Ko.
    La première affectation de module est la bonne : les suivantes sont des paramètres.
    """
    if root is not None:
        source = root / "wgp.py"
        if source.is_file():
            match = _SETTINGS_VERSION_RE.search(source.read_text(encoding="utf-8", errors="replace"))
            if match:
                return float(match.group(1))
    return SETTINGS_VERSION


def require_ffmpeg(root: Path | None = None) -> FfmpegBinaries:
    """Vérifie le ffmpeg que Wan2GP va réellement lancer, avant le GPU.

    ``download_ffmpeg`` déplace d'abord ``<root>/ffmpeg`` vers ``ffmpeg_quarantine``,
    puis prend ``<root>/ffmpeg_bins``, puis le PATH, et écrase ``FFMPEG_BINARY``.
    ``ffprobe`` à la racine du clone n'est pas déplacé. Un binaire cassé dans
    ``ffmpeg_bins`` masque donc un ffmpeg système correct. Toute bibliothèque
    ``*.so`` laissée dans ``ffmpeg_bins`` est ajoutée à ``LD_LIBRARY_PATH``
    même si l'encodeur retenu est celui du PATH : ``libx264.so*`` comme
    ``libav*.so*`` peuvent faire échouer le mux, et le processus sort quand
    même avec le code 0. Un dossier, un tube ou un socket nommé ``ffmpeg``
    est aussi retenu (``Path.exists()``) et copié dans ``FFMPEG_BINARY``.
    ``wgp_config.json`` choisit le conteneur : autre chose que ``mp4`` sort
    un MKV ou un MOV, toujours avec le code 0, et le dossier suivi n'a pas
    de MP4.
    """
    _require_wan_binary_is_a_file(root)
    ffmpeg = _resolve_binary("ffmpeg", root)
    ffprobe = _resolve_binary("ffprobe", root)
    if ffmpeg is None or ffprobe is None:
        quarantined = ""
        if root is not None and (root / _binary_name("ffmpeg")).is_file():
            quarantined = (
                " Le ffmpeg à la racine du clone ne compte pas : Wan2GP le déplace "
                "dans ffmpeg_quarantine avant l'encodage."
            )
        raise Wan2GPError(
            "ffmpeg et ffprobe sont introuvables. Wan2GP les cherche dans "
            "<WAN2GP_ROOT>/ffmpeg_bins, puis dans le PATH. ffprobe est aussi accepté "
            "à la racine du clone."
            + quarantined
            + " Sous Debian/Ubuntu : apt install ffmpeg."
        )
    _require_encoder(ffmpeg, _wan_mp4_target(root))
    _require_ffprobe(ffprobe)
    _require_bundle_libs(ffmpeg, root)
    _require_ld_library_shadow(ffmpeg, root)
    binaries = FfmpegBinaries(ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    os.environ["FFMPEG_BINARY"] = binaries.ffmpeg
    os.environ["FFPROBE_BINARY"] = binaries.ffprobe
    os.environ["IMAGEIO_FFMPEG_EXE"] = binaries.ffmpeg
    return binaries


def _binary_name(name: str) -> str:
    if os.name == "nt" and not name.endswith(".exe"):
        return f"{name}.exe"
    return name


def _resolve_binary(name: str, root: Path | None) -> Path | None:
    filename = _binary_name(name)
    if root is not None:
        bundled = root / "ffmpeg_bins" / filename
        if bundled.is_file():
            return bundled
        # Seul ffmpeg.exe / ffmpeg est mis en quarantaine. ffprobe à la racine reste utilisé.
        if name != "ffmpeg":
            at_root = root / filename
            if at_root.is_file():
                return at_root
    found = shutil.which(filename)
    if found is None and filename != name:
        found = shutil.which(name)
    return Path(found) if found else None


def _path_kind(path: Path) -> str:
    if path.is_dir():
        return "dossier"
    if path.is_fifo():
        return "tube"
    if path.is_socket():
        return "socket"
    return "chemin"


def _exists_and_is_not_a_file(path: Path) -> bool:
    """``Path.exists()`` sans ``is_file`` : le cas que ``download_ffmpeg`` retient."""
    try:
        return path.exists() and not path.is_file()
    except OSError:
        return True


def _wan_non_file_selection(name: str, root: Path) -> Path | None:
    """Chemin non exécutable que ``_resolve_path`` copierait dans l'environnement.

    L'ordre est celui de WanGP : ``ffmpeg_bins``, puis la racine du clone.
    Un vrai fichier dans ``ffmpeg_bins`` gagne sur un homonyme à la racine.
    Un fichier ``ffmpeg`` à la racine est mis en quarantaine avant cette
    recherche, donc il n'est pas le binaire retenu.
    """
    filename = _binary_name(name)
    bundled = root / "ffmpeg_bins" / filename
    if _exists_and_is_not_a_file(bundled):
        return bundled
    if bundled.is_file():
        return None
    at_root = root / filename
    if name == "ffmpeg" and at_root.is_file():
        return None
    if _exists_and_is_not_a_file(at_root):
        return at_root
    return None


def _require_wan_binary_is_a_file(root: Path | None) -> None:
    """Refuse un ``ffmpeg`` ou ``ffprobe`` que WanGP lancerait sans que ce soit un fichier.

    ``download_ffmpeg`` écrit ce chemin dans ``FFMPEG_BINARY`` ou
    ``FFPROBE_BINARY``. ``save_video`` imprime ``error saving`` après l'échec
    du mux, et ``process_tasks_cli`` compte quand même la tâche.
    """
    if root is None:
        return
    found = [
        (name, selected)
        for name in ("ffmpeg", "ffprobe")
        if (selected := _wan_non_file_selection(name, root)) is not None
    ]
    if not found:
        return
    shown = " ; ".join(f"{name} : {_path_kind(path)} {path}" for name, path in found)
    raise Wan2GPError(
        f"Wan2GP prendrait un chemin qui n'est pas un exécutable ({shown}). "
        "download_ffmpeg le choisit avec Path.exists() et l'écrit dans FFMPEG_BINARY "
        "ou FFPROBE_BINARY. Seul un fichier ffmpeg à la racine du clone part en "
        "quarantaine. save_video imprime alors error saving et le processus quitte "
        "avec le code 0. Retirer ce dossier, ce tube ou ce socket. Un binaire réel "
        "dans ffmpeg_bins reste prioritaire sur un homonyme à la racine."
    )


def _require_mp4_container(payload: dict[str, object], where: str) -> None:
    """Refuse un ``video_container`` que WanGP n'écrira pas en ``.mp4``.

    La clé absente vaut ``mp4``. Une clé présente est copiée telle quelle dans
    l'extension (WanGP 18.84, ``generate_media``). ``null`` devient ``.None``,
    une chaîne vide un point final, des espaces un suffixe différent de ``.mp4``.
    ``save_video`` normalise ensuite cette valeur pour la validation (donc
    ``mp4``) et n'aligne pas le nom du fichier. Le CLI sort avec le code 0.
    """
    if "video_container" not in payload:
        return
    raw = payload["video_container"]
    if isinstance(raw, str) and raw.strip() == raw and raw.lower() == "mp4":
        return
    if isinstance(raw, str) and raw.strip() and raw.strip() == raw:
        raise Wan2GPError(
            f"video_container={raw!r} dans {where}. "
            "WanGP écrirait ce conteneur et le dossier de sortie n'aurait pas de MP4. "
            "Remettre video_container à mp4."
        )
    shown = "null" if raw is None else repr(raw)
    raise Wan2GPError(
        f"video_container={shown} dans {where}. "
        "WanGP copie cette valeur telle quelle dans l'extension du fichier. "
        "null, une chaîne vide ou des espaces donnent un nom qui ne finit pas par .mp4. "
        "La validation interne ramène pourtant le conteneur à mp4, et le processus quitte avec le code 0. "
        "Remettre video_container à mp4, ou retirer la clé."
    )


def _wan_mp4_target(root: Path | None) -> str:
    """Encodeur ffmpeg exigé par ``wgp_config.json``. Défaut WanGP 18.84 : libx264.

    Le conteneur vient du fichier de config du clone, pas de la file. ``mkv``
    ou ``mov`` produit une vidéo que ce pipeline ne peut pas accepter comme MP4.
    Une clé ``video_container`` nulle ou vide non plus : le fichier n'a pas
    l'extension ``.mp4``.
    """
    codec = "libx264_8"
    payload: dict[str, object] = {}
    config_path = None if root is None else root / "wgp_config.json"
    if config_path is not None and config_path.is_file():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Wan2GPError(f"wgp_config.json illisible ({config_path}) : {exc}") from exc
        if not isinstance(loaded, dict):
            raise Wan2GPError(f"wgp_config.json doit être un objet ({config_path}).")
        payload = loaded
        codec = str(payload.get("video_output_codec") or "libx264_8").strip().lower() or "libx264_8"
    where = "wgp_config.json" if config_path is None else str(config_path)
    _require_mp4_container(payload, where)
    refused = _NON_MP4_CODECS.get(codec)
    if refused is not None:
        raise Wan2GPError(f"{refused} Réglage video_output_codec={codec!r} dans {where}.")
    encoder = _MP4_ENCODER_FOR_CODEC.get(codec)
    if encoder is None:
        raise Wan2GPError(
            f"video_output_codec={codec!r} dans {where} n'est pas un encodeur MP4 connu. "
            "Utiliser libx264_8."
        )
    return encoder


def _require_encoder(ffmpeg: Path, encoder: str) -> None:
    try:
        encoders = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise Wan2GPError(f"Impossible de lancer {ffmpeg} : {exc}") from exc
    if encoders.returncode != 0 or encoder not in encoders.stdout:
        raise Wan2GPError(
            f"{ffmpeg} n'offre pas {encoder}. Wan2GP prend ce binaire avant le PATH "
            "et ne peut pas muxer le MP4."
        )


def _require_bundle_libs(ffmpeg: Path, root: Path | None) -> None:
    """Wan2GP n'accepte un ffmpeg de ``ffmpeg_bins`` que si ``libavdevice.so*`` y est.

    Sans ces bibliothèques, ``download_ffmpeg`` tente un téléchargement et peut
    remplacer le binaire, puis quitter sans avoir muxé de MP4.
    """
    if os.name == "nt" or root is None or ffmpeg.parent != root / "ffmpeg_bins":
        return
    bundle = root / "ffmpeg_bins"
    if any(bundle.glob("libavdevice.so*")):
        return
    raise Wan2GPError(
        f"{ffmpeg} est dans ffmpeg_bins sans libavdevice.so*. "
        "Wan2GP traite ce dossier comme un build incomplet : il tente de le retélécharger "
        "et peut ensuite ne pas muxer de MP4. Retirer ce binaire pour utiliser le ffmpeg du PATH, "
        "ou y déposer le build complet (binaire et bibliothèques)."
    )


def _is_shared_library(path: Path) -> bool:
    """Nom de bibliothèque dynamique Linux (``libx264.so``, ``libx264.so.164``)."""
    name = path.name
    return name.endswith(".so") or ".so." in name


def _require_ld_library_shadow(ffmpeg: Path, root: Path | None) -> None:
    """Refuse un ``ffmpeg_bins`` qui n'est pas le dossier de l'encodeur retenu.

    ``download_ffmpeg`` préfixe ``LD_LIBRARY_PATH`` avec ce dossier avant de
    savoir quel binaire sera lancé. Une ``libx264.so*`` ou une ``libav*.so*``
    partielle y fait échouer le mux du système, sans code de sortie non nul.
    ``save_video`` imprime ``error saving`` et le CLI compte quand même la tâche.
    """
    if os.name == "nt" or root is None:
        return
    bundle = root / "ffmpeg_bins"
    if ffmpeg.parent == bundle or not bundle.is_dir():
        return
    shadows = sorted(path.name for path in bundle.iterdir() if path.is_file() and _is_shared_library(path))
    if not shadows:
        return
    shown = ", ".join(shadows)
    raise Wan2GPError(
        f"{bundle} contient {shown} alors que l'encodeur retenu est {ffmpeg}. "
        "Wan2GP ajoute ce dossier en tête de LD_LIBRARY_PATH : ces bibliothèques "
        "peuvent empêcher le mux MP4, puis le processus quitte avec le code 0. "
        "Retirer ffmpeg_bins pour garder le ffmpeg du PATH, ou y déposer le build "
        "complet (ffmpeg, ffprobe et libavdevice.so*)."
    )


def _require_ffprobe(ffprobe: Path) -> None:
    try:
        probe = subprocess.run(
            [str(ffprobe), "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise Wan2GPError(f"Impossible de lancer {ffprobe} : {exc}") from exc
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        raise Wan2GPError(f"ffprobe ne démarre pas ({ffprobe}). {detail}".strip())


def ffmpeg_env(binaries: FfmpegBinaries) -> dict[str, str]:
    """Variables lues par WanGP, imageio et les outils qui cherchent ffmpeg hors PATH."""
    env = os.environ.copy()
    env["FFMPEG_BINARY"] = binaries.ffmpeg
    env["FFPROBE_BINARY"] = binaries.ffprobe
    env["IMAGEIO_FFMPEG_EXE"] = binaries.ffmpeg
    binary_dir = str(Path(binaries.ffmpeg).resolve().parent)
    env["PATH"] = binary_dir + os.pathsep + env.get("PATH", "")
    return env


def _settings_payload(
    request: GenerationRequest,
    model_type: str,
    frames: int,
    resolution: str,
    stem: str,
    settings_version: float,
) -> dict[str, object]:
    prompt = request.prompt
    payload: dict[str, object] = {
        "model_type": model_type,
        "base_model_type": model_type,
        "settings_version": settings_version,
        "image_mode": 0,
        "prompt": prompt.prompt,
        "negative_prompt": prompt.negative_prompt,
        "resolution": resolution,
        "video_length": frames,
        "force_fps": str(request.settings.fps),
        "num_inference_steps": request.settings.steps,
        "guidance_scale": request.settings.guidance_scale,
        "seed": prompt.seed,
        "repeat_generation": 1,
        "output_filename": stem,
        "prompt_enhancer": "",
        "multi_prompts_gen_type": "FG",
    }
    payload.update(request.settings.extra_settings)
    _lock_text_to_video(payload, model_type, settings_version)
    _apply_profile(payload, request.settings, settings_version)
    return payload


def _profile_fields(settings: WanSettings, settings_version: float) -> dict[str, object]:
    """Profil mémoire dans le champ que cette version de WanGP lit vraiment.

    Avant 2.81 la clé est ``override_profile``. Ensuite c'est ``overrides``,
    par exemple ``memory_profile=4, attention_mode='sdpa'``.
    """
    if settings_version < _OVERRIDES_VERSION:
        return {"override_profile": settings.profile}
    values: dict[str, object] = {}
    if settings.profile != -1:
        values["memory_profile"] = settings.profile
    if settings.attention:
        values["attention_mode"] = settings.attention
    rendered = ", ".join(f"{name}={values[name]!r}" for name in _OVERRIDE_FIELDS if name in values)
    return {"overrides": rendered}


def _apply_profile(payload: dict[str, object], settings: WanSettings, settings_version: float) -> None:
    fields = _profile_fields(settings, settings_version)
    if "overrides" in fields:
        payload.pop("override_profile", None)
    else:
        payload.pop("overrides", None)
    payload.update(fields)


def _lock_text_to_video(payload: dict[str, object], model_type: str, settings_version: float) -> None:
    requested = payload.get("model_type", model_type)
    if not isinstance(requested, str) or _is_image_model(requested) or requested != model_type:
        raise Wan2GPError(
            f"model_type {requested!r} refusé. La génération utilise {model_type} "
            "(text-to-video). Fun_InP et les modèles I2V peuvent enregistrer un JPEG puis quitter en succès."
        )
    image_mode = payload.get("image_mode", 0)
    if image_mode != 0:
        raise Wan2GPError(
            f"image_mode {image_mode!r} refusé. Le mode image de Wan2GP écrit un JPEG, pas un MP4."
        )
    for key in ATTACHMENT_KEYS:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            raise Wan2GPError(
                f"{key} est refusé. Une image de départ bascule vers Fun_InP_1.3B, qui peut ne pas produire de MP4."
            )
    payload["model_type"] = model_type
    payload["base_model_type"] = model_type
    payload["settings_version"] = settings_version
    payload["image_mode"] = 0
    payload["multi_prompts_gen_type"] = "FG"


def _write_queue(path: Path, payload: dict[str, object]) -> None:
    manifest = [{"id": 1, "params": payload}]
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "queue.json",
            json.dumps(manifest, ensure_ascii=False, indent=4),
        )


def _run_subprocess(
    command: tuple[str, ...],
    root: Path,
    log_path: Path,
    timeout: int,
    binaries: FfmpegBinaries,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Wan2GP : %s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=ffmpeg_env(binaries),
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


def _require_mp4(directory: Path, before: dict[Path, int], ffprobe: str, log_path: Path) -> Path:
    fresh = _new_files(directory, before)
    valid = [path for path in fresh if path.suffix.lower() == ".mp4" and _mp4_problem(path, ffprobe) is None]
    if valid:
        return max(valid, key=lambda path: path.stat().st_mtime_ns).resolve()

    images = [path.name for path in fresh if path.suffix.lower() in IMAGE_SUFFIXES]
    rejected = [
        f"{path.name} ({reason})"
        for path in fresh
        if path.suffix.lower() == ".mp4" and (reason := _mp4_problem(path, ffprobe)) is not None
    ]
    details: list[str] = []
    if images:
        kind = "JPEG" if any(name.lower().endswith((".jpg", ".jpeg")) for name in images) else "image"
        details.append(
            f"Wan2GP a écrit un {kind} ({', '.join(images)}) à la place d'une vidéo. "
            "Ce comportement est celui de Fun_InP_1.3B en I2V, ou d'un image_mode différent de 0."
        )
    if rejected:
        details.append("Fichiers .mp4 illisibles comme vidéo : " + "; ".join(rejected) + ".")
    others = sorted(
        (
            path
            for path in fresh
            if path.suffix.lower() not in IMAGE_SUFFIXES and path.suffix.lower() != ".mp4"
        ),
        key=lambda path: path.name,
    )
    if others:
        names = ", ".join(path.name for path in others)
        containers = sorted(
            {path.suffix.lower() for path in others if path.suffix.lower() in OTHER_VIDEO_SUFFIXES}
        )
        if containers:
            shown = ", ".join(suffix.lstrip(".") for suffix in containers)
            details.append(
                f"WanGP a écrit un autre conteneur ({names}), pas un MP4. "
                f"video_container de wgp_config.json vaut probablement {shown}. "
                "Le remettre à mp4."
            )
        else:
            details.append(f"Fichiers nouveaux qui ne sont pas des MP4 : {names}.")
    if not details:
        details.append(f"Aucun fichier nouveau dans {directory}.")
    details.append("Wan2GP a quitté avec le code 0.")
    details.append(f"Journal : {log_path}.")
    excerpt = _failure_excerpt(log_path)
    if excerpt:
        details.append(excerpt)
    raise Wan2GPError(" ".join(details))


def _mp4_problem(path: Path, ffprobe: str) -> str | None:
    if not path.is_file() or path.stat().st_size < 8:
        return "fichier vide"
    payload = _probe(path, ffprobe, count_frames=False)
    if payload is None:
        return "ffprobe ne lit pas ce fichier"
    video = _video_stream(payload)
    if video is None:
        return "aucune piste vidéo"
    codec = str(video.get("codec_name") or "").lower()
    if codec in IMAGE_CODECS:
        return f"codec image {codec}"
    duration = _duration(payload, video)
    frames = _frame_count(video)
    if frames is None and duration < MIN_VIDEO_SECONDS:
        counted = _probe(path, ffprobe, count_frames=True)
        if counted is not None:
            counted_video = _video_stream(counted)
            if counted_video is not None:
                frames = _frame_count(counted_video)
                duration = max(duration, _duration(counted, counted_video))
    if duration >= MIN_VIDEO_SECONDS or (frames is not None and frames >= MIN_VIDEO_FRAMES):
        return None
    shown = "inconnue" if frames is None else str(frames)
    return f"trop court ({duration:.3f} s, {shown} images)"


def _probe(path: Path, ffprobe: str, count_frames: bool) -> dict[str, object] | None:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_type,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    if count_frames:
        command.insert(1, "-count_frames")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _video_stream(payload: dict[str, object]) -> dict[str, object] | None:
    streams = payload.get("streams")
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type", "video") == "video":
            return stream
    return None


def _duration(payload: dict[str, object], stream: dict[str, object]) -> float:
    format_info = payload.get("format")
    raw = None
    if isinstance(format_info, dict):
        raw = format_info.get("duration")
    if raw in (None, "", "N/A"):
        raw = stream.get("duration")
    try:
        return float(raw or 0)
    except (TypeError, ValueError):
        return 0.0


def _frame_count(stream: dict[str, object]) -> int | None:
    for key in ("nb_read_frames", "nb_frames"):
        raw = stream.get(key)
        if raw in (None, "", "N/A"):
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _reject_image_input(request: GenerationRequest) -> None:
    if request.image_path is None:
        return
    raise Wan2GPError(
        "L'image de départ est refusée. Fun_InP_1.3B (I2V) peut écrire un JPEG et quitter "
        "avec le code 0. Ce pipeline ne lance que le text-to-video, sans --image."
    )


def _is_image_model(model_type: str) -> bool:
    token = model_type.lower().replace("-", "_")
    return any(marker in token for marker in ("i2v", "fun_inp", "funinp", "flf2v", "inpaint"))


def _model_type(configured: str) -> str:
    if _is_image_model(configured):
        raise Wan2GPError(
            f"Le modèle {configured} est un modèle image-vers-vidéo. "
            "Utiliser t2v_1.3B. Fun_InP_1.3B enregistre parfois un JPEG à la place du MP4."
        )
    return configured


def _require_root(wan: WanSettings) -> Path:
    if wan.root is None:
        raise Wan2GPError(
            "WAN2GP_ROOT est vide. Renseigner wan2gp.root ou la variable WAN2GP_ROOT."
        )
    return wan.root


def _file_index(directory: Path) -> dict[Path, int]:
    return {
        path: path.stat().st_mtime_ns
        for path in directory.rglob("*")
        if path.is_file()
    }


def _new_files(directory: Path, before: dict[Path, int]) -> list[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and before.get(path) != path.stat().st_mtime_ns
    ]


def _relay_output(stream: TextIO, log: TextIO) -> None:
    for line in stream:
        log.write(line)
        log.flush()
        sys.stdout.write(line)


_LOG_MARKERS = (
    "[SKIP]",
    "[ERROR]",
    "nothing to process",
    "failed validation",
    "error saving",
    "Failed to download FFmpeg",
)


def _failure_excerpt(path: Path) -> str:
    """Reprend les lignes qui expliquent un succès CLI sans MP4.

    ``save_video`` imprime ``error saving`` puis retourne sans chemin. Le CLI
    compte la tâche comme réussie. Ce message est souvent au milieu du journal,
    bien avant les quarante dernières lignes de progression.
    """
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    matched = [line for line in lines if any(marker in line for marker in _LOG_MARKERS)]
    if not matched:
        return ""
    return "\n".join(matched[-12:])


def _tail(path: Path, lines: int = 40) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _assert_never(value: Never) -> Never:
    raise Wan2GPError(f"Backend non géré: {value}.")
