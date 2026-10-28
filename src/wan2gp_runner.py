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
import struct
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
_NEEDED_LIBRARY_RE = re.compile(r"Shared library: \[([^\]]+)\]")
_INTERPRETER_RE = re.compile(r"Requesting program interpreter:\s*(\S+)")
# readelf -d est relancé pour chaque bibliothèque de la fermeture. Le cache
# évite de relire les mêmes fichiers système à chaque contrôle.
_READELF_NEEDED: dict[tuple[str, int, int], frozenset[str] | None] = {}
_LDCONFIG_SONAMES: dict[str, Path] | None = None
_GLIBC_HWCAPS: frozenset[str] | None = None
_HWCAP_HEADER = "Subdirectories of glibc-hwcaps directories, in priority order:"
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
# Deux images rawvideo tiennent largement dans cette borne. Au-delà, un tube
# du soname NEEDED a bloqué le chargeur : on échoue au lieu d'attendre.
_MUX_PROBE_TIMEOUT_SECONDS = 20
# ``-encoders`` sur le binaire du système revient tout de suite. Un script
# ``exec ffmpeg`` relancé avec le PATH de WanGP ne revient pas : on n'attend pas
# le délai GPU pour le dire.
_PATH_PROBE_TIMEOUT_SECONDS = 5
# Sonames que ``Path.glob("libavdevice.so*")`` verrait si le dossier était
# listable. Le numéro dépasse les builds FFmpeg courants (libavdevice.so.60
# et .61 sur ce pipeline).
_LIBAVDEVICE_PROBE_NAMES = ("libavdevice.so", *(f"libavdevice.so.{index}" for index in range(1, 81)))
_LIBSTDCXX_NAME = "libstdc++.so.6"
_LIBSTDCXX_MARKER = "WAN2GP_LIBSTDCXX\t"
# Même ordre que wgp.py : tqdm, puis la lecture de /proc/self/maps, puis CDLL.
_LIBSTDCXX_PROBE = """
import os
import sys
from pathlib import Path

try:
    import tqdm  # noqa: F401
except Exception:
    pass

loaded = False
maps = Path("/proc/self/maps")
if maps.is_file():
    try:
        loaded = "libstdc++.so.6" in maps.read_text(encoding="utf-8", errors="replace")
    except OSError:
        loaded = False
if loaded:
    raise SystemExit(0)

prefixes = [sys.prefix, sys.exec_prefix, os.environ.get("CONDA_PREFIX")]
seen: set[str] = set()
for prefix in prefixes:
    if not prefix:
        continue
    path = str(Path(prefix) / "lib" / "libstdc++.so.6")
    key = os.path.normcase(os.path.abspath(path))
    if key in seen or not os.path.isfile(path):
        continue
    seen.add(key)
    try:
        import ctypes

        mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 0)
        ctypes.CDLL(path, mode=mode)
    except OSError:
        continue
    print("WAN2GP_LIBSTDCXX\\t" + os.path.dirname(path), flush=True)
    raise SystemExit(0)
"""

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

    binaries = require_ffmpeg(root, wan.python or sys.executable)
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


def require_ffmpeg(root: Path | None = None, python: str | None = None) -> FfmpegBinaries:
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
    de MP4. Quand le binaire est dans ``ffmpeg_bins``, ``-encoders`` sans
    ``LD_LIBRARY_PATH`` peut encore voir le libx264 du système : une
    ``libavcodec.so*`` du bon soname, vide ou trop courte, à côté du binaire
    casse le chargement dès que WanGP préfixe ce dossier. Le même préfixe
    est posé quand l'encodeur vient du PATH. Un dossier, un tube ou un
    périphérique (souvent un lien vers ``/dev/null``) qui porte le soname
    réellement chargé (``libavcodec.so.60``, pas un autre numéro) n'est pas
    un fichier : le contrôle des ``*.so`` le laisse passer, ``-encoders``
    démarre, puis le chargeur échoue (``file too short``, ``invalid ELF header``)
    ou se bloque. Le même tube à côté du binaire, dans ``ffmpeg_bins``,
    satisfait le glob ``libavdevice.so*`` : ``_libs_present`` croit le build
    complet, et le mux d'essai ouvrirait le tube sans jamais revenir.
    Un soname qui n'est pas un NEEDED direct du binaire casse le même
    démarrage : ``libx264.so.164`` est demandé par ``libavcodec``. ``-encoders``
    sans préfixe réussit, puis le chargeur ouvre ce nom dès que
    ``LD_LIBRARY_PATH`` commence par ``ffmpeg_bins``.
    glibc ouvre d'abord ``ffmpeg_bins/glibc-hwcaps/<niveau>/<soname>``
    (niveaux de ``ld.so --help``) avant le fichier du même nom à la racine.
    Un fichier trop court, un dossier ou un tube à cet endroit fait échouer
    le mux, ou le bloque, alors que le contrôle à la racine de ``ffmpeg_bins``
    ne voit rien. Sans ``readelf``, le chemin PT_INTERP est lu dans l'en-tête
    ELF du binaire : sinon les niveaux restent inconnus et ce soname passe.
    Un ``glibc-hwcaps`` sans droit de lecture n'est pas listable, mais le
    chargeur ouvre le chemin complet dès que le dossier est exécutable.
    Le runner statue alors chaque soname connu, sans parcourir le dossier.
    ``ffmpeg_bins`` lui-même peut être dans le même état : ``iterdir`` échoue,
    ``download_ffmpeg`` préfixe quand même ``LD_LIBRARY_PATH``, et le chargeur
    ouvre ``ffmpeg_bins/<soname>`` comme le chemin sous ``glibc-hwcaps``.
    Ce même dossier rend ``Path.glob("libavdevice.so*")`` vide. S'il est
    inscriptible, le téléchargement de WanGP peut remplacer le binaire.
    S'il ne l'est pas, ``shutil.copy2`` peut quand même tronquer un
    ``ffmpeg`` déjà présent : écraser un fichier ne demande pas le droit
    d'écriture du dossier. Le nouvel exécutable arrive sans les
    bibliothèques, qui ne peuvent pas être créées. ``save_video`` imprime
    ``error saving`` et le CLI sort 0.     Un lien vers un binaire non
    inscriptible, comme celui du système, laisse le mux d'essai trancher
    quand ``lstat`` voit ``libavdevice.so*``. Un lien de soname dont le
    ``stat`` lève ``PermissionError`` (dossier parent en mode 000) est
    ignoré : le chargeur passe au fichier suivant.
    Le chemin retenu est rendu absolu sans suivre le lien. Le processus a
    pour cwd le clone : imageio exécute ``IMAGEIO_FFMPEG_EXE`` tel quel.
    Un chemin relatif au dossier du pipeline n'existe plus depuis ce cwd,
    ``save_video`` imprime ``error saving`` et le CLI sort 0.
    ``download_ffmpeg`` met ensuite ``ffmpeg_bins`` puis la racine du clone
    en tête du PATH. Un script ``exec ffmpeg`` (sans chemin) lancé par son
    chemin absolu passe ``-encoders`` : le PATH d'origine trouve le binaire
    du système, qui annonce libx264. Au mux, le même script se retrouve
    lui-même et ne revient pas, ou s'arrête quand la chaîne de processus
    casse. ``save_video`` imprime ``error saving`` et le CLI sort 0, sans MP4.
    Un ``exec /usr/bin/ffmpeg`` reste accepté.
    ``download_ffmpeg`` ne préfixe pas le ``ffmpeg_bins`` du cwd. Il résout
    ``shared/ffmpeg_setup.py`` (``Path(__file__).resolve()``). Un lien vers
    un autre clone, ou un ``wgp.py`` qui est lui-même un lien (``sys.path[0]``
    devient le dossier réel), charge les bibliothèques de cet autre dossier.
    imageio exécute pourtant ``IMAGEIO_FFMPEG_EXE``, souvent le binaire du
    clone. Une ``libx264.so*`` trop courte là-bas casse le chargeur,
    ``-encoders`` sans ce préfixe annonce encore libx264, ``save_video``
    imprime ``error saving`` et le CLI sort 0.
    Avant ce téléchargement, ``preload_preferred_libstdcxx``
    (``shared/native_runtime.py``) ajoute le dossier de ``libstdc++.so.6``
    si le processus ne l'a pas encore chargé. Le fichier est cherché dans
    ``sys.prefix/lib``, ``sys.exec_prefix/lib``, puis ``CONDA_PREFIX/lib``.
    ``download_ffmpeg`` préfixe ensuite ``ffmpeg_bins`` : le dossier de
    ``libstdc++`` reste sur ``LD_LIBRARY_PATH``. Une ``libx264.so*`` trop
    courte à cet endroit, ou un tube du même soname, casse le mux du binaire
    système. ``-encoders`` sans ce préfixe annonce encore libx264,
    ``save_video`` imprime ``error saving`` et le CLI sort 0. Un
    ``libstdc++.so.6`` qui n'est pas un ELF chargeable est ignoré : WanGP
    ne préfixe pas ce dossier.
    ``os.path.dirname`` est recopié tel quel. Un ``CONDA_PREFIX`` relatif
    reste relatif. Le processus a pour cwd le clone : le chargeur ouvre
    ``<clone>/<préfixe>/lib/<soname>``, pas le même chemin depuis le dossier
    du pipeline. Un fichier trop court ou un tube seulement sous le clone
    casse le mux, ``-encoders`` lancé ailleurs annonce encore libx264,
    ``save_video`` imprime ``error saving`` et le CLI sort 0.
    ``os.path.isfile`` et ``CDLL`` utilisent ce même cwd. La sonde tourne
    donc dans le clone : un ``libstdc++.so.6`` présent seulement sous ce
    dossier est retenu, même si le dossier du pipeline n'a pas le préfixe.
    Une bibliothèque trop courte à côté casse alors le mux.
    ``download_ffmpeg`` conserve ensuite les entrées déjà présentes de
    ``LD_LIBRARY_PATH``, après ``ffmpeg_bins``. Une entrée relative est
    ouverte depuis le clone. ``-encoders`` part de ce cwd : depuis le
    dossier du pipeline, le libx264 du système répond encore.
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
    preload = _libstdcxx_preload_directory(python or sys.executable, root)
    encoder = _wan_mp4_target(root)
    _require_inherited_library_traps(ffmpeg, root)
    _require_encoder(ffmpeg, encoder, root)
    _require_ffprobe(ffprobe, root)
    _require_bundle_libs(ffmpeg, root)
    _require_ld_library_shadow(ffmpeg, root)
    _require_preload_library_traps(ffmpeg, root, preload)
    _require_bundled_mux(ffmpeg, ffprobe, root, encoder, preload)
    ffmpeg = _absolute_binary(ffmpeg)
    ffprobe = _absolute_binary(ffprobe)
    binaries = FfmpegBinaries(ffmpeg=str(ffmpeg), ffprobe=str(ffprobe))
    os.environ["FFMPEG_BINARY"] = binaries.ffmpeg
    os.environ["FFPROBE_BINARY"] = binaries.ffprobe
    os.environ["IMAGEIO_FFMPEG_EXE"] = binaries.ffmpeg
    return binaries


def _absolute_binary(path: Path) -> Path:
    """Chemin absolu du binaire, sans remplacer un lien par sa cible.

    ``Path.resolve()`` suivrait ``ffmpeg_bins/ffmpeg`` jusqu'au binaire du
    système. WanGP choisit pourtant ce lien, et le mux d'essai l'a validé
    avec le ``LD_LIBRARY_PATH`` du clone. ``absolute()`` laisse le lien en
    place et reste valable quand le cwd devient le clone.
    """
    return path.absolute()


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
    if path.is_char_device() or path.is_block_device():
        return "périphérique"
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
        "avec le code 0. Retirer ce dossier, ce tube, ce socket ou ce périphérique. "
        "Un binaire réel dans ffmpeg_bins reste prioritaire sur un homonyme à la racine."
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


def _probe_cwd(root: Path | None) -> str | None:
    """Cwd du processus WanGP, pour ouvrir un chemin relatif au même endroit."""
    if os.name == "nt" or root is None:
        return None
    return str(root.absolute())


def _probe_binary(path: Path, root: Path | None) -> Path:
    """Binaire absolu quand le probe change de cwd.

    Un ``ffmpeg_bins/ffmpeg`` relatif au dossier du pipeline n'existe plus
    depuis le clone. ``absolute()`` ne suit pas le lien.
    """
    if os.name == "nt" or root is None:
        return path
    return _absolute_binary(path)


def _relative_ld_entries() -> list[str]:
    """Entrées relatives de ``LD_LIBRARY_PATH``, ouvertes depuis le cwd."""
    found: list[str] = []
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if part and not Path(part).is_absolute():
            found.append(part)
    return found


def _loader_failed(returncode: int, detail: str) -> bool:
    text = detail.lower()
    return (
        returncode == 127
        or "error while loading shared libraries" in text
        or "file too short" in text
        or "cannot read file data" in text
    )


def _relative_ld_failure_message(
    binary: Path,
    root: Path | None,
    returncode: int,
    detail: str,
) -> str | None:
    """Message si une entrée relative de ``LD_LIBRARY_PATH`` casse le chargeur.

    ``download_ffmpeg`` la conserve. Le processus l'ouvre depuis le clone.
    """
    if root is None or os.name == "nt" or not _loader_failed(returncode, detail):
        return None
    relative = _relative_ld_entries()
    if not relative:
        return None
    shown = os.pathsep.join(relative)
    libraries = os.environ.get("LD_LIBRARY_PATH", "")
    return (
        f"{binary} ne démarre pas avec LD_LIBRARY_PATH={libraries} "
        f"depuis le cwd du clone {root.absolute()}. "
        f"L'entrée relative {shown} est ouverte sous ce dossier, pas depuis le dossier du pipeline. "
        "download_ffmpeg (_ensure_library_path dans shared/ffmpeg_setup.py) préfixe "
        "ffmpeg_bins puis conserve cette entrée. "
        "Une bibliothèque du soname demandé, vide ou trop courte, masque celle du système. "
        "ffmpeg -encoders, lancé depuis le dossier du pipeline, peut encore annoncer libx264. "
        "save_video imprime error saving et le processus quitte avec le code 0. "
        f"{detail}".rstrip()
    )


def _inherited_library_dirs(root: Path | None) -> list[tuple[str, Path]]:
    """Dossiers de ``LD_LIBRARY_PATH`` tels que le clone va les ouvrir."""
    found: list[tuple[str, Path]] = []
    for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not part:
            continue
        path = Path(part)
        seen = path if path.is_absolute() or root is None else _from_process_cwd(path, root)
        found.append((part, seen))
    return found


def _require_inherited_library_traps(ffmpeg: Path, root: Path | None) -> None:
    """Refuse un soname non fichier déjà présent dans ``LD_LIBRARY_PATH``.

    ``download_ffmpeg`` préfixe ``ffmpeg_bins`` puis garde le reste. Une
    entrée relative est ouverte depuis le cwd du clone. Un tube de ce
    soname bloquerait ``-encoders``. ``lstat`` suffit. ``ffmpeg_bins`` est
    contrôlé à part. Un fichier régulier du même soname dans ce dossier
    est ouvert avant.
    """
    if os.name == "nt":
        return
    bundle = _library_bundle(root) if root is not None else None
    for part, seen in _inherited_library_dirs(root):
        if bundle is not None and _same_directory(seen, bundle):
            continue
        try:
            if not seen.is_dir():
                continue
        except OSError:
            continue
        entries = _ld_search_entries(seen, ffmpeg)
        needed = (
            _needed_sonames(ffmpeg, seen)
            if any(_nonfile_library_name(path) for path in entries)
            else set()
        )
        masked = _preload_masked_sonames(ffmpeg, root, seen) if root is not None else set()
        traps = [
            trap
            for trap in _library_traps(entries, needed, seen)
            if Path(trap.split(" ", 1)[1]).name not in masked
        ]
        if not traps:
            continue
        shown = ", ".join(traps)
        located = part
        if Path(part) != seen:
            located = f"{part} (ouvert depuis le cwd du clone : {seen})"
        raise Wan2GPError(
            f"LD_LIBRARY_PATH contient {located} avec {shown}, soname chargé par {ffmpeg}. "
            "download_ffmpeg (_ensure_library_path dans shared/ffmpeg_setup.py) "
            "préfixe ffmpeg_bins puis conserve cette entrée. "
            "Le processus a pour cwd le clone : une entrée relative y est ouverte, "
            "pas depuis le dossier du pipeline. "
            "Ce n'est pas un fichier. ffmpeg -encoders, lancé depuis le dossier du pipeline, "
            "démarre encore. Un dossier ou un périphérique donne file too short ou "
            "invalid ELF header. Un tube bloque l'ouverture, y compris sous "
            "glibc-hwcaps/<niveau>/. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            "Retirer cette entrée de LD_LIBRARY_PATH. "
            "Un fichier régulier du même soname déjà présent dans ffmpeg_bins est ouvert avant."
        )


def _require_encoder(ffmpeg: Path, encoder: str, root: Path | None) -> None:
    """``-encoders`` avec le PATH et le cwd que ``download_ffmpeg`` va installer.

    Sans ce préfixe, un script ``exec ffmpeg`` délègue au binaire du système
    et annonce libx264. WanGP met ``ffmpeg_bins`` puis la racine du clone
    devant le PATH avant le mux : le script se relance et n'écrit pas de MP4.
    Le cwd est celui du clone. Une entrée relative de ``LD_LIBRARY_PATH``
    y est ouverte : depuis le dossier du pipeline, libx264 répond encore.
    """
    binary = _probe_binary(ffmpeg, root)
    try:
        encoders = subprocess.run(
            [str(binary), "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            env=_command_env(root),
            cwd=_probe_cwd(root),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(_path_reexec_message(ffmpeg, root)) from exc
    except OSError as exc:
        raise Wan2GPError(f"Impossible de lancer {ffmpeg} : {exc}") from exc
    if encoders.returncode != 0 or encoder not in encoders.stdout:
        detail = (encoders.stderr or encoders.stdout).strip()
        relative = _relative_ld_failure_message(binary, root, encoders.returncode, detail)
        if relative is not None:
            raise Wan2GPError(relative)
        raise Wan2GPError(
            f"{ffmpeg} n'offre pas {encoder}. Wan2GP prend ce binaire avant le PATH "
            "et ne peut pas muxer le MP4."
        )


def _ffmpeg_library_root(root: Path) -> Path:
    """Racine dont ``download_ffmpeg`` préfixe ``LD_LIBRARY_PATH``.

    ``Path(__file__).resolve().parents[1]`` suit un lien
    ``shared/ffmpeg_setup.py`` vers un autre clone. Si ``wgp.py`` est un lien,
    ``sys.path[0]`` est le dossier réel et le module est chargé là. Le cwd du
    processus reste le clone passé au runner : ce n'est pas ce dossier qui
    fournit les bibliothèques.
    """
    if os.name == "nt":
        return root
    resolved = _resolved_ffmpeg_setup(root / "shared" / "ffmpeg_setup.py")
    if resolved is not None:
        return resolved
    wgp = root / "wgp.py"
    try:
        linked = wgp.is_symlink()
    except OSError:
        linked = False
    if not linked:
        return root
    try:
        target = wgp.resolve().parent
    except OSError:
        return root
    nested = _resolved_ffmpeg_setup(target / "shared" / "ffmpeg_setup.py")
    if nested is not None:
        return nested
    return target


def _resolved_ffmpeg_setup(path: Path) -> Path | None:
    """Parent du clone réel, ou ``None`` si le module n'est pas un fichier."""
    try:
        if not path.is_symlink() and not path.is_file():
            return None
        return path.resolve().parents[1]
    except (OSError, IndexError):
        return None


def _same_directory(left: Path, right: Path) -> bool:
    """``True`` quand les deux chemins désignent le même dossier.

    Un chemin relatif et son absolu, ou un lien de dossier et sa cible,
    restent le même ``ffmpeg_bins``. Deux clones distincts ne le sont pas.
    """
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


def _library_bundle(root: Path) -> Path:
    """``ffmpeg_bins`` que ``download_ffmpeg`` ajoute à ``LD_LIBRARY_PATH``."""
    return _ffmpeg_library_root(root) / "ffmpeg_bins"


def _require_bundle_libs(ffmpeg: Path, root: Path | None) -> None:
    """Wan2GP n'accepte un ffmpeg de ``ffmpeg_bins`` que si ``libavdevice.so*`` y est.

    Sans ces bibliothèques, ``download_ffmpeg`` tente un téléchargement et peut
    remplacer le binaire, puis quitter sans avoir muxé de MP4.
    ``_libs_present`` s'appuie sur ``Path.glob``. Un dossier sans droit de
    lecture (mode 111 ou 311) rend ce glob vide alors que ``lstat`` voit le
    fichier. Le mode 311 est inscriptible : ``open`` crée un nouveau nom,
    le téléchargement peut remplacer le binaire. Le mode 111 refuse ce
    nouveau nom, mais ``shutil.copy2`` ouvre quand même un fichier déjà là
    en ``wb`` et le tronque. Le binaire validé disparaît, les bibliothèques
    nouvelles ne sont pas créées, ``save_video`` imprime ``error saving``
    et le CLI sort 0. Un lien vers un exécutable non inscriptible ne peut
    pas être tronqué : le mux d'essai décide.
    """
    if os.name == "nt" or root is None or not _same_directory(ffmpeg.parent, _library_bundle(root)):
        return
    bundle = _library_bundle(root)
    if _bundle_lists_libavdevice(bundle):
        return
    if _directory_is_listable(bundle):
        _raise_incomplete_bundle(ffmpeg)
    present = _visible_libavdevice(bundle)
    writable = _directory_is_writable(bundle)
    if present is not None and not writable:
        replaceable = _bundle_binary_can_be_replaced(bundle)
        if replaceable is None:
            return
        raise Wan2GPError(
            f"{bundle} n'est pas listable. Le glob libavdevice.so* de download_ffmpeg "
            f"est vide, alors que lstat voit {present}, et le dossier refuse un nouveau nom. "
            f"WanGP tente quand même un téléchargement. shutil.copy2 peut écraser {replaceable} "
            "sans droit d'écriture sur le dossier : ouvrir un fichier déjà présent en wb le tronque. "
            "Les bibliothèques nouvelles ne peuvent pas être créées. Le binaire remplacé ne démarre plus. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            "Rendre ffmpeg_bins listable, ou retirer le droit d'écriture de ce fichier. "
            "Un lien vers un ffmpeg non inscriptible, comme celui du système, n'est pas écrasé."
        )
    if writable:
        seen = (
            f"lstat voit pourtant {present}"
            if present is not None
            else "lstat ne voit pas non plus de libavdevice.so*"
        )
        raise Wan2GPError(
            f"{bundle} n'est pas listable et il est inscriptible. "
            f"Le glob libavdevice.so* de download_ffmpeg est vide ({seen}). "
            "Wan2GP traite le build comme incomplet et tente un téléchargement "
            "qui peut remplacer le binaire déjà dans ce dossier. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            "Rendre ffmpeg_bins listable, ou lui retirer le droit d'écriture."
        )
    _raise_incomplete_bundle(ffmpeg)


def _raise_incomplete_bundle(ffmpeg: Path) -> None:
    raise Wan2GPError(
        f"{ffmpeg} est dans ffmpeg_bins sans libavdevice.so*. "
        "Wan2GP traite ce dossier comme un build incomplet : il tente de le retélécharger "
        "et peut ensuite ne pas muxer de MP4. Retirer ce binaire pour utiliser le ffmpeg du PATH, "
        "ou y déposer le build complet (binaire et bibliothèques)."
    )


def _bundle_lists_libavdevice(bundle: Path) -> bool:
    """``True`` seulement si ``glob`` voit ``libavdevice.so*``.

    Un dossier non listable renvoie une liste vide sans lever d'exception.
    """
    try:
        return any(bundle.glob("libavdevice.so*"))
    except OSError:
        return False


def _directory_is_listable(path: Path) -> bool:
    try:
        next(path.iterdir(), None)
    except OSError:
        return False
    return True


def _directory_is_writable(path: Path) -> bool:
    """``True`` si ``download_ffmpeg`` peut créer un nouveau nom dans ``path``.

    Le mode 311 (écriture et exécution, pas de lecture) laisse ``open``
    réussir et ``os.listdir`` échouer. Le mode 111 refuse un nouveau nom.
    Il peut quand même laisser tronquer un fichier déjà présent : ce cas
    est ``_bundle_binary_can_be_replaced``.
    """
    probe = path / ".wan2gp_write_probe"
    try:
        with open(probe, "wb") as handle:
            handle.write(b"")
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        return True
    return True


def _bundle_binary_can_be_replaced(bundle: Path) -> Path | None:
    """``ffmpeg`` que ``shutil.copy2`` tronquerait sans créer de nom.

    ``open(..., "wb")`` suit les liens. Le droit d'écriture du dossier
    n'intervient pas. ``O_WRONLY`` sans ``O_TRUNC`` vérifie ce droit et
    laisse le fichier intact. Un lien vers le binaire du système échoue,
    et le mux d'essai reste possible.
    """
    candidate = bundle / _binary_name("ffmpeg")
    try:
        if not candidate.is_file():
            return None
    except OSError:
        return None
    try:
        descriptor = os.open(candidate, os.O_WRONLY)
    except OSError:
        return None
    os.close(descriptor)
    return candidate


def _visible_libavdevice(bundle: Path) -> str | None:
    """Nom ``libavdevice.so*`` visible par ``lstat``, sans lister le dossier."""
    for name in _LIBAVDEVICE_PROBE_NAMES:
        candidate = bundle / name
        try:
            candidate.lstat()
        except OSError:
            continue
        return name
    return None


def _is_regular_file(path: Path) -> bool:
    """Fichier régulier, ou ``False`` si ``stat`` échoue.

    Un lien dont le dossier cible est en mode 000 lève ``PermissionError``.
    Le chargeur ignore ce lien et continue. Ce n'est pas une bibliothèque
    qui masque celle du système.
    """
    try:
        return path.is_file()
    except OSError:
        return False


def _is_shared_library(path: Path) -> bool:
    """Nom de bibliothèque dynamique Linux (``libx264.so``, ``libx264.so.164``)."""
    name = path.name
    return name.endswith(".so") or ".so." in name


def _require_ld_library_shadow(ffmpeg: Path, root: Path | None) -> None:
    """Refuse un ``ffmpeg_bins`` qui casse le chargeur au moment du mux.

    ``download_ffmpeg`` préfixe ``LD_LIBRARY_PATH`` avec ce dossier avant de
    savoir quel binaire sera lancé, y compris quand ce binaire est déjà dans
    ``ffmpeg_bins``. Une ``libx264.so*`` ou une ``libav*.so*`` partielle, si
    l'encodeur vient du PATH, fait échouer le mux du système sans code de
    sortie non nul. Un dossier, un tube ou un périphérique du soname NEEDED
    n'est pas un fichier. ``_libs_present`` accepte pourtant un tube
    ``libavdevice.so*`` : le dossier passe pour complet, ``-encoders`` sans
    le préfixe réussit, puis le chargeur se bloque. Le mux d'essai ne doit
    pas ouvrir ce tube. Un lien vers ``/dev/null`` ou ``/dev/zero`` est un
    périphérique. Le soname n'est pas forcément un NEEDED direct : ``libx264.so.164``
    vient de ``libavcodec``, et ``-encoders`` sans préfixe réussit quand même.
    ``save_video`` imprime ``error saving`` et le CLI compte quand même la tâche.
    Le même soname sous ``ffmpeg_bins/glibc-hwcaps/<niveau>/`` est ouvert avant
    celui de la racine. Un fichier directement dans ``glibc-hwcaps``, ou dans
    un sous-dossier qui n'est pas un niveau de ``ld.so --help``, ne l'est pas.
    Retirer le droit de lecture de ce dossier n'y change rien : le chargeur
    n'a pas besoin de le lister, seulement de traverser un chemin exécutable.
    Un lien vers un fichier dont le dossier parent est en mode 000 lève
    ``PermissionError`` sur ``stat``. Le chargeur saute ce lien (le binaire
    démarre). Ce n'est pas une bibliothèque qui masque le système.
    Le dossier préfixé est celui de ``shared/ffmpeg_setup.py`` après
    ``resolve``, pas forcément le ``ffmpeg_bins`` du cwd. Un lien vers un
    autre clone y dépose une bibliothèque que le binaire du cwd ne voit pas
    tant que ``LD_LIBRARY_PATH`` ne commence pas par cet autre dossier.
    """
    if os.name == "nt" or root is None:
        return
    library_root = _ffmpeg_library_root(root)
    bundle = library_root / "ffmpeg_bins"
    try:
        if not bundle.is_dir():
            return
    except OSError:
        return
    entries = _ld_search_entries(bundle, ffmpeg)
    # Un dossier ou un tube dont le nom ne ressemble pas à une bibliothèque
    # n'est pas ouvert par le chargeur. Inutile de parcourir la fermeture.
    needed = (
        _needed_sonames(ffmpeg, bundle)
        if any(_nonfile_library_name(path) for path in entries)
        else set()
    )
    traps = _library_traps(entries, needed, bundle)
    foreign = "" if _same_directory(library_root, root) else _foreign_library_note(library_root, root)
    if traps:
        shown = ", ".join(traps)
        raise Wan2GPError(
            f"{bundle} contient {shown}, soname chargé par {ffmpeg}. "
            + foreign
            + "Ce n'est pas un fichier : le contrôle des bibliothèques *.so ne le voit pas, "
            "et ffmpeg -encoders, lancé sans LD_LIBRARY_PATH, démarre encore. "
            "Le nom peut être un NEEDED direct, ou seulement celui d'une bibliothèque "
            "chargée ensuite (libavcodec demande libx264.so.164, pas le binaire ffmpeg). "
            "Un tube libavdevice.so* satisfait le glob de _libs_present, donc le build "
            "passe pour complet même quand le binaire est déjà dans ffmpeg_bins. "
            "download_ffmpeg préfixe quand même LD_LIBRARY_PATH avec ce dossier. "
            "Le chargeur ouvre ce soname au démarrage : un dossier ou un périphérique "
            "(souvent un lien vers /dev/null) donne cannot read file data, file too short "
            "ou invalid ELF header, un tube bloque l'ouverture. Le mux d'essai ne doit pas l'ouvrir. "
            "glibc ouvre d'abord ffmpeg_bins/glibc-hwcaps/<niveau>/<soname> avant le "
            "fichier du même nom à la racine du dossier, même si ce dossier n'est pas "
            "listable : le droit d'exécution suffit à ouvrir le chemin complet. "
            "Un ffmpeg_bins sans droit de lecture n'est pas listable non plus : "
            "iterdir échoue, download_ffmpeg préfixe quand même LD_LIBRARY_PATH, "
            "et le chargeur ouvre ffmpeg_bins/<soname> sans lister le dossier. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            "Retirer ce dossier, ce tube ou ce périphérique."
        )
    if _same_directory(ffmpeg.parent, bundle):
        return
    shadows = sorted(
        _library_label(bundle, path)
        for path in entries
        if _is_regular_file(path) and _is_shared_library(path)
    )
    if not shadows:
        return
    shown = ", ".join(shadows)
    raise Wan2GPError(
        f"{bundle} contient {shown} alors que l'encodeur retenu est {ffmpeg}. "
        + foreign
        + "Wan2GP ajoute ce dossier en tête de LD_LIBRARY_PATH : ces bibliothèques "
        "peuvent empêcher le mux MP4. save_video imprime error saving et le processus "
        "quitte avec le code 0. "
        "glibc charge aussi ffmpeg_bins/glibc-hwcaps/<niveau>/<soname> avant la racine, "
        "même quand ce dossier n'est pas listable. "
        "Un ffmpeg_bins sans droit de lecture n'est pas listable non plus : "
        "iterdir échoue, et le chargeur ouvre quand même ces sonames. "
        "Retirer ffmpeg_bins pour garder le ffmpeg du PATH, ou y déposer le build "
        "complet (ffmpeg, ffprobe et libavdevice.so*)."
    )


def _foreign_library_note(library_root: Path, root: Path) -> str:
    """Phrase ajoutée quand le préfixe n'est pas le ``ffmpeg_bins`` du cwd."""
    return (
        f"Ce dossier est {library_root}, pas le clone {root}. "
        "download_ffmpeg fait Path(__file__).resolve() sur shared/ffmpeg_setup.py. "
        "Un lien vers un autre clone, ou un wgp.py qui est un lien, préfixe "
        "LD_LIBRARY_PATH avec cet autre ffmpeg_bins. "
        "imageio exécute IMAGEIO_FFMPEG_EXE, qui peut rester le binaire du clone, "
        "avec ces bibliothèques. "
    )


def _ld_search_entries(bundle: Path, ffmpeg: Path) -> list[Path]:
    """Chemins que glibc essaie quand ``LD_LIBRARY_PATH`` commence par ``bundle``.

    L'ordre réel est ``bundle/glibc-hwcaps/<niveau>/<soname>``, puis
    ``bundle/<soname>``. Seuls les niveaux imprimés par ``ld.so --help``
    sont parcourus. Un fichier posé directement dans ``glibc-hwcaps`` n'est
    pas ouvert : il n'est pas listé ici. Un dossier de niveau sans droit de
    lecture n'est pas listable non plus, mais le chargeur ouvre quand même
    ``<niveau>/<soname>`` s'il peut le traverser : ces chemins sont statués
    un par un, sans ``open`` sur un tube. Si ``ffmpeg_bins`` lui-même n'est
    pas listable, ``iterdir`` échoue et ces chemins sont statués de la même
    façon : le chargeur n'a pas besoin de lire le dossier.
    """
    try:
        entries = list(bundle.iterdir())
    except OSError:
        return _probe_unlisted_bundle(bundle, ffmpeg)
    allowed = _glibc_hwcap_names(ffmpeg)
    if not allowed:
        return entries
    hwcaps = bundle / "glibc-hwcaps"
    try:
        if not hwcaps.is_dir():
            return entries
    except OSError:
        return entries
    try:
        children = list(hwcaps.iterdir())
    except OSError:
        entries.extend(_probe_hwcap_sonames(bundle, ffmpeg, sorted(allowed)))
        return entries
    hidden: list[str] = []
    for child in children:
        if child.name not in allowed:
            continue
        try:
            is_directory = child.is_dir()
        except OSError:
            hidden.append(child.name)
            continue
        if not is_directory:
            continue
        try:
            entries.extend(child.iterdir())
        except OSError:
            hidden.append(child.name)
    if hidden:
        entries.extend(_probe_hwcap_sonames(bundle, ffmpeg, hidden))
    return entries


def _probe_unlisted_bundle(bundle: Path, ffmpeg: Path) -> list[Path]:
    """Sonames connus quand ``ffmpeg_bins`` lui-même n'est pas listable.

    ``download_ffmpeg`` préfixe ``LD_LIBRARY_PATH`` sans lister le dossier.
    Le chargeur ouvre d'abord ``glibc-hwcaps/<niveau>/<soname>``, puis
    ``<soname>`` à la racine, dès que le dossier est exécutable. ``lstat``
    ne bloque pas sur un tube.
    Un dossier en mode ``000`` ne se traverse pas : ``lstat`` échoue comme
    ``open``, et ces chemins ne sont pas signalés.
    """
    sonames = _needed_sonames(ffmpeg, bundle)
    if sonames is None:
        raise Wan2GPError(
            f"{bundle} n'est pas listable, et les sonames de {ffmpeg} sont illisibles. "
            "download_ffmpeg préfixe quand même LD_LIBRARY_PATH avec ce dossier. "
            "Le chargeur ouvre ffmpeg_bins/<soname> et "
            "ffmpeg_bins/glibc-hwcaps/<niveau>/<soname> sans lister le dossier. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            "Rendre ffmpeg_bins listable, ou le retirer."
        )
    if not sonames:
        return []
    found: list[Path] = []
    for name in sorted(sonames):
        candidate = bundle / name
        try:
            candidate.lstat()
        except OSError:
            continue
        found.append(candidate)
    levels = _glibc_hwcap_names(ffmpeg)
    if levels:
        found.extend(_probe_hwcap_sonames(bundle, ffmpeg, sorted(levels)))
    return found


def _probe_hwcap_sonames(bundle: Path, ffmpeg: Path, levels: list[str]) -> list[Path]:
    """Sonames connus sous des niveaux ``glibc-hwcaps`` qu'on ne peut pas lister.

    ``lstat`` ne bloque pas sur un tube. Un nom absent est ignoré. Si la
    fermeture des sonames est illisible, le dossier est refusé : le chargeur
    peut encore ouvrir un fichier qu'on ne sait pas nommer.
    """
    sonames = _needed_sonames(ffmpeg, bundle)
    if sonames is None:
        raise Wan2GPError(
            f"{bundle / 'glibc-hwcaps'} n'est pas listable, et les sonames de {ffmpeg} "
            "sont illisibles. Le chargeur ouvre pourtant "
            "ffmpeg_bins/glibc-hwcaps/<niveau>/<soname> dès que le dossier est exécutable. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            "Rendre ce dossier listable, ou le retirer."
        )
    if not sonames:
        return []
    found: list[Path] = []
    hwcaps = bundle / "glibc-hwcaps"
    for level in levels:
        slot = hwcaps / level
        for name in sonames:
            candidate = slot / name
            try:
                candidate.lstat()
            except OSError:
                continue
            found.append(candidate)
    return found


def _glibc_hwcap_names(ffmpeg: Path) -> frozenset[str]:
    """Niveaux ``glibc-hwcaps`` que le chargeur de ce binaire parcourt.

    ``ld.so --help`` les liste sans charger les bibliothèques de
    ``ffmpeg_bins``. Un tube de soname ne peut donc pas bloquer cette lecture.
    """
    global _GLIBC_HWCAPS
    if _GLIBC_HWCAPS is not None:
        return _GLIBC_HWCAPS
    interpreter = _interpreter_path(ffmpeg)
    if interpreter is None:
        return frozenset()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["LANGUAGE"] = "C"
    env.pop("LD_PRELOAD", None)
    try:
        completed = subprocess.run(
            [interpreter, "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if completed.returncode != 0:
        return frozenset()
    names: set[str] = set()
    capture = False
    for line in completed.stdout.splitlines():
        if not capture:
            if line.strip() == _HWCAP_HEADER:
                capture = True
            continue
        if not line.startswith((" ", "\t")):
            break
        token = line.strip().split(" ", 1)[0]
        if token:
            names.add(token)
    if not names:
        return frozenset()
    _GLIBC_HWCAPS = frozenset(names)
    return _GLIBC_HWCAPS


def _library_label(bundle: Path, path: Path) -> str:
    """Nom du soname, ou chemin relatif s'il est sous ``glibc-hwcaps``."""
    try:
        relative = path.relative_to(bundle)
    except ValueError:
        return path.name
    if relative.parent == Path("."):
        return path.name
    return relative.as_posix()


def _nonfile_library_name(path: Path) -> bool:
    """Nom de bibliothèque qui n'est pas un fichier.

    ``stat`` ne bloque pas sur un tube. Seul un ``open`` le ferait.
    """
    try:
        if path.is_file():
            return False
    except OSError:
        return _is_shared_library(path)
    return _is_shared_library(path)


def _needed_sonames(ffmpeg: Path, bundle: Path | None = None) -> set[str] | None:
    """Sonames que le chargeur ouvre si ``LD_LIBRARY_PATH`` commence par ``ffmpeg_bins``.

    ``readelf -d`` ne charge pas les bibliothèques : un tube du bon soname
    dans ``ffmpeg_bins`` ne peut pas le bloquer. La fermeture inclut les
    NEEDED des bibliothèques elles-mêmes. ``libx264.so.164`` n'apparaît pas
    dans le binaire ffmpeg, seulement dans ``libavcodec``. L'interpréteur
    (``ld-linux``) est retiré : un dossier de ce nom ne casse pas le démarrage.
    Un binaire statique n'a pas de NEEDED. ``None`` si la liste est illisible.
    """
    readelf = shutil.which("readelf")
    if readelf is None:
        return _without_interpreter(_ldd_sonames(ffmpeg), ffmpeg)
    direct = _readelf_needed(ffmpeg, readelf)
    if direct is None:
        return _without_interpreter(_ldd_sonames(ffmpeg), ffmpeg)
    found: set[str] = set()
    pending = list(direct)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        resolved = _resolve_loaded_library(name, bundle)
        nested = _readelf_needed(resolved, readelf) if resolved is not None else None
        if nested is None and resolved is not None and bundle is not None and resolved.parent == bundle:
            fallback = _ldconfig_sonames().get(name)
            if fallback is not None and fallback != resolved:
                nested = _readelf_needed(fallback, readelf)
        if not nested:
            continue
        for item in nested:
            if item not in found:
                pending.append(item)
    return _without_interpreter(found, ffmpeg)


def _without_interpreter(names: set[str] | None, binary: Path) -> set[str] | None:
    if names is None:
        return None
    interpreter = _interpreter_basename(binary)
    if interpreter is not None:
        names.discard(interpreter)
    return names


def _interpreter_path(binary: Path) -> str | None:
    """Chemin PT_INTERP. ``None`` si le binaire n'est pas dynamique.

    ``readelf -l`` est la source habituelle. S'il manque, ou s'il ne donne
    pas de chemin, l'en-tête ELF est lu directement. Sans ce chemin, les
    niveaux ``glibc-hwcaps`` restent inconnus : un soname dans
    ``ffmpeg_bins/glibc-hwcaps/<niveau>/`` échappe au contrôle, ``save_video``
    imprime ``error saving`` et le CLI sort avec le code 0.
    """
    readelf = shutil.which("readelf")
    if readelf is not None:
        try:
            completed = subprocess.run(
                [readelf, "-l", str(binary)],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            completed = None
        if completed is not None and completed.returncode == 0:
            match = _INTERPRETER_RE.search(completed.stdout)
            if match is not None:
                path = match.group(1).rstrip("]")
                if path:
                    return path
    return _elf_interpreter(binary)


def _elf_interpreter(binary: Path) -> str | None:
    """PT_INTERP lu dans le binaire, sans ``readelf`` et sans le chargeur.

    Seuls l'en-tête ELF et le segment d'interpréteur sont ouverts. Un tube
    qui porte un soname dans ``ffmpeg_bins`` ne peut pas bloquer cette lecture.
    """
    try:
        with binary.open("rb") as handle:
            ident = handle.read(16)
            if len(ident) != 16 or ident[:4] != b"\x7fELF":
                return None
            elf_class = ident[4]
            elf_data = ident[5]
            if elf_data == 1:
                endian = "<"
            elif elf_data == 2:
                endian = ">"
            else:
                return None
            if elf_class == 2:
                handle.seek(32)
                phoff_raw = handle.read(8)
                if len(phoff_raw) != 8:
                    return None
                e_phoff = struct.unpack(endian + "Q", phoff_raw)[0]
                handle.seek(54)
                ph_meta = handle.read(4)
                if len(ph_meta) != 4:
                    return None
                e_phentsize, e_phnum = struct.unpack(endian + "HH", ph_meta)
                program_header = endian + "IIQQQQQQ"
            elif elf_class == 1:
                handle.seek(28)
                phoff_raw = handle.read(4)
                if len(phoff_raw) != 4:
                    return None
                e_phoff = struct.unpack(endian + "I", phoff_raw)[0]
                handle.seek(42)
                ph_meta = handle.read(4)
                if len(ph_meta) != 4:
                    return None
                e_phentsize, e_phnum = struct.unpack(endian + "HH", ph_meta)
                program_header = endian + "IIIIIIII"
            else:
                return None
            header_size = struct.calcsize(program_header)
            if e_phnum == 0 or e_phnum > 128 or e_phentsize < header_size or e_phoff > 1_048_576:
                return None
            handle.seek(e_phoff)
            for _ in range(e_phnum):
                raw = handle.read(e_phentsize)
                if len(raw) < header_size:
                    return None
                fields = struct.unpack_from(program_header, raw)
                if elf_class == 2:
                    p_type, _flags, p_offset, _vaddr, _paddr, p_filesz, _memsz, _align = fields
                else:
                    p_type, p_offset, _vaddr, _paddr, p_filesz, _memsz, _flags, _align = fields
                if p_type != 3:
                    continue
                if p_filesz == 0 or p_filesz > 4096:
                    return None
                handle.seek(p_offset)
                text = handle.read(p_filesz).split(b"\x00", 1)[0]
                try:
                    decoded = text.decode("utf-8")
                except UnicodeDecodeError:
                    return None
                return decoded or None
    except OSError:
        return None
    return None


def _interpreter_basename(binary: Path) -> str | None:
    path = _interpreter_path(binary)
    if path is None:
        return None
    name = Path(path).name
    return name or None


def _readelf_needed(path: Path, readelf: str) -> frozenset[str] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _READELF_NEEDED:
        return _READELF_NEEDED[key]
    try:
        completed = subprocess.run(
            [readelf, "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        _READELF_NEEDED[key] = None
        return None
    if completed.returncode != 0:
        _READELF_NEEDED[key] = None
        return None
    found = frozenset(_NEEDED_LIBRARY_RE.findall(completed.stdout))
    _READELF_NEEDED[key] = found
    return found


def _ldconfig_sonames() -> dict[str, Path]:
    """Premier fichier réel de ``ldconfig -p`` pour chaque soname.

    Cette table ne charge pas ffmpeg et n'ouvre pas les tubes de ``ffmpeg_bins``.
    """
    global _LDCONFIG_SONAMES
    if _LDCONFIG_SONAMES is not None:
        return _LDCONFIG_SONAMES
    mapping: dict[str, Path] = {}
    ldconfig = shutil.which("ldconfig")
    if ldconfig is None:
        _LDCONFIG_SONAMES = mapping
        return mapping
    try:
        completed = subprocess.run(
            [ldconfig, "-p"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        _LDCONFIG_SONAMES = mapping
        return mapping
    if completed.returncode != 0:
        _LDCONFIG_SONAMES = mapping
        return mapping
    for line in completed.stdout.splitlines():
        if "=>" not in line:
            continue
        left, _, right = line.partition("=>")
        name = left.strip().split(" ", 1)[0]
        if not name or name in mapping:
            continue
        candidate = Path(right.strip())
        try:
            if candidate.is_file():
                mapping[name] = candidate
        except OSError:
            continue
    _LDCONFIG_SONAMES = mapping
    return mapping


def _resolve_loaded_library(name: str, bundle: Path | None) -> Path | None:
    """Fichier ELF que le chargeur lirait pour ce soname, jamais un tube.

    Un fichier régulier dans ``ffmpeg_bins`` masque celui du système, comme
    ``LD_LIBRARY_PATH``. Un dossier ou un tube n'est pas lu ici : il est
    refusé à part, et ses NEEDED viennent du fichier système.
    """
    if bundle is not None:
        candidate = bundle / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            pass
    return _ldconfig_sonames().get(name)


def _ldd_sonames(ffmpeg: Path) -> set[str] | None:
    """Repli si ``readelf`` ne lit pas le binaire. ``None`` si ``ldd`` bloque."""
    ldd = shutil.which("ldd")
    if ldd is None:
        return None
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
    try:
        completed = subprocess.run(
            [ldd, str(ffmpeg)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = f"{completed.stdout}\n{completed.stderr}"
    if "not a dynamic executable" in text:
        return set()
    found = {
        line.strip().split(" ", 1)[0]
        for line in completed.stdout.splitlines()
        if ".so" in line and "=>" in line
    }
    if not found:
        return None
    return found


def _library_traps(entries: list[Path], needed: set[str] | None, bundle: Path) -> list[str]:
    """Chemins que le chargeur ouvrirait à la place d'une bibliothèque.

    Un lien cassé ou un socket de ce nom laisse le chargeur passer au
    suivant. Un dossier, un tube, ou un périphérique (lien vers ``/dev/null``)
    est ouvert et fait échouer ffmpeg. La liste ``needed`` couvre les
    sonames directs et ceux demandés par les bibliothèques déjà chargées.
    Le contrôle vaut aussi quand le binaire est dans ``ffmpeg_bins`` : un
    tube du soname y bloque le mux d'essai, et ``libavdevice.so*`` suffit
    au glob de présence. Un soname sous ``glibc-hwcaps/<niveau>/`` est
    ouvert avant celui de la racine.
    """
    traps: list[str] = []
    for path in sorted(entries, key=lambda item: _library_label(bundle, item)):
        try:
            if path.is_file():
                continue
        except OSError:
            continue
        if needed is None:
            if not _is_shared_library(path):
                continue
        elif path.name not in needed:
            continue
        kind = _path_kind(path)
        if kind not in {"dossier", "tube", "périphérique"}:
            continue
        traps.append(f"{kind} {_library_label(bundle, path)}")
    return traps


def _command_env(root: Path | None) -> dict[str, str]:
    """Environnement d'un probe : le PATH de WanGP, pas celui du pipeline.

    ``LD_LIBRARY_PATH`` n'est pas modifié. ``-encoders`` doit encore voir les
    bibliothèques du système ; le préfixe de dossiers est contrôlé à part.
    """
    env = os.environ.copy()
    if os.name == "nt" or root is None:
        return env
    env["PATH"] = _wan_path(root, env.get("PATH", ""))
    return env


def _wan_path(root: Path, current: str) -> str:
    """PATH après ``_ensure_bin_dir_on_path`` : ``ffmpeg_bins``, puis la racine.

    Les entrées sont absolues et ne suivent pas les liens. ``ffmpeg_bins/ffmpeg``
    peut rester un lien vers le binaire du système : c'est ce nom que WanGP
    met en tête, pas ``/usr/bin``.
    """
    directories = [
        os.path.normpath(str((root / "ffmpeg_bins").absolute())),
        os.path.normpath(str(root.absolute())),
    ]
    prioritized: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        if directory in seen:
            continue
        prioritized.append(directory)
        seen.add(directory)
    kept: list[str] = []
    for part in current.split(os.pathsep):
        if not part:
            continue
        normalized = os.path.normpath(part)
        if normalized in seen:
            continue
        kept.append(part)
        seen.add(normalized)
    return os.pathsep.join([*prioritized, *kept])


def _path_reexec_message(binary: Path, root: Path | None) -> str:
    if root is None:
        where = "ffmpeg_bins puis la racine du clone"
    else:
        bundle = os.path.normpath(str((root / "ffmpeg_bins").absolute()))
        repo = os.path.normpath(str(root.absolute()))
        where = f"{bundle} puis {repo}"
    return (
        f"{binary} ne revient pas quand le PATH commence par {where}. "
        "download_ffmpeg place ces dossiers devant le PATH avant le mux. "
        "Un script qui relance ffmpeg ou ffprobe sans chemin (exec ffmpeg) "
        "retrouve alors ce même fichier. "
        "ffmpeg -encoders, lancé avec le PATH d'origine, voit le binaire du système "
        "et peut encore annoncer libx264. "
        "imageio exécute IMAGEIO_FFMPEG_EXE : le script boucle, ou la chaîne de "
        "processus casse. save_video imprime error saving et le processus quitte "
        "avec le code 0, sans MP4. "
        "Appeler le binaire par un chemin absolu, ou remplacer le script."
    )


def _libstdcxx_preload_directory(python: str, root: Path | None = None) -> Path | None:
    """Dossier que ``preload_preferred_libstdcxx`` va préfixer, ou ``None``.

    Le contrôle tourne dans l'interpréteur qui lancera ``wgp.py``, avec le
    cwd du clone. ``os.path.isfile`` et ``CDLL`` y résolvent un
    ``CONDA_PREFIX`` relatif. tqdm est importé avant la lecture de
    ``/proc/self/maps``, comme au démarrage de WanGP. Un ``libstdc++.so.6``
    déjà chargé, ou un fichier que ``CDLL`` refuse, ne change pas
    ``LD_LIBRARY_PATH``.
    """
    if os.name == "nt":
        return None
    try:
        completed = subprocess.run(
            [python, "-c", _LIBSTDCXX_PROBE],
            check=False,
            capture_output=True,
            text=True,
            cwd=_probe_cwd(root),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(
            f"{python} ne revient pas en cherchant {_LIBSTDCXX_NAME}. "
            "preload_preferred_libstdcxx fait le même contrôle avant le mux. "
            "save_video imprime error saving et le processus quitte avec le code 0."
        ) from exc
    except OSError as exc:
        raise Wan2GPError(
            f"Impossible de lancer {python} pour le contrôle {_LIBSTDCXX_NAME} : {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise Wan2GPError(
            f"{python} n'a pas pu reproduire preload_preferred_libstdcxx "
            f"({_LIBSTDCXX_NAME}). {detail}".rstrip()
        )
    directory = ""
    for line in completed.stdout.splitlines():
        if line.startswith(_LIBSTDCXX_MARKER):
            directory = line[len(_LIBSTDCXX_MARKER) :].strip()
            break
    if not directory:
        return None
    return Path(directory)


def _preload_masked_sonames(ffmpeg: Path, root: Path | None, preload: Path) -> set[str]:
    """Sonames déjà fournis par ``ffmpeg_bins``, donc ouverts avant le préchargement.

    ``download_ffmpeg`` préfixe ``ffmpeg_bins`` après ``preload_preferred_libstdcxx``.
    Un fichier régulier de ce soname dans le premier dossier masque un tube du
    même nom dans le dossier de ``libstdc++``.
    """
    if root is None or _same_directory(preload, _library_bundle(root)):
        return set()
    bundle = _library_bundle(root)
    try:
        if not bundle.is_dir():
            return set()
    except OSError:
        return set()
    return {
        path.name
        for path in _ld_search_entries(bundle, ffmpeg)
        if _is_regular_file(path)
    }


def _from_process_cwd(path: Path, root: Path | None) -> Path:
    """Chemin ouvert par le chargeur quand le cwd du processus est le clone.

    ``preload_preferred_libstdcxx`` copie ``os.path.dirname`` dans
    ``LD_LIBRARY_PATH`` sans le rendre absolu. Un ``CONDA_PREFIX`` relatif
    est donc résolu depuis le clone, pas depuis le dossier du pipeline.
    """
    if path.is_absolute() or root is None:
        return path
    return root / path


def _require_preload_library_traps(
    ffmpeg: Path,
    root: Path | None,
    preload: Path | None,
) -> None:
    """Refuse un soname non fichier dans le dossier préfixé pour ``libstdc++``.

    Un fichier trop court est laissé au mux d'essai : l'ouvrir avec ffmpeg
    échoue tout de suite. Un tube, lui, bloquerait ce mux. ``lstat`` suffit.
    Le dossier inspecté est celui que le clone verra : un préfixe relatif
    est joint au cwd du processus, pas lu depuis le dossier du pipeline.
    """
    if os.name == "nt" or preload is None:
        return
    seen = _from_process_cwd(preload, root)
    if root is not None and _same_directory(seen, _library_bundle(root)):
        return
    try:
        if not seen.is_dir():
            return
    except OSError:
        return
    masked = _preload_masked_sonames(ffmpeg, root, seen)
    entries = _ld_search_entries(seen, ffmpeg)
    needed = (
        _needed_sonames(ffmpeg, seen)
        if any(_nonfile_library_name(path) for path in entries)
        else set()
    )
    traps = [
        trap
        for trap in _library_traps(entries, needed, seen)
        if Path(trap.split(" ", 1)[1]).name not in masked
    ]
    if not traps:
        return
    shown = ", ".join(traps)
    located = str(preload)
    if seen != preload:
        located = f"{preload} (ouvert depuis le cwd du clone : {seen})"
    raise Wan2GPError(
        f"{located} contient {shown}, soname chargé par {ffmpeg}. "
        "preload_preferred_libstdcxx (shared/native_runtime.py) ajoute ce dossier "
        f"à LD_LIBRARY_PATH quand {_LIBSTDCXX_NAME} n'est pas encore chargé. "
        "Le fichier est cherché dans sys.prefix/lib, sys.exec_prefix/lib, puis "
        "CONDA_PREFIX/lib. download_ffmpeg préfixe ensuite ffmpeg_bins : ce dossier "
        "reste sur le chemin du chargeur, après ffmpeg_bins. "
        "Ce n'est pas un fichier. ffmpeg -encoders, lancé sans LD_LIBRARY_PATH, "
        "démarre encore. Un dossier ou un périphérique (souvent un lien vers /dev/null) "
        "donne file too short ou invalid ELF header. Un tube bloque l'ouverture, "
        "y compris sous glibc-hwcaps/<niveau>/. Le mux d'essai ne doit pas l'ouvrir. "
        "Un CONDA_PREFIX relatif n'est pas rendu absolu : le processus a pour cwd "
        "le clone, donc ce n'est pas le dossier vu depuis le pipeline. "
        "save_video imprime error saving et le processus quitte avec le code 0. "
        "Retirer ce dossier, ce tube ou ce périphérique du répertoire de libstdc++. "
        "Un fichier régulier du même soname déjà présent dans ffmpeg_bins est ouvert avant."
    )


def _runtime_env(root: Path | None, preload: Path | None = None) -> dict[str, str]:
    """Environnement du mux : PATH et ``LD_LIBRARY_PATH`` que WanGP va poser.

    ``preload_preferred_libstdcxx`` préfixe d'abord le dossier de
    ``libstdc++.so.6``, même relatif. ``download_ffmpeg`` préfixe ensuite
    ``ffmpeg_bins``, y compris quand le binaire est déjà dans ce dossier.
    Le PATH reçoit ``ffmpeg_bins`` puis la racine du clone, sans suivre un
    lien ``ffmpeg``. Le cwd du mux doit être le clone pour qu'un préfixe
    relatif désigne le même dossier que dans le processus.
    """
    env = _command_env(root)
    if os.name == "nt":
        return env
    parts = [part for part in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    changed = False
    if preload is not None:
        directory = str(preload)
        if directory not in parts:
            parts = [directory, *parts]
            changed = True
    if root is not None:
        bundle = _library_bundle(root)
        try:
            bundle_ready = bundle.is_dir()
        except OSError:
            bundle_ready = False
        if bundle_ready:
            directory = str(bundle)
            if directory not in parts:
                parts = [directory, *parts]
                changed = True
    if changed:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def _require_bundled_mux(
    ffmpeg: Path,
    ffprobe: Path,
    root: Path | None,
    encoder: str,
    preload: Path | None = None,
) -> None:
    """Muxe deux images avec le ``LD_LIBRARY_PATH`` que WanGP va réellement poser.

    ``_require_encoder`` lance ``-encoders`` sans ce préfixe. Un lien vers le
    ffmpeg du système, posé dans ``ffmpeg_bins`` à côté d'une ``libavdevice.so*``
    qui ne porte pas le soname attendu, passe ce contrôle. La bibliothèque
    ``libavcodec.so.<soname>`` du même dossier, vide, est celle que le chargeur
    retient dès que ``LD_LIBRARY_PATH`` commence par ``ffmpeg_bins``. Le
    processus n'atteint pas imageio : ``save_video`` imprime ``error saving``
    et ``process_tasks_cli`` sort avec le code 0. Un tube du soname NEEDED
    est refusé avant cet appel : l'ouvrir bloquerait le chargeur.
    Le même préfixe inclut le dossier de ``libstdc++.so.6`` quand
    ``preload_preferred_libstdcxx`` va l'ajouter. Le mux tourne alors même
    si l'encodeur vient du PATH : une bibliothèque trop courte dans ce
    dossier masque celle du système, et ``-encoders`` sans préfixe reste vert.
    Le cwd est celui du processus WanGP. Un ``CONDA_PREFIX`` relatif est
    ouvert sous le clone : le même nom, depuis le dossier du pipeline, peut
    être un préfixe sain et laisser passer ``-encoders``.
    """
    in_bundle = root is not None and _same_directory(ffmpeg.parent, _library_bundle(root))
    if os.name == "nt" or (preload is None and not in_bundle):
        return
    env = _runtime_env(root, preload)
    run_cwd = None if root is None else str(root.absolute())
    ffmpeg_bin = _absolute_binary(ffmpeg)
    ffprobe_bin = _absolute_binary(ffprobe)
    width = height = 64
    frame = bytes(width * height * 3)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "probe.mp4"
        try:
            completed = subprocess.run(
                [
                    str(ffmpeg_bin),
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    "8",
                    "-i",
                    "pipe:0",
                    "-frames:v",
                    "2",
                    "-c:v",
                    encoder,
                    "-pix_fmt",
                    "yuv420p",
                    str(output),
                ],
                input=frame + frame,
                check=False,
                capture_output=True,
                env=env,
                cwd=run_cwd,
                timeout=_MUX_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise Wan2GPError(
                f"{ffmpeg_bin} ne revient pas quand LD_LIBRARY_PATH commence par {ffmpeg.parent}. "
                "Un tube qui porte un soname NEEDED bloque le chargeur. "
                "Le PATH aussi commence par ffmpeg_bins : un script exec ffmpeg se relance. "
                "Le cwd du mux est le clone : un préfixe relatif (CONDA_PREFIX) y est ouvert. "
                "save_video n'écrit pas de MP4. Retirer ce tube, ou appeler le binaire par un chemin absolu."
            ) from exc
        except OSError as exc:
            raise Wan2GPError(f"Impossible de muxer un MP4 avec {ffmpeg} : {exc}") from exc
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 8:
            libraries = env.get("LD_LIBRARY_PATH", "")
            if preload is None:
                reason = (
                    f"{ffmpeg} ne produit pas de MP4 quand LD_LIBRARY_PATH commence par {ffmpeg.parent}. "
                    "download_ffmpeg ajoute ce dossier même si le binaire y est déjà. "
                    "Une bibliothèque du bon soname (libavcodec.so*, libavdevice.so*), vide ou trop courte, "
                    "masque alors celle du système. "
                )
            else:
                relative = ""
                if preload is not None and not preload.is_absolute() and root is not None:
                    relative = (
                        f"Ce préfixe est relatif et le cwd du processus est {root.absolute()} : "
                        "le chargeur ouvre ce dossier sous le clone, pas depuis le dossier du pipeline. "
                    )
                reason = (
                    f"{ffmpeg_bin} ne produit pas de MP4 avec LD_LIBRARY_PATH={libraries}. "
                    "preload_preferred_libstdcxx (shared/native_runtime.py) ajoute le dossier de "
                    f"{_LIBSTDCXX_NAME} ({preload}) quand cette bibliothèque n'est pas encore chargée. "
                    "Le fichier est pris dans sys.prefix/lib, sys.exec_prefix/lib ou CONDA_PREFIX/lib. "
                    "download_ffmpeg préfixe ensuite ffmpeg_bins, donc ce dossier reste visible du chargeur. "
                    "Une bibliothèque du soname demandé par ffmpeg, vide ou trop courte, masque celle du système. "
                    + relative
                )
            raise Wan2GPError(
                reason
                + "-encoders, lancé sans ce préfixe, peut encore annoncer "
                + f"{encoder}. save_video imprime error saving et le processus quitte avec le code 0. "
                + detail
            )
        try:
            probe = subprocess.run(
                [
                    str(ffprobe_bin),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_type",
                    "-of",
                    "csv=p=0",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=run_cwd,
                timeout=_MUX_PROBE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise Wan2GPError(
                f"{ffprobe} ne revient pas avec LD_LIBRARY_PATH={ffmpeg.parent}. "
                "Un tube qui porte un soname NEEDED bloque le chargeur. "
                "Le PATH aussi commence par ffmpeg_bins : un script exec ffprobe se relance. "
                "save_video n'écrit pas de MP4. Retirer ce tube, ou appeler le binaire par un chemin absolu."
            ) from exc
        except OSError as exc:
            raise Wan2GPError(f"Impossible de lire le MP4 d'essai avec {ffprobe} : {exc}") from exc
    if probe.returncode != 0 or "video" not in probe.stdout:
        probe_detail = (probe.stderr or probe.stdout).strip()
        raise Wan2GPError(
            f"{ffprobe} ne lit pas le MP4 muxé par {ffmpeg} avec LD_LIBRARY_PATH={ffmpeg.parent}. "
            "save_video imprime error saving et le processus quitte avec le code 0. "
            f"{probe_detail}".rstrip()
        )


def _require_ffprobe(ffprobe: Path, root: Path | None) -> None:
    binary = _probe_binary(ffprobe, root)
    try:
        probe = subprocess.run(
            [str(binary), "-version"],
            check=False,
            capture_output=True,
            text=True,
            env=_command_env(root),
            cwd=_probe_cwd(root),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(_path_reexec_message(ffprobe, root)) from exc
    except OSError as exc:
        raise Wan2GPError(f"Impossible de lancer {ffprobe} : {exc}") from exc
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip()
        relative = _relative_ld_failure_message(binary, root, probe.returncode, detail)
        if relative is not None:
            raise Wan2GPError(relative)
        raise Wan2GPError(f"ffprobe ne démarre pas ({ffprobe}). {detail}".strip())


def ffmpeg_env(binaries: FfmpegBinaries, root: Path | None = None) -> dict[str, str]:
    """Variables lues par WanGP, imageio et les outils qui cherchent ffmpeg hors PATH.

    ``IMAGEIO_FFMPEG_EXE`` est utilisé tel quel, sans recherche dans le PATH.
    Le chemin doit être absolu : le cwd du processus est le clone WanGP.
    Le PATH du processus reprend celui de ``download_ffmpeg`` : ``ffmpeg_bins``,
    puis la racine du clone, sans remplacer un lien par ``/usr/bin``.
    """
    env = os.environ.copy()
    env["FFMPEG_BINARY"] = binaries.ffmpeg
    env["FFPROBE_BINARY"] = binaries.ffprobe
    env["IMAGEIO_FFMPEG_EXE"] = binaries.ffmpeg
    if root is not None and os.name != "nt":
        env["PATH"] = _wan_path(root, env.get("PATH", ""))
    else:
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
            cwd=str(root.absolute()),
            env=ffmpeg_env(binaries, root),
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
