"""Lance Wan2GP en headless et exige un MP4 en sortie.

La commande documentée pour vider une file sans interface est :

    python wgp.py --process <file.zip> --output-dir <dossier> --verbose 2

Le zip contient ``queue.json``. Le modèle est ``t2v_1.3B`` en ``image_mode`` 0.
``Fun_InP_1.3B`` (I2V) est refusé : cette famille peut écrire un JPEG et
quitter avec le code 0, sans vidéo. Une définition ``defaults``,
``finetunes`` ou le ``defaults`` d'un plugin activé, pour le même modèle,
avec ``image_outputs``, a le même effet. ``init_model_def`` n'est appelé
que sur le premier JSON : ``query_model_def("chrono_edit")`` pose le
drapeau même si la clé est absente, et le ``defaults`` officiel de
``t2v_1.3B`` ne l'efface pas. L'architecture finale, elle, est celle du
dernier JSON : ``validate_generative_settings`` force ``image_mode`` à 1
pour ``chrono_edit`` tant que ``model_mode`` n'est pas 2, même si
``image_outputs`` est faux. ``fix_settings`` de ``qwen_image_21_7B``
fait la même chose pour toute file en ``image_mode`` 0, que la clé
``image_outputs`` soit absente ou fausse. Ce même premier
``query_model_def`` injecte ``model_modes.image_modes`` ``[2]`` pour
``qwen_image_21_7B``, ``qwen_image_20B``, les Qwen edit, Flux lorsque
``supports_inpaint`` est vrai (pas ``pi_flux2``) et Krea 2. Un JSON
suivant qui passe à ``chrono_edit``, met ``image_outputs`` à faux et
``model_mode`` à 2, sans réécrire ``model_modes``, laisse cette liste :
``0`` n'y est pas, ``model_mode`` devient None, Chrono Edit écrit un
JPEG et le CLI sort 0. Pour une architecture I2V finale
(``fun_inp_1.3B``, ``chrono_edit``, …), ``update_default_settings`` pose
aussi ``image_prompt_type`` à ``S`` quand ``image_prompt_types_allowed``
contient ``S``. Cette chaîne est injectée par ``query_model_def`` du
premier JSON si la clé manque (``SEVL`` pour Fun InP et Chrono). Sans
image de départ, et sans ``black_frame``, ``validate_settings`` renvoie
une erreur : la tâche est sautée, ``process_tasks_cli`` sort 0, sans MP4.
Le même ``update_default_settings`` pose ``video_prompt_type`` ou
``audio_prompt_type`` pour d'autres architectures (``phantom_1.3B``,
``standin``, ``recam_1.3B``, ``fantasy``, …). ``fix_settings`` ne retire
``I`` et ``K`` que si ``guide_custom_choices`` est absent et que
``image_ref_choices`` est vide. ``one_image_ref_needed``, injecté par
le premier ``query_model_def`` pour StandIn et Lynx, exige une image
même après ce filtre. Sans le média, la tâche est sautée et le CLI
sort 0, sans MP4. ``query_model_def("wanmove")`` pose aussi un
``custom_guide`` obligatoire (fichier ``.npy``). ``init_model_def``
n'est appelé que pour ce premier JSON : un finetune qui repasse
l'architecture à ``t2v_1.3B`` ne retire pas le guide. ``validate_settings``
l'exige avant l'image de départ. La tâche est sautée et le CLI sort 0,
sans MP4. ``models/wan/ovi_handler.py`` pose ``sliding_window`` et
``sliding_window_defaults.overlap_max`` à 1. ``models/_settings.json``
laisse ``sliding_window_overlap`` à 5, et ``update_default_settings``
d'Ovi ne le rabaisse pas. ``extra_settings.validate_inputs`` refuse
cette valeur. La tâche est sautée et le CLI sort 0, sans MP4.
``init_model_def`` n'est appelé que pour le premier JSON : un finetune
qui change l'architecture ne retire pas ce plafond. Le même validateur
refuse ``sliding_window_size`` avant le recouvrement. MiniMax H3
(``minimax_h3_ref2va`` et la famille) pose ``window_min`` à 124.
``ti2v_2_2`` laisse la taille à 121 : la tâche est sautée et le CLI
sort 0, sans MP4. Un JSON qui écrit ``window_min`` ou ``window_max``
hors de la taille d'usine fait pareil. Le même validateur contrôle
ensuite ``sub_parallel_window_size`` puis ``sub_parallel_window_overlap``
si ``sub_parallel_windows`` est vrai. La taille d'usine est 0 : le
minimum est fixe, pas ``window_min``, donc 0 passe. Une valeur 300 avec
le ``window_max`` d'usine 257 est refusée avant le recouvrement, même
quand celui-ci tient dans la borne (8). ``sliding_window_size`` 129
reste sous 257 : le contrôle de taille de fenêtre ne voit pas ce saut.
La tâche est sautée et le CLI sort 0, sans MP4.
``models/_settings.json`` laisse ensuite le recouvrement à 17. Un
``overlap_max`` de 8 laisse passer le recouvrement 5, puis refuse 17.
``query_model_def`` de ``scail2`` pose le drapeau sans resserrer
``window_max`` (257) ni ``overlap_max`` (97) : la taille 0 et le
recouvrement 17 passent. Le même validateur contrôle ensuite
``sliding_window_color_correction_strength`` si ``color_correction``
est vrai. Le défaut d'usine est 0 (minimum 0, maximum 1) : il passe.
``query_model_def`` de Wan pose ce drapeau à vrai pour le premier JSON.
``animate2`` le remet à faux dans le même appel. Une valeur 2 ou 1.01
est refusée même si ``sub_parallel_window_size`` vaut 0 et que le
recouvrement parallèle tient dans la borne. La tâche est sautée et le
CLI sort 0, sans MP4. Un ``color_correction`` faux ignore une force 2.
Si le recouvrement 17 dépasse ``overlap_max`` 8, cette erreur-là sort
avant. Le même validateur contrôle ensuite
``sliding_window_overlap_noise`` si ``vace_class`` est vrai, ou si
``model_type`` contient ``sky_df`` ou ``diffusion_forcing``. Le défaut
d'usine est 0 (minimum 0, maximum 150) : il passe.
``query_model_def`` de Wan pose ``vace_class`` pour ``test_vace`` au
premier JSON seulement. Un finetune VACE après ``t2v_1.3B`` ne
l'injecte pas. Une valeur 151 est refusée même si la force de couleur
vaut 0, ``sub_parallel_window_size`` vaut 0 et le recouvrement
parallèle tient dans la borne. La tâche est sautée et le CLI sort 0,
sans MP4. ``sky_df_1.3B`` rend le réglage visible sans ``vace_class``.
``df_handler`` ne pose pas ``color_correction`` : une force 2 est
ignorée et le bruit 151 est l'erreur. Sans ``vace_class`` et avec
``model_type`` ``t2v_1.3B``, le bruit 151 est ignoré. Si
``color_correction`` est vrai et que la force vaut 2, cette erreur-là
sort avant. Le même validateur contrôle ensuite
``sliding_window_discard_last_frames`` si ``sliding_window`` est vrai
et que ``model_type`` ne contient ni ``sky_df`` ni
``diffusion_forcing``. Le défaut d'usine est 0 (minimum 0, maximum 20) :
il passe. ``query_model_def`` de ``t2v_1.3B`` pose ``sliding_window``
à faux. Il faut l'écrire à vrai, ou un premier JSON qui l'injecte
(``t2v``, la famille I2V, VACE, ``fantasy``, ``lynx``, ``animate``,
Wan 5B, ``multitalk``, ``infinitetalk``). Une valeur 21 est refusée
même si le bruit vaut 0, la force de couleur vaut 0 et le recouvrement
parallèle tient dans la borne. La tâche est sautée et le CLI sort 0,
sans MP4. ``multitalk`` pose la valeur à 4, dans la borne. ``sky_df``
ou ``diffusion_forcing`` cache ce contrôle. Si le bruit 151 est visible,
cette erreur-là sort avant. ``large`` et ``21.5`` en chaîne lèvent
``ValueError`` dans le ``int()`` du frame scheduler, avant
``extra_settings``, quand ``image_mode`` est 0 et ``sliding_window``
est vrai : le processus ne sort pas 0. Le même validateur contrôle
ensuite ``sliding_window_trim_first_frames`` dès que ``sliding_window``
est vrai, y compris pour ``sky_df`` et ``diffusion_forcing``. Le défaut
d'usine est 0 (minimum 0, maximum 10) : il passe. ``wan_handler`` ne
réécrit pas cette clé. Une valeur 11 est refusée même si le rejet vaut
0. La tâche est sautée et le CLI sort 0, sans MP4. Si le rejet 21 est
visible, cette erreur-là sort avant. ``large`` n'est pas un ``int()``
du frame scheduler : ``extra_settings`` renvoie ``must be an integer``
et le CLI sort 0. Un ``discard_step`` 0, ou un rejet que ce ``int()``
refuse, lève avant ``extra_settings`` : ce n'est pas ce saut.
``temperature`` suit le trim, mais seulement si ``audio_only`` est vrai :
ce drapeau est déjà refusé. Le validateur contrôle ensuite ``top_p``
(nombre, minimum 0, maximum 1) si ``top_p_slider`` est vrai. Le défaut
d'usine est 0,9 : il passe. ``query_model_def`` de Wan ne pose pas le
drapeau. Un JSON qui l'écrit rend 1,01 ou 2 refusé. La tâche est sautée
et le CLI sort 0, sans MP4. ``top_k_slider`` seul ne rend pas ``top_p``
visible. Si le trim 11 est visible, cette erreur-là sort avant.
Le validateur contrôle ensuite ``top_k`` (entier, minimum 0, maximum
100) si ``top_k_slider`` est vrai. Le défaut d'usine est 50 : il passe.
``_trim_label`` coupe le libellé au premier ``(`` : l'erreur dit
``Top-k``. ``query_model_def`` de Wan ne pose pas le drapeau. Un JSON
qui l'écrit rend 101 refusé. La tâche est sautée et le CLI sort 0,
sans MP4. ``top_p_slider`` seul ne rend pas ``top_k`` visible. Si
``top_p`` 2 est visible, cette erreur-là sort avant. Le validateur
contrôle ensuite ``NAG_scale`` (nombre, minimum 1, maximum 20) si le
conteneur ``NAG_col`` est visible, donc si ``NAG`` est vrai. Le défaut
d'usine est 1 : il passe. ``query_model_def`` pose le drapeau pour
``test_class_t2v``, ``test_class_i2v`` et ``test_vace``, sauf
``bernini`` et ``animate2``. ``t2v_1.3B`` n'est pas dans
``test_class_t2v``. Un JSON qui écrit ``NAG`` rend 21 refusé. Un objet
``model.NAG_scale`` peut changer le libellé et les bornes. La tâche est
sautée et le CLI sort 0, sans MP4. Le validateur contrôle ensuite
``NAG_tau`` (nombre, minimum 1, maximum 5), dans le même conteneur.
Le défaut d'usine est 3,5 : il passe. Un JSON qui écrit ``NAG`` rend
5,01 refusé. Un objet ``model.NAG_tau`` peut changer le libellé et les
bornes. La tâche est sautée et le CLI sort 0, sans MP4. Si
``NAG_scale`` 21 est visible, cette erreur-là sort avant. Le validateur
contrôle ensuite ``NAG_alpha`` (nombre, minimum 0, maximum 2), dans le
même conteneur. Le défaut d'usine est 0,5 : il passe. Un JSON qui écrit
``NAG`` rend 2,01 refusé. Un objet ``model.NAG_alpha`` peut changer le
libellé et les bornes. La tâche est sautée et le CLI sort 0, sans MP4.
Si ``NAG_tau`` 5,01 est visible, cette erreur-là sort avant. Le validateur
contrôle ensuite ``pace`` (nombre, minimum 0,2, maximum 1) seulement si
``custom_settings`` contient l'id ``pace`` parmi les cinq premiers objets.
Il n'y a pas de défaut d'usine : une valeur absente passe. La valeur est
lue dans le dict ``custom_settings``, pas à la racine. 9, ``9`` et 1,01
sont refusés. La tâche est sautée et le CLI sort 0, sans MP4. L'objet
peut changer le libellé, le type et les bornes. Si ``NAG_alpha`` 2,01
est visible, cette erreur-là sort avant. Un objet illisible sur
``NAG_scale``, ``NAG_tau``, ``NAG_alpha``, ``switch_threshold`` ou
``temperature_slider`` lève ``AttributeError`` avant cette erreur : le
processus ne sort pas 0. Le validateur contrôle ensuite
``exaggeration`` (nombre, minimum 0,25, maximum 2), même mécanisme
custom. Il n'y a pas de défaut d'usine. La valeur est lue dans le dict,
pas à la racine. 3, ``3`` et 2,01 sont refusés. 0,24 est sous 0,25.
La tâche est sautée et le CLI sort 0, sans MP4. Si ``pace`` 9 est
visible, cette erreur-là sort avant. ``collect_custom_settings_from_inputs``
réécrit le dict avant ce contrôle. Un type ``integer`` devient ``text`` :
le flottant 1,5 est la chaîne ``1.5``, puis extra_settings refuse
« must be an integer ». Un type ``int`` reste entier : le même flottant
est refusé avant extra_settings (« Expected an integer value. »).
Un type ``dropdown`` avec des ``choices`` non vides refuse une valeur
hors liste avant extra_settings (« Expected one of the dropdown choices. »).
``_setting_kind("dropdown")`` vaut ``string`` : extra_settings ignore
la valeur, donc le saut vient seulement de ce parse. Une liste de choix
vide, ou une valeur qui n'est pas une liste, devient ``text`` et ne saute
pas. ``parse_custom_sigmas`` tourne juste après, seulement si
``model.custom_sigmas`` est vrai. ``query_model_def`` de Wan pose ce
drapeau, y compris pour ``t2v_1.3B``. ``models/_settings.json`` laisse
la chaîne vide et ``guidance_phases`` à 1 : vide et espaces passent.
``nope``, un entier, un ordre non descendant, une valeur hors de
``[0, 1[``, un ``0`` seul ou un mauvais nombre de phases sautent la
tâche. Le CLI sort 0, sans MP4. Un menu hors liste ou un type ``int``
sort avant. ``NAG`` sort après. Si au moins un créneau n'est pas
``None`` et que ``custom_sigmas_schedulers`` n'est pas ``None``, un
``sample_solver`` absent de cette liste saute aussi. Wan injecte
``["", "unipc", "euler", "dpm++"]``. ``lcm`` et ``causvid`` avec
``0.9,0.2`` sont refusés. L'usine devient ``unipc`` (``euler`` pour
``multitalk`` et ``infinitetalk``, ``dpm++`` pour ``animate2``). Une
chaîne de sigmas vide ignore le solveur. Le contrôle continu des
phases vient après.
"""

from __future__ import annotations

import contextvars
import json
import logging
import math
import os
import re
import shutil
import stat
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
# La file n'a pas toujours ``model_mode``. None voudrait dire « null explicite ».
_QUEUE_MODEL_MODE_UNSET = object()
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
# Pendant les probes, la racine du clone est remplacée par un dossier qui ne
# contient pas le ``ffmpeg`` que ``download_ffmpeg`` va mettre en quarantaine.
_PROBE_REPO_OVERRIDE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "wan2gp_probe_repo",
    default=None,
)
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
    if _visible_wan_script(root) is None:
        raise Wan2GPError(f"wgp.py introuvable dans {root}.")

    queue_model_mode = payload["model_mode"] if "model_mode" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_image_prompt_type = (
        payload["image_prompt_type"] if "image_prompt_type" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    queue_video_prompt_type = (
        payload["video_prompt_type"] if "video_prompt_type" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    queue_audio_prompt_type = (
        payload["audio_prompt_type"] if "audio_prompt_type" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    queue_image_refs = payload["image_refs"] if "image_refs" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_video_guide = payload["video_guide"] if "video_guide" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_audio_guide = payload["audio_guide"] if "audio_guide" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_custom_guide = payload["custom_guide"] if "custom_guide" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_sliding_window_overlap = (
        payload["sliding_window_overlap"]
        if "sliding_window_overlap" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_sliding_window_size = (
        payload["sliding_window_size"]
        if "sliding_window_size" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_sub_parallel_window_overlap = (
        payload["sub_parallel_window_overlap"]
        if "sub_parallel_window_overlap" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_sub_parallel_window_size = (
        payload["sub_parallel_window_size"]
        if "sub_parallel_window_size" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_color_correction_strength = (
        payload["sliding_window_color_correction_strength"]
        if "sliding_window_color_correction_strength" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_overlap_noise = (
        payload["sliding_window_overlap_noise"]
        if "sliding_window_overlap_noise" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_discard_last_frames = (
        payload["sliding_window_discard_last_frames"]
        if "sliding_window_discard_last_frames" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_trim_first_frames = (
        payload["sliding_window_trim_first_frames"]
        if "sliding_window_trim_first_frames" in payload
        else _QUEUE_MODEL_MODE_UNSET
    )
    queue_top_p = payload["top_p"] if "top_p" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_top_k = payload["top_k"] if "top_k" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_nag_scale = payload["NAG_scale"] if "NAG_scale" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_nag_tau = payload["NAG_tau"] if "NAG_tau" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_nag_alpha = payload["NAG_alpha"] if "NAG_alpha" in payload else _QUEUE_MODEL_MODE_UNSET
    queue_custom_settings = (
        payload["custom_settings"] if "custom_settings" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    queue_custom_sigmas = (
        payload["custom_sigmas"] if "custom_sigmas" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    queue_guidance_phases = (
        payload["guidance_phases"] if "guidance_phases" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    queue_sample_solver = (
        payload["sample_solver"] if "sample_solver" in payload else _QUEUE_MODEL_MODE_UNSET
    )
    binaries = require_ffmpeg(
        root,
        wan.python or sys.executable,
        model_type,
        queue_model_mode=queue_model_mode,
        queue_image_prompt_type=queue_image_prompt_type,
        queue_video_prompt_type=queue_video_prompt_type,
        queue_audio_prompt_type=queue_audio_prompt_type,
        queue_image_refs=queue_image_refs,
        queue_video_guide=queue_video_guide,
        queue_audio_guide=queue_audio_guide,
        queue_custom_guide=queue_custom_guide,
        queue_sliding_window_overlap=queue_sliding_window_overlap,
        queue_sliding_window_size=queue_sliding_window_size,
        queue_sub_parallel_window_overlap=queue_sub_parallel_window_overlap,
        queue_sub_parallel_window_size=queue_sub_parallel_window_size,
        queue_color_correction_strength=queue_color_correction_strength,
        queue_overlap_noise=queue_overlap_noise,
        queue_discard_last_frames=queue_discard_last_frames,
        queue_trim_first_frames=queue_trim_first_frames,
        queue_top_p=queue_top_p,
        queue_top_k=queue_top_k,
        queue_nag_scale=queue_nag_scale,
        queue_nag_tau=queue_nag_tau,
        queue_nag_alpha=queue_nag_alpha,
        queue_custom_settings=queue_custom_settings,
        queue_custom_sigmas=queue_custom_sigmas,
        queue_guidance_phases=queue_guidance_phases,
        queue_sample_solver=queue_sample_solver,
    )
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
        source = _visible_wan_script(root)
        if source is not None:
            match = _SETTINGS_VERSION_RE.search(source.read_text(encoding="utf-8", errors="replace"))
            if match:
                return float(match.group(1))
    return SETTINGS_VERSION


def require_ffmpeg(
    root: Path | None = None,
    python: str | None = None,
    model_type: str = "t2v_1.3B",
    queue_model_mode: object = _QUEUE_MODEL_MODE_UNSET,
    queue_image_prompt_type: object = _QUEUE_MODEL_MODE_UNSET,
    queue_video_prompt_type: object = _QUEUE_MODEL_MODE_UNSET,
    queue_audio_prompt_type: object = _QUEUE_MODEL_MODE_UNSET,
    queue_image_refs: object = _QUEUE_MODEL_MODE_UNSET,
    queue_video_guide: object = _QUEUE_MODEL_MODE_UNSET,
    queue_audio_guide: object = _QUEUE_MODEL_MODE_UNSET,
    queue_custom_guide: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sliding_window_overlap: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sliding_window_size: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sub_parallel_window_overlap: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sub_parallel_window_size: object = _QUEUE_MODEL_MODE_UNSET,
    queue_color_correction_strength: object = _QUEUE_MODEL_MODE_UNSET,
    queue_overlap_noise: object = _QUEUE_MODEL_MODE_UNSET,
    queue_discard_last_frames: object = _QUEUE_MODEL_MODE_UNSET,
    queue_trim_first_frames: object = _QUEUE_MODEL_MODE_UNSET,
    queue_top_p: object = _QUEUE_MODEL_MODE_UNSET,
    queue_top_k: object = _QUEUE_MODEL_MODE_UNSET,
    queue_nag_scale: object = _QUEUE_MODEL_MODE_UNSET,
    queue_nag_tau: object = _QUEUE_MODEL_MODE_UNSET,
    queue_nag_alpha: object = _QUEUE_MODEL_MODE_UNSET,
    queue_custom_settings: object = _QUEUE_MODEL_MODE_UNSET,
    queue_custom_sigmas: object = _QUEUE_MODEL_MODE_UNSET,
    queue_guidance_phases: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sample_solver: object = _QUEUE_MODEL_MODE_UNSET,
) -> FfmpegBinaries:
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
    de MP4. Ce fichier est ouvert depuis le cwd du clone. Un lien vers
    ``/proc/self/cwd/secret.json``, ``/proc/net/../cwd/secret.json`` ou
    ``hop1/../cwd/secret.json`` (``hop1`` → ``hop2`` → ``/proc/net``) suit
    le cwd du processus. ``Path.is_file()`` ici ouvre le dossier du pipeline :
    un leurre ``mp4`` laisse passer, ou le fichier du clone n'est pas vu.
    WanGP lit pourtant le JSON du clone (``mkv``, ou un codec ProRes) et
    quitte avec le code 0, sans MP4. Un ``mkv`` qui n'existe que dans le
    dossier du pipeline n'est pas celui que WanGP ouvre : le clone sans ce
    fichier repart du défaut ``mp4``.
    ``refresh_model_defs`` charge ``defaults/*.json`` puis ``finetunes/*.json``
    (triés par ce chemin relatif) pendant que le cwd est le clone. Si le
    modèle demandé a ``image_outputs``, ``validate_settings`` force
    ``image_mode`` à 1 même quand la file demande 0, et la tâche écrit un
    JPEG. ``audio_only``, ``mesh_outputs`` ou ``world_outputs`` écrivent
    un autre fichier. ``process_tasks_cli`` compte la tâche réussie et sort
    avec le code 0. Un lien ``defaults/t2v_1.3B.json`` ou le dossier
    ``defaults`` vers ``/proc/self/cwd``, ``/proc/net/../cwd`` ou
    ``hop1/../cwd`` (``hop1`` → ``hop2`` → ``/proc/net``) n'est pas le fichier
    que ``Path.is_file()`` ou ``iterdir`` voient depuis le pipeline. Le
    leurre y a ``image_outputs`` faux, ou le lien est cassé, alors que le
    clone a le drapeau vrai. L'inverse, un JSON seulement dans le dossier
    du pipeline, n'est pas lu : WanGP garde la définition sans ce drapeau.
    Un plugin listé dans ``enabled_plugins``, ou un plugin système
    (``guides``, ``configuration``, ``plugin_manager``, ``about``,
    ``video_mask_creator``), ajoute son dossier ``defaults``.
    ``plugin_info.json`` doit donner ``model_handlers``, ``defaults`` et
    ``profiles``. ``os.path.abspath`` rend ce dossier absolu : le tri le
    place avant ``defaults/*.json`` et ``finetunes/*.json``. Le drapeau
    du plugin reste si les fichiers suivants ne réécrivent pas la clé.
    Le ``defaults`` officiel de ``t2v_1.3B`` ne mentionne pas
    ``image_outputs``. ``init_model_def`` ne voit que le premier JSON de
    ce modèle (le chemin absolu d'un plugin passe avant ``defaults/``).
    ``query_model_def`` de ``chrono_edit``, de Flux, d'un modèle image,
    audio, mesh ou world pose le drapeau correspondant même si la clé
    manque. Un fichier suivant qui change ``architecture`` sans réécrire
    la clé la laisse vraie : le JPEG, le son, le glb ou le ``.wgpworld``
    sort quand même, et le CLI quitte avec le code 0. Un fichier qui
    écrit le drapeau à faux l'efface. L'architecture finale reste celle du
    dernier JSON. ``get_base_model_type`` la donne à
    ``validate_generative_settings`` : pour ``chrono_edit``, ``model_mode``
    différent de 2 force ``image_mode`` à 1 et écrit un JPEG, même quand
    ``image_outputs`` est faux parce que ``query_model_def`` n'a pas été
    rappelé. ``model_mode`` vient des clés hors ``model`` (réglages du
    JSON) puis de la file si elle l'envoie non nul. Ce 2 ne tient pas si
    ``model_modes.image_modes`` ne contient pas 0 : ``validate_settings``
    remet ``model_mode`` à None avant le handler, le JPEG est écrit, et
    le CLI sort 0. Sans la clé ``image_modes`` dans le JSON, et sans
    l'injection du premier ``query_model_def``, la liste reste
    ``[0, 1, 2]`` et la vidéo est écrite. ``fix_settings`` est celui de
    l'architecture finale, après ``clean_settings``. Pour
    ``qwen_image_21_7B``, ``models/qwen/qwen_handler.py`` appelle
    ``models/qwen21/qwen21_handler.py`` : ``image_mode`` 0, ``None`` ou
    ``"0"`` redevient 1, même si ``image_outputs`` est faux ou absent
    (``query_model_def`` n'est pas rappelé). Un JPEG est écrit et le CLI
    sort 0. ``setdefault`` (Flux, Krea, Z-Image, Ideogram) ne remplace
    pas le 0 de la file. ``query_model_def`` du premier JSON injecte
    aussi ``model_modes.image_modes`` ``[2]`` pour ``qwen_image_21_7B``,
    ``qwen_image_20B``, les Qwen edit, Flux si ``supports_inpaint``
    (``pi_flux2`` a l'inpaint mais pas ``model_modes``) et Krea 2.
    ``init_model_def`` fait ``default_model_def.update(model_def)`` :
    un JSON qui contient ``model_modes`` remplace toute l'injection.
    Aucun JSON ne l'écrit : elle reste. Un fichier suivant peut alors
    passer l'architecture à ``chrono_edit``, ``image_outputs`` à faux
    et ``model_mode`` à 2. Le runner ne voyait pas la clé et laissait
    ce 2. WanGP garde ``[2]``, ``0`` n'y est pas, ``model_mode``
    devient None, Chrono Edit écrit un JPEG et le CLI sort 0.
    ``defaults`` peut valoir
    ``/proc/self/cwd/<dossier>`` :
    ``glob`` suit le cwd du clone. Un lien ``plugins`` ou
    ``plugin_info.json`` vers ``/proc/self`` a le même effet.
    ``Path.is_file()`` depuis le pipeline lit le leurre, ou ne voit pas
    le fichier, et le GPU part. Un plugin qui n'est ni activé ni système
    n'est pas chargé. Un finetune qui remet le drapeau à faux gagne,
    parce que ``finetunes/*.json`` reste après les chemins absolus.
    Quand le binaire est dans ``ffmpeg_bins``, ``-encoders`` sans
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
    ``download_ffmpeg`` déplace d'abord un fichier ``ffmpeg`` posé à la racine
    du clone (lien compris) vers ``ffmpeg_quarantine``, puis seulement il met
    ``ffmpeg_bins`` et cette racine en tête du PATH. Tant que ce fichier est
    là, un script ``exec ffmpeg`` le trouve et ``-encoders`` annonce libx264.
    Après la quarantaine, le même script se retrouve lui-même et ne revient
    pas. ``save_video`` imprime ``error saving`` et le CLI sort 0, sans MP4.
    ``shutil.which`` peut aussi renvoyer ce fichier de la racine quand son
    dossier est sur le PATH. Le contrôle le voit encore, donc ``-encoders``
    réussit, et ``IMAGEIO_FFMPEG_EXE`` garde ce chemin. ``download_ffmpeg``
    le déplace ensuite : imageio ne trouve plus l'exécutable, ``save_video``
    imprime ``error saving`` et le CLI sort 0. Le binaire retenu est celui
    que ``which`` trouve une fois ce fichier écarté. Une entrée relative du
    PATH est lue depuis le clone, pas depuis le dossier du pipeline.
    Le dossier préfixé est absolu : ``Path(__file__).resolve()`` ne laisse pas
    un ``WAN2GP_ROOT`` relatif. Depuis le cwd du clone, ``wan/ffmpeg_bins``
    viserait un autre dossier, et une bibliothèque trop courte passerait.
    ``download_ffmpeg`` ne préfixe pas le ``ffmpeg_bins`` du cwd. Il résout
    ``shared/ffmpeg_setup.py`` (``Path(__file__).resolve()``). Un lien vers
    un autre clone, ou un ``wgp.py`` qui est lui-même un lien (``sys.path[0]``
    devient le dossier réel), charge les bibliothèques de cet autre dossier.
    imageio exécute pourtant ``IMAGEIO_FFMPEG_EXE``, souvent le binaire du
    clone. Une ``libx264.so*`` trop courte là-bas casse le chargeur,
    ``-encoders`` sans ce préfixe annonce encore libx264, ``save_video``
    imprime ``error saving`` et le CLI sort 0.
    Ce ``resolve`` suit ``/proc/self`` du processus qui charge le module.
    Le contrôle n'a pas le cwd du clone. ``shared`` → ``/proc/self/cwd/nested/shared``,
    ``/proc/net/../cwd/nested/shared``, ou ``hop1/../cwd/nested/shared``
    (``hop1`` → ``hop2`` → ``/proc/net``) : ``is_file`` et ``Path.resolve()``
    ici ouvrent le dossier du pipeline, ou ne voient pas le fichier.
    Un leurre ``ffmpeg_setup.py`` dans ce cwd fait croire que le préfixe
    est le sien. Le processus WanGP a pour cwd le clone, donc
    ``Path(__file__).resolve().parents[1]`` est ``<clone>/nested``.
    ``download_ffmpeg`` préfixe ``<clone>/nested/ffmpeg_bins``. Une
    ``libx264.so*`` trop courte seulement là donne ``file too short``,
    ``-encoders`` sans ce préfixe annonce encore libx264, ``save_video``
    imprime ``error saving`` et le CLI sort 0, sans MP4. Le même lien sur
    le fichier ``ffmpeg_setup.py`` (pas sur le dossier ``shared``) a le
    même effet.     Un soname présent seulement dans le cwd du pipeline n'est
    pas celui que le chargeur ouvre.
    ``wgp.py`` peut être ce lien, plutôt que ``shared``. ``sys.path[0]``
    initial est le ``realpath`` du script, calculé avec le cwd du clone.
    ``wgp.py`` insère ensuite ``os.path.dirname(os.path.abspath(__file__))``,
    qui ne résout pas le lien. ``wgp.py`` → ``/proc/self/cwd/nested/wgp.py``,
    ``/proc/net/../cwd/nested/wgp.py`` ou ``hop1/../cwd/nested/wgp.py``
    (``hop1`` → ``hop2`` → ``/proc/net``) : ``is_file`` et ``Path.resolve()``
    ici ouvrent le dossier du pipeline. Un leurre ``wgp.py`` rend le script
    présent, et sans leurre ``is_file`` est faux alors que le clone a le
    fichier. Sans ``shared/ffmpeg_setup.py`` dans le dossier du lien,
    l'import tombe sur ``<clone>/nested/shared``. ``download_ffmpeg`` préfixe
    ``<clone>/nested/ffmpeg_bins``. Une ``libx264.so*`` trop courte seulement
    là donne ``file too short``. ``-encoders`` sans ce préfixe annonce encore
    libx264, ``save_video`` imprime ``error saving`` et le CLI sort 0, sans
    MP4.
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
    ``ffmpeg_bins/ffmpeg`` peut être un lien vers le fichier ``ffmpeg`` de
    la racine, éventuellement via un autre lien. ``is_file`` suit cette
    chaîne tant que la cible est là : ``-encoders`` et le mux d'essai
    réussissent, et ``IMAGEIO_FFMPEG_EXE`` garde ce chemin.
    ``download_ffmpeg`` déplace d'abord la cible vers ``ffmpeg_quarantine``.
    ``_resolve_path`` ne voit plus le lien (``exists`` est faux) et prend
    le ffmpeg du PATH. imageio n'examine pas ``IMAGEIO_FFMPEG_EXE`` et
    l'exécute tel quel : ``FileNotFoundError``, ``save_video`` imprime
    ``error saving``, le CLI sort 0, sans MP4. Un lien direct vers le
    binaire du système, ou un lien dur, reste en place après ce
    déplacement. Un ``ffprobe`` qui passe par le même fichier est aussi
    écarté : le contrôle du MP4 l'utilise encore après la quarantaine.
    La chaîne du lien n'a pas besoin d'être le même texte. ``normpath``
    conserve un ``//`` initial : ``//<racine>/ffmpeg`` et ``<racine>/ffmpeg``
    sont la même entrée pour le noyau, pas pour une comparaison de chaînes.
    ``../self/ffmpeg`` aussi, quand ``self`` est un lien vers ``.``.
    ``-encoders`` et le mux d'essai réussissent, le lien est publié, puis
    la quarantaine le casse.
    ``normpath`` annule ``..`` avant de suivre un lien de dossier.
    ``ffmpeg_bins/ffmpeg`` → ``trick/../ffmpeg``, avec ``trick`` qui pointe
    vers un sous-dossier du clone, revient sur ``ffmpeg_bins/ffmpeg`` dans
    la chaîne écrite. Le noyau entre dans ``trick``, applique ``..`` ensuite,
    et ouvre le ``ffmpeg`` de la racine. ``-encoders`` et le mux d'essai
    réussissent, le lien est publié, puis la quarantaine le casse.
    Un ``trick/../bin/ffmpeg`` qui aboutit au binaire du système, sans
    passer par ce fichier, reste accepté.
    Le PATH du probe a le même trou. ``download_ffmpeg`` ne retire que les
    dossiers dont le texte, après ``normpath``, est déjà ``ffmpeg_bins`` ou
    la racine. ``normpath`` annule ``..`` avant le lien : ``holder/trick/..``
    devient ``holder``, alors que le noyau entre dans ``trick`` et ouvre la
    racine. Une entrée relative (``.`` ou ``trick/..``) reste écrite telle
    quelle et est ouverte depuis le cwd du clone. ``-encoders`` du script
    ``exec ffmpeg`` trouve encore le fichier, annonce libx264, et le probe
    accepte le script. ``download_ffmpeg`` déplace ensuite ce fichier.
    Le script se relance, ``save_video`` imprime ``error saving`` et le CLI
    sort 0, sans MP4.     Le probe retire ces entrées. Un ``trick/..`` qui ouvre
    un autre dossier, avec son propre ``ffmpeg``, reste sur le PATH.
    ``/proc/self/cwd`` et ``/proc/thread-self/cwd`` suivent le cwd du
    processus qui cherche sur le PATH, pas celui du contrôle. Le processus
    WanGP a pour cwd le clone : cette entrée ouvre la racine, et
    ``exec ffmpeg`` trouve encore le fichier. ``Path.resolve()`` dans le
    contrôle ouvre le dossier du pipeline, donc l'entrée restait sur le
    PATH du probe. ``-encoders`` annonce libx264, ``IMAGEIO_FFMPEG_EXE``
    garde le script, puis la quarantaine retire le fichier. Le script se
    relance, ``save_video`` imprime ``error saving`` et le CLI sort 0,
    sans MP4. Un lien vers ce chemin proc, ou ``/proc/self/cwd/sous/..``,
    ouvre la même racine. Le probe compare l'inode vu par un processus
    dont le cwd est le clone.
    ``normpath`` annule ``..`` avant de suivre ces liens. ``/proc/self/../self/cwd``
    redevient ``/proc/self/cwd`` dans le texte, ``/proc/thread-self/../../cwd``
    devient ``/cwd``, et ``/proc/self/root/proc/self/cwd`` ne change pas :
    ``/proc/self/root`` est ``/``. Le noyau entre dans le lien, puis applique
    ``..``. Depuis le clone, ces entrées ouvrent encore la racine.
    ``Path.resolve()`` dans le contrôle suit ``/proc/self`` du processus
    courant et tombe sur le dossier du pipeline, donc l'entrée restait sur
    le PATH du probe. ``-encoders`` annonce libx264, puis la quarantaine
    retire le fichier. Le script se relance, ``save_video`` imprime
    ``error saving`` et le CLI sort 0, sans MP4. Un lien de dossier vers
    ``/proc/self`` ou ``/proc/thread-self``, suivi de ``cwd``, a le même
    effet. Le probe reconnaît la paire de composants, sans annuler ``..``,
    et compare l'inode vu avec le cwd du clone.
    ``/proc/sys`` est un dossier réel. ``/proc/sys/../self/cwd`` et
    ``/proc/sys/kernel/../../self/cwd`` n'ont pas ``self`` juste après
    ``proc``. ``normpath`` les réécrit en ``/proc/self/cwd``, donc
    ``download_ffmpeg`` ne retire pas l'entrée : ce n'est pas le texte de
    la racine. Le noyau entre dans ``sys`` avant ``..`` et ouvre le cwd du
    clone. ``Path.resolve()`` dans le contrôle suit ``/proc/self`` du
    processus courant et tombe sur le dossier du pipeline. ``-encoders``
    annonce libx264, puis la quarantaine retire le fichier. Le script se
    relance, ``save_video`` imprime ``error saving`` et le CLI sort 0,
    sans MP4. Un lien vers ``/proc/sys`` suivi de ``../self/cwd``, un lien
    dont la cible est ``/proc/sys/../self/cwd``, et
    ``/proc/sys/../thread-self/cwd`` font pareil. Le probe reconnaît
    ``proc`` puis, plus loin, ``self`` ou ``thread-self``, sans exiger
    qu'ils se touchent, et compare l'inode vu avec le cwd du clone.
    Deux liens de dossier font échouer cette lecture si ``..`` est annulé
    trop tôt. ``hop1`` pointe vers ``hop2`` et ``hop2`` vers ``/proc/net``.
    L'entrée ``hop1/../cwd`` ne contient ni ``proc`` ni ``self``.
    ``normpath`` en fait le même texte que ``hop2/../cwd``. Une boucle
    détectée sur ce texte s'arrête avant de lire ``hop2``. Le noyau entre
    dans ``hop1``, puis dans ``hop2`` (``/proc/net``), puis ``..`` retombe
    dans ``/proc/self`` et ``cwd`` ouvre le cwd du clone.
    ``download_ffmpeg`` ne retire pas l'entrée : ce n'est pas le texte de
    la racine. ``-encoders`` du script ``exec ffmpeg`` annonce libx264,
    puis la quarantaine retire le fichier. Le script se relance,
    ``save_video`` imprime ``error saving`` et le CLI sort 0, sans MP4.
    Un lien relatif (``rel1`` → ``rel2``) et une chaîne ``hop1`` → ``hop2``
    → ``/proc/sys`` avec l'entrée ``hop1/../self/cwd`` font pareil :
    ``self`` est écrit, mais ``proc`` n'apparaît qu'au second lien.
    ``hop1/../cwd`` quand ``hop2`` pointe vers ``/proc/sys`` n'ouvre pas la
    racine (``/proc/cwd`` n'existe pas). Le probe conserve ``..`` dans la
    clé de boucle, lit le second lien, et compare l'inode.
    ``ffmpeg_bins/ffmpeg`` peut viser ``tonet/../cwd/ffmpeg``, ``tonet``
    pointant vers ``/proc/net``. ``normpath`` annule ``..`` avant ce lien :
    la chaîne écrite ne nomme pas le fichier de la racine. Depuis le dossier
    du pipeline, ``is_file`` ouvre le ``ffmpeg`` de ce cwd quand il existe.
    Le processus a pour cwd le clone. Le noyau entre dans ``/proc/net``
    (``self/net``), ``..`` retombe dans ``/proc/self`` et ``cwd`` ouvre le
    fichier que la quarantaine déplace. ``-encoders`` et le mux, lancés
    avec ce cwd, réussissent tant que le fichier est là.
    ``IMAGEIO_FFMPEG_EXE`` garderait ce lien. ``download_ffmpeg`` déplace
    ensuite la cible. imageio l'exécute tel quel : ``FileNotFoundError``,
    ``save_video`` imprime ``error saving`` et le CLI sort 0, sans MP4.
    Une chaîne ``hop1`` → ``hop2`` → ``/proc/net`` fait pareil. Un
    ``tonet/../root/usr/bin/ffmpeg`` ou un ``tonet/../cwd/sub/ffmpeg``
    n'ouvre pas ce fichier et reste accepté.
    ``ffmpeg_bins`` peut lui-même être ce lien, vers ``/proc/self/cwd``
    ou vers ``/proc/net/../cwd``. Le fichier contrôlé est alors
    ``ffmpeg_bins/ffmpeg``. ``is_symlink`` suit le dossier et statue le
    dernier nom dans le cwd du pipeline. Un fichier régulier posé là
    (pas un lien) rend ``is_symlink`` faux et ``is_file`` vrai : ce n'est
    pas le fichier de la racine. Le processus a pour cwd le clone, donc
    le même chemin ouvre le ``ffmpeg`` que la quarantaine déplace.
    ``-encoders`` réussit tant qu'il est là. ``IMAGEIO_FFMPEG_EXE``
    garderait ce chemin. Après le déplacement, imageio ne le trouve plus.
    ``save_video`` imprime ``error saving`` et le CLI sort 0, sans MP4.
    Une fois ce binaire écarté, le runner garde le ffmpeg du PATH.
    ``download_ffmpeg`` préfixe quand même ``LD_LIBRARY_PATH`` avec le
    lien. ``iterdir`` depuis le pipeline liste le cwd du pipeline, pas
    la racine du clone. Un ``libx264.so*`` trop court posé là n'est pas
    vu. Le chargeur, dont le cwd est le clone, ouvre
    ``ffmpeg_bins/<soname>`` et échoue (``file too short``). ``-encoders``
    sans ce préfixe annonce encore libx264. ``save_video`` imprime
    ``error saving`` et le CLI sort 0, sans MP4.
    Un ``ffmpeg_bins`` qui pointe vers un autre dossier, avec son propre
    binaire, reste accepté.
    ``glibc-hwcaps`` ou un niveau ``glibc-hwcaps/<niveau>`` peut être un lien
    vers ``/proc/self/cwd``, ``/proc/net/../cwd`` ou ``/proc/self/cwd/only``
    pendant que ``ffmpeg_bins`` reste un dossier réel. ``iterdir`` depuis le
    pipeline suit le cwd du contrôle et rate le ``libx264.so*`` trop court
    du clone. Un lien de soname vers ``/proc/self/cwd/<fichier>`` fait pareil :
    ``is_file`` ouvre le cwd du pipeline. Le chargeur, dont le cwd est le
    clone, lit ce fichier (``file too short``). ``-encoders`` sans le préfixe
    annonce encore libx264. ``save_video`` imprime ``error saving`` et le CLI
    sort 0, sans MP4. Ces dossiers et ces liens sont relus depuis un
    processus dont le cwd est le clone. Un fichier présent seulement dans le
    cwd du pipeline n'est pas celui que le chargeur ouvre.
    """
    _require_video_model_definition(
        root,
        model_type,
        queue_model_mode,
        queue_image_prompt_type,
        queue_video_prompt_type,
        queue_audio_prompt_type,
        queue_image_refs,
        queue_video_guide,
        queue_audio_guide,
        queue_custom_guide,
        queue_sliding_window_overlap,
        queue_sliding_window_size,
        queue_sub_parallel_window_overlap,
        queue_sub_parallel_window_size,
        queue_color_correction_strength,
        queue_overlap_noise,
        queue_discard_last_frames,
        queue_trim_first_frames,
        queue_top_p,
        queue_top_k,
        queue_nag_scale,
        queue_nag_tau,
        queue_nag_alpha,
        queue_custom_settings,
        queue_custom_sigmas,
        queue_guidance_phases,
        queue_sample_solver,
    )
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
    shadow, token = _activate_probe_repo(root)
    try:
        _require_inherited_library_traps(ffmpeg, root)
        _require_encoder(ffmpeg, encoder, root)
        _require_ffprobe(ffprobe, root)
        _require_bundle_libs(ffmpeg, root)
        _require_ld_library_shadow(ffmpeg, root)
        _require_preload_library_traps(ffmpeg, root, preload)
        _require_bundled_mux(ffmpeg, ffprobe, root, encoder, preload)
    finally:
        _release_probe_repo(shadow, token)
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
    doomed = None if root is None else _repo_ffmpeg_file(root)
    if root is not None:
        bundled = root / "ffmpeg_bins" / filename
        if _survives_quarantine(bundled, doomed):
            return bundled
        # Seul ffmpeg.exe / ffmpeg est mis en quarantaine. ffprobe à la racine reste utilisé,
        # sauf s'il est un lien vers ce fichier : la quarantaine casse alors le lien.
        if name != "ffmpeg":
            at_root = root / filename
            if _survives_quarantine(at_root, doomed):
                return at_root
    if name == "ffmpeg" and doomed is not None and root is not None:
        found = _which_after_quarantine(filename, root)
    else:
        found = shutil.which(filename)
        if found is not None and doomed is not None and _depends_on_quarantined_file(Path(found), doomed):
            found = _which_ignoring_quarantine(filename, os.environ.get("PATH", ""), doomed)
    if found is None and filename != name:
        found = shutil.which(name)
    return Path(found) if found else None


def _path_from_clone(path: str, root: Path) -> str:
    """PATH tel que ``shutil.which`` le voit quand le cwd est le clone.

    Une entrée relative (``.`` compris) est ouverte depuis ce dossier.
    Le contrôle tourne dans le dossier du pipeline : sans cette réécriture,
    ``which`` choisirait un autre ``ffmpeg`` que ``download_ffmpeg``.
    """
    parts: list[str] = []
    for part in path.split(os.pathsep):
        if not part:
            continue
        if os.path.isabs(part):
            parts.append(part)
            continue
        parts.append(os.path.normpath(str((root / part).absolute())))
    return os.pathsep.join(parts)


def _which_after_quarantine(filename: str, root: Path) -> str | None:
    """``ffmpeg`` que WanGP trouve après avoir déplacé celui de la racine.

    ``shutil.which`` sur le PATH d'origine peut renvoyer ce fichier. Il
    n'existera plus quand imageio lira ``IMAGEIO_FFMPEG_EXE``. La recherche
    utilise le PATH de ``download_ffmpeg`` : ``ffmpeg_bins``, puis une racine
    sans ce ``ffmpeg``, puis le reste. Le fichier de la racine est ignoré.
    Un lien de ``ffmpeg_bins`` dont la chaîne aboutit à ce fichier l'est
    aussi : ``which`` le verrait encore, ``exists`` plus tard non.
    """
    shadow, token = _activate_probe_repo(root)
    try:
        search = _wan_path(root, _path_from_clone(os.environ.get("PATH", ""), root))
        return _which_ignoring_quarantine(filename, search, _repo_ffmpeg_file(root))
    finally:
        _release_probe_repo(shadow, token)


def _path_key(path: Path) -> str:
    """Clé comparable d'une entrée de répertoire, sans suivre le lien."""
    return os.path.normpath(os.path.abspath(str(path)))


def _entry_key(path: Path) -> str | None:
    """Entrée de répertoire : ancêtres résolus, dernier composant non suivi.

    ``abspath`` et ``normpath`` laissent un ``//`` initial. Le noyau ouvre
    la même entrée que sous ``/``. Un dossier intermédiaire qui est un lien
    (``self`` → ``.``) est aussi la même entrée. Le dernier nom n'est pas
    suivi : deux liens vers ``/usr/bin/ffmpeg`` restent deux entrées, et un
    lien dur aussi. ``samefile`` confondrait ces cas, parce qu'il suit le
    lien jusqu'au binaire.
    """
    try:
        parent = path.parent.resolve()
    except OSError:
        return None
    return os.path.normpath(str(parent / path.name))


def _depends_on_quarantined_file(path: Path, doomed: Path) -> bool:
    """``True`` si ouvrir ``path`` passe par le fichier que la quarantaine déplace.

    La comparaison porte sur l'entrée de répertoire, pas sur ``samefile`` :
    deux liens vers ``/usr/bin/ffmpeg`` restent valides quand l'un des deux
    est déplacé. Un lien ``ffmpeg_bins/ffmpeg`` → ``../ffmpeg`` casse.
    ``../self/ffmpeg`` aussi, quand ``self`` pointe vers le dossier du
    fichier déplacé. ``//<chemin>`` désigne cette même entrée : la chaîne
    ``normpath`` ne suffit pas à la reconnaître.
    ``normpath`` annule aussi ``..`` avant le lien de dossier. ``trick/../ffmpeg``
    redevient le lien de départ dans la chaîne écrite, alors que le noyau
    entre dans ``trick`` puis remonte, et ouvre le fichier déplacé. Cette
    entrée est comparée avant de traiter la répétition lexicale comme une
    boucle. Un ``..`` qui retombe sur le binaire du système ne passe pas
    par ce fichier : le lien reste valable après la quarantaine.
    ``tonet/../cwd/ffmpeg`` non plus n'est pas cette entrée dans la chaîne
    écrite, quand ``tonet`` pointe vers ``/proc/net``. ``normpath`` annule
    ``..`` avant le lien. Depuis le dossier du pipeline, ``resolve`` suit
    ``/proc/self`` de ce processus. Avec le cwd du clone, le noyau entre
    dans ``/proc/net``, ``..`` ouvre ``/proc/self/cwd`` et le dernier nom
    est le fichier déplacé. Une chaîne ``hop1`` → ``hop2`` → ``/proc/net``
    fait pareil. ``is_file`` peut quand même être vrai : il ouvre le
    ``ffmpeg`` du cwd du pipeline. ``-encoders`` avec le cwd du clone
    réussit tant que le fichier de la racine est là.
    ``ffmpeg_bins`` peut être ce lien de dossier (``/proc/self/cwd`` ou
    ``/proc/net/../cwd``). ``is_symlink`` suit alors le dossier et voit
    le dernier nom dans le cwd du pipeline. Un fichier régulier à ce
    nom n'est pas un lien : l'entrée publiée reste quand même celle que
    la quarantaine déplace, dès que le cwd est le clone.
    """
    doomed_key = _path_key(doomed)
    doomed_entry = _entry_key(doomed)
    current = path
    seen: set[str] = set()
    seen_entries: set[str] = set()
    while True:
        key = _path_key(current)
        entry = _entry_key(current)
        if key == doomed_key or (entry is not None and entry == doomed_entry):
            return True
        if entry is not None:
            if entry in seen_entries:
                return _proc_cwd_entry_is_quarantined(path, doomed)
            seen_entries.add(entry)
        elif key in seen:
            return _proc_cwd_entry_is_quarantined(path, doomed)
        seen.add(key)
        try:
            is_link = current.is_symlink()
        except OSError:
            return _proc_cwd_entry_is_quarantined(path, doomed)
        if not is_link:
            return _proc_cwd_entry_is_quarantined(path, doomed)
        try:
            raw = os.readlink(current)
        except OSError:
            return _proc_cwd_entry_is_quarantined(path, doomed)
        nxt = Path(raw)
        if not nxt.is_absolute():
            nxt = current.parent / nxt
        current = nxt


_PROC_ENTRY_WALK = """
import os
import stat
import sys

parent_dev = int(sys.argv[2])
parent_ino = int(sys.argv[3])
wanted = sys.argv[4]
current = sys.argv[1]
seen: set[tuple[int, int]] = set()
for _ in range(32):
    try:
        info = os.lstat(current)
    except OSError:
        raise SystemExit(1)
    parent = os.path.dirname(current) or "/"
    base = os.path.basename(current)
    try:
        parent_info = os.stat(parent)
    except OSError:
        raise SystemExit(1)
    if (
        stat.S_ISDIR(parent_info.st_mode)
        and (parent_info.st_dev, parent_info.st_ino, base) == (parent_dev, parent_ino, wanted)
    ):
        print("yes")
        raise SystemExit(0)
    if not stat.S_ISLNK(info.st_mode):
        raise SystemExit(1)
    key = (info.st_dev, info.st_ino)
    if key in seen:
        raise SystemExit(1)
    seen.add(key)
    try:
        raw = os.readlink(current)
    except OSError:
        raise SystemExit(1)
    if os.path.isabs(raw):
        current = raw
    else:
        current = os.path.join(os.path.dirname(current), raw)
raise SystemExit(1)
"""


def _proc_cwd_entry_is_quarantined(path: Path, doomed: Path) -> bool:
    """``True`` si le cwd du clone ouvre l'entrée que la quarantaine déplace.

    La marche lexicale annule ``..`` avant ``/proc/net``. ``resolve`` suit
    ``/proc/self`` du processus courant, souvent le dossier du pipeline.
    ``is_file`` y voit alors un autre ``ffmpeg``. Le processus WanGP a pour
    cwd le clone : ``tonet/../cwd/ffmpeg`` (``tonet`` → ``/proc/net``) ouvre
    le fichier de la racine. ``stat`` du dossier parent suit ``cwd`` ; le
    dernier nom n'est pas suivi, donc deux liens vers ``/usr/bin/ffmpeg``
    restent distincts. Un lien dur non plus : ce n'est pas un lien, et son
    dossier n'est pas celui du fichier déplacé.
    Si ``ffmpeg_bins`` est le lien (vers ``/proc/self/cwd`` ou
    ``/proc/net/../cwd``), le chemin ``ffmpeg_bins/ffmpeg`` n'est pas un
    lien quand le cwd du pipeline contient un fichier régulier de ce nom.
    ``is_symlink`` est alors faux, mais le cwd du clone ouvre quand même
    l'entrée déplacée. Le probe ne s'arrête pas à ``is_symlink``.
    """
    if os.name == "nt":
        return False
    try:
        # Le dernier composant n'a pas besoin d'être un lien. ``ffmpeg_bins``
        # peut l'être, vers ``/proc/self/cwd``. ``is_symlink`` suit ce dossier
        # et voit un fichier régulier du cwd du pipeline. Le clone, lui,
        # ouvre le fichier que la quarantaine déplace.
        if not _path_depends_on_proc_cwd(str(path.absolute())):
            return False
    except OSError:
        return False
    try:
        parent_info = os.stat(doomed.parent)
    except OSError:
        return False
    if not stat.S_ISDIR(parent_info.st_mode):
        return False
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                _PROC_ENTRY_WALK,
                str(path.absolute()),
                str(parent_info.st_dev),
                str(parent_info.st_ino),
                doomed.name,
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(doomed.parent.absolute()),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(
            f"{path} ne revient pas quand le cwd est {doomed.parent}. "
            "Ce lien peut traverser /proc/net puis cwd et ouvrir le ffmpeg "
            "que download_ffmpeg déplace vers ffmpeg_quarantine. "
            "is_file depuis le dossier du pipeline voit un autre fichier. "
            "save_video imprime error saving et le processus quitte avec le code 0, sans MP4."
        ) from exc
    return completed.returncode == 0 and completed.stdout.strip() == "yes"


def _survives_quarantine(path: Path, doomed: Path | None) -> bool:
    """``True`` si ce chemin est un fichier que la quarantaine ne va pas casser."""
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    if doomed is None:
        return True
    return not _depends_on_quarantined_file(path, doomed)


def _which_ignoring_quarantine(
    filename: str,
    search: str,
    doomed: Path | None,
) -> str | None:
    """``shutil.which`` en sautant un lien qui suit le fichier mis en quarantaine."""
    parts = [part for part in search.split(os.pathsep) if part]
    while parts:
        found = shutil.which(filename, path=os.pathsep.join(parts))
        if found is None:
            return None
        if doomed is None or not _depends_on_quarantined_file(Path(found), doomed):
            return found
        parent = os.path.normpath(str(Path(found).parent))
        remaining = [part for part in parts if os.path.normpath(part) != parent]
        if len(remaining) == len(parts):
            return None
        parts = remaining
    return None


def _bundled_ffmpeg_lost_in_quarantine(root: Path | None) -> Path | None:
    """Lien ``ffmpeg_bins/ffmpeg`` que ``_resolve_path`` ne verra plus."""
    if root is None or os.name == "nt":
        return None
    doomed = _repo_ffmpeg_file(root)
    if doomed is None:
        return None
    bundled = root / "ffmpeg_bins" / _binary_name("ffmpeg")
    try:
        visible = bundled.is_symlink() or bundled.is_file()
    except OSError:
        return None
    if visible and _depends_on_quarantined_file(bundled, doomed):
        return bundled
    return None


def _quarantine_link_note(root: Path | None) -> str:
    """Pourquoi l'encodeur retenu n'est pas le lien encore visible dans ``ffmpeg_bins``."""
    bundled = _bundled_ffmpeg_lost_in_quarantine(root)
    if bundled is None or root is None:
        return ""
    doomed = _repo_ffmpeg_file(root)
    return (
        f"{bundled} pointe vers {doomed}, que download_ffmpeg déplace vers "
        "ffmpeg_quarantine avant _resolve_path. Le lien reste un exécutable "
        "jusqu'à ce déplacement : -encoders et le mux d'essai réussissent, et "
        "IMAGEIO_FFMPEG_EXE garderait ce chemin. imageio ne vérifie pas la variable "
        "et l'exécute telle quelle. Après la quarantaine, le lien est cassé "
        "(FileNotFoundError). Un lien vers tonet/../cwd/ffmpeg, tonet pointant "
        "vers /proc/net, ouvre ce même fichier quand le cwd est le clone : "
        "le noyau entre dans /proc/net (self/net), .. retombe dans /proc/self "
        "et cwd ouvre la racine. normpath annule .. avant ce lien. is_file "
        "depuis le dossier du pipeline peut voir un autre ffmpeg, celui de ce "
        "cwd. -encoders et le mux, lancés avec le cwd du clone, réussissent "
        "tant que le fichier est là. Une chaîne hop1 vers hop2 vers /proc/net "
        "fait pareil. ffmpeg_bins peut lui-même pointer vers /proc/self/cwd "
        "ou vers /proc/net/../cwd. is_symlink suit ce dossier et voit un "
        "fichier régulier du cwd du pipeline : is_file est vrai, le dernier "
        "nom n'est pas un lien. Avec le cwd du clone, ce chemin ouvre le "
        "fichier déplacé. save_video imprime error saving et le processus "
        "quitte avec le code 0, sans MP4. "
    )


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


def _wan_config_file(root: Path) -> Path | None:
    """``wgp_config.json`` que WanGP ouvre quand le cwd est le clone.

    ``open("wgp_config.json")`` suit ``/proc/self`` de ce processus. Un lien
    vers ``/proc/self/cwd/secret.json`` n'est pas le fichier que
    ``Path.is_file()`` voit depuis le pipeline : le leurre y est un ``mp4``,
    ou le lien est cassé, alors que le clone a un ``mkv``. ``None`` si le
    clone ne peut pas ouvrir le fichier. WanGP écrit alors la config par
    défaut, dont le conteneur est ``mp4``.
    """
    path = root / "wgp_config.json"
    try:
        absolute = str(path.absolute())
    except OSError:
        return None
    if os.name != "nt" and _path_depends_on_proc_cwd(absolute):
        # realpath réussit même si le dernier nom n'existe pas. WanGP ne
        # lit alors pas de JSON : il écrit la config par défaut (mp4).
        opened = _realpath_from_cwd(path, root)
        if opened is None:
            return None
        try:
            if opened.is_file():
                return opened
        except OSError:
            return None
        return None
    try:
        if path.is_file():
            return path
    except OSError:
        return None
    return None


def _config_where(logical: Path | None, opened: Path | None) -> str:
    """Emplacement cité dans l'erreur, avec le fichier vu par le clone."""
    if logical is None:
        return "wgp_config.json"
    where = str(logical)
    if opened is None or os.name == "nt":
        return where
    try:
        absolute = str(logical.absolute())
    except OSError:
        return where
    if not _path_depends_on_proc_cwd(absolute):
        return where
    return (
        f"{logical} (ouvert depuis le cwd du clone : {opened}). "
        "Path.is_file() depuis le pipeline suit /proc/self de ce processus. "
        "WanGP a pour cwd le clone et lit ce wgp_config.json. "
        "Un video_container mkv ou mov, ou un codec ProRes, n'écrit pas de MP4 "
        "et le CLI sort avec le code 0"
    )


_MODEL_DEF_FOLDERS = ("defaults", "finetunes")
_NON_VIDEO_MODEL_FLAGS = (
    (
        "image_outputs",
        "image_outputs est vrai. WanGP force image_mode à 1 même si la file demande 0 "
        "(validate_settings). La tâche écrit un JPEG et process_tasks_cli quitte avec le code 0, sans MP4.",
    ),
    (
        "audio_only",
        "audio_only est vrai. WanGP écrit un fichier audio et process_tasks_cli quitte avec le code 0, sans MP4.",
    ),
    (
        "mesh_outputs",
        "mesh_outputs est vrai. WanGP écrit un glb ou un ply et process_tasks_cli quitte avec le code 0, sans MP4.",
    ),
    (
        "world_outputs",
        "world_outputs est vrai. Le world_output par défaut est world : WanGP écrit un .wgpworld "
        "et process_tasks_cli quitte avec le code 0, sans MP4.",
    ),
)


# Architectures dont query_model_def (WanGP 18.84) met un drapeau non vidéo
# à vrai sans que le JSON le contienne. init_model_def n'est appelé que
# pour le premier fichier du modèle. Un JSON suivant ne retire le drapeau
# que s'il réécrit la clé.
_HANDLER_FORCED_FLAGS: dict[str, frozenset[str]] = {
    "chrono_edit": frozenset({"image_outputs"}),
    "flux": frozenset({"image_outputs"}),
    "flux2_dev": frozenset({"image_outputs"}),
    "pi_flux2": frozenset({"image_outputs"}),
    "flux2_klein_4b": frozenset({"image_outputs"}),
    "flux2_klein_9b": frozenset({"image_outputs"}),
    "flux_chroma": frozenset({"image_outputs"}),
    "flux_chroma_radiance": frozenset({"image_outputs"}),
    "flux_dev_kontext": frozenset({"image_outputs"}),
    "flux_dev_umo": frozenset({"image_outputs"}),
    "flux_dev_uso": frozenset({"image_outputs"}),
    "flux_schnell": frozenset({"image_outputs"}),
    "flux_dev_kontext_dreamomni2": frozenset({"image_outputs"}),
    "flux3_klein": frozenset({"image_outputs"}),
    "flux3_klein_base": frozenset({"image_outputs"}),
    "fibo2_dev": frozenset({"image_outputs"}),
    "fibo2_edit_turbo": frozenset({"image_outputs"}),
    "hidream_o1": frozenset({"image_outputs"}),
    "hidream_o1_dev": frozenset({"image_outputs"}),
    "ideogram4": frozenset({"image_outputs"}),
    "ideogram4_turbotime": frozenset({"image_outputs"}),
    "krea2_raw": frozenset({"image_outputs"}),
    "krea2_turbo": frozenset({"image_outputs"}),
    "krea2_raw_edit": frozenset({"image_outputs"}),
    "krea2_turbo_edit": frozenset({"image_outputs"}),
    "moss_image": frozenset({"image_outputs"}),
    "ming_image_0_1_design": frozenset({"image_outputs"}),
    "ming_image_0_1_design_layer": frozenset({"image_outputs"}),
    "sensenova_u1_5_8b_mot": frozenset({"image_outputs"}),
    "qwen_image_20B": frozenset({"image_outputs"}),
    "qwen_image_edit_20B": frozenset({"image_outputs"}),
    "qwen_image_edit_plus_20B": frozenset({"image_outputs"}),
    "qwen_image_edit_plus2_20B": frozenset({"image_outputs"}),
    "qwen_image_layered_20B": frozenset({"image_outputs"}),
    "qwen_image_21_7B": frozenset({"image_outputs"}),
    "z_image": frozenset({"image_outputs"}),
    "z_image_base": frozenset({"image_outputs"}),
    "z_image_control": frozenset({"image_outputs"}),
    "z_image_control2": frozenset({"image_outputs"}),
    "z_image_control2_1": frozenset({"image_outputs"}),
    "animate_any_mesh": frozenset({"image_outputs", "mesh_outputs"}),
    "hunyuan3d_21_shape": frozenset({"image_outputs", "mesh_outputs"}),
    "hunyuan3d_21": frozenset({"image_outputs", "mesh_outputs"}),
    "moge2": frozenset({"image_outputs", "mesh_outputs"}),
    "mapanything": frozenset({"image_outputs", "mesh_outputs"}),
    "da3_points": frozenset({"image_outputs", "mesh_outputs"}),
    "trellis2": frozenset({"image_outputs", "mesh_outputs"}),
    "triposg": frozenset({"image_outputs", "mesh_outputs"}),
    "partcrafter": frozenset({"image_outputs", "mesh_outputs"}),
    "geosam2": frozenset({"image_outputs", "mesh_outputs"}),
    "triposplat": frozenset({"image_outputs", "mesh_outputs"}),
    "worldmirror_2": frozenset({"image_outputs", "mesh_outputs"}),
    "evoke_turbo": frozenset({"world_outputs"}),
    "hy_nomas": frozenset({"world_outputs"}),
    "waypoint_1_5": frozenset({"world_outputs"}),
    "waypoint_1_5_360p": frozenset({"world_outputs"}),
    "ace_step_v1": frozenset({"audio_only"}),
    "ace_step_v1_5": frozenset({"audio_only"}),
    "ace_step_v1_5_xl": frozenset({"audio_only"}),
    "auk": frozenset({"audio_only"}),
    "chatterbox": frozenset({"audio_only"}),
    "fish_drama3": frozenset({"audio_only"}),
    "heartmula_oss_3b": frozenset({"audio_only"}),
    "index_tts2": frozenset({"audio_only"}),
    "kugelaudio_0_open": frozenset({"audio_only"}),
    "minimax_music3": frozenset({"audio_only"}),
    "moss_tts_v1_5": frozenset({"audio_only"}),
    "moss_tts_local_v1_5": frozenset({"audio_only"}),
    "omnivoice": frozenset({"audio_only"}),
    "qwen3_tts_customvoice": frozenset({"audio_only"}),
    "qwen3_tts_voicedesign": frozenset({"audio_only"}),
    "qwen3_tts_base": frozenset({"audio_only"}),
    "stable_audio3_small": frozenset({"audio_only"}),
    "stable_audio3_medium": frozenset({"audio_only"}),
    "yue2": frozenset({"audio_only"}),
    "yue2_hum": frozenset({"audio_only"}),
    "scenema_audio": frozenset({"audio_only"}),
    "dramabox_audio": frozenset({"audio_only"}),
}

# Architectures dont ``fix_settings`` (architecture finale, après
# ``clean_settings``) remet ``image_mode`` 0 à 1. ``update_default_settings``
# ne suffit pas : la file envoie 0 et gagne. ``setdefault`` non plus.
_FIX_SETTINGS_FORCED_IMAGE = frozenset({"qwen_image_21_7B"})

# ``query_model_def`` du premier JSON injecte ``model_modes.image_modes``
# ``[2]`` (WanGP 18.84, commit 97845dc88f). ``init_model_def`` n'est
# appelé que pour ce fichier. ``default_model_def.update`` ne retire
# ``model_modes`` que si un JSON contient la clé. Un fichier suivant
# qui passe à ``chrono_edit`` sans la réécrire laisse ``[2]``.
# ``validate_settings`` efface alors ``model_mode`` 2 (la file demande
# ``image_mode`` 0) et Chrono Edit écrit un JPEG.
# Flux : ``supports_inpaint`` vaut vrai pour ``flux``, ``flux_schnell``,
# ``flux_dev_kontext`` et ``flux2`` (``flux2_dev``, ``flux2_klein_4b``,
# ``flux2_klein_9b``, ``pi_flux2``). ``pi_flux2`` n'écrit pas
# ``model_modes``. ``flux_chroma``, ``flux_chroma_radiance``,
# ``flux_dev_umo``, ``flux_dev_uso`` et ``flux_dev_kontext_dreamomni2``
# non plus. ``qwen_image_layered_20B`` non plus.
# ``test_class_i2v`` de models/wan/wan_handler.py (WanGP 18.84).
_WAN_I2V = frozenset(
    {
        "i2v",
        "i2v_2_2",
        "fun_inp_1.3B",
        "fun_inp",
        "flf2v_720p",
        "fantasy",
        "multitalk",
        "infinitetalk",
        "i2v_2_2_multitalk",
        "animate",
        "animate2",
        "chrono_edit",
        "steadydancer",
        "wanmove",
        "scail",
        "scail2_14B",
        "scail2_1.3B",
        "i2v_2_2_svi2pro",
        "fun_camera_1.3B",
        "uni3c",
    }
)

# Architectures dont query_model_def pose ``black_frame``. Une image noire
# remplace alors l'image de départ manquante : la validation ne saute pas.
_WAN_BLACK_FRAME = frozenset(
    {
        "i2v",
        "flf2v_720p",
        "i2v_2_2",
        "i2v_2_2_multitalk",
        "i2v_2_2_svi2pro",
    }
)

_WAN_VACE = frozenset(
    {
        "vace_14B",
        "vace_14B_2_2",
        "vace_1.3B",
        "vace_multitalk_14B",
        "vace_standin_14B",
        "vace_lynx_14B",
        "vace_ditto_14B",
    }
)

# ``test_class_t2v`` de models/wan/wan_handler.py (WanGP 18.84).
# ``t2v_1.3B`` n'y est pas : l'architecture est passée telle quelle.
_WAN_T2V_CLASS = frozenset(
    {
        "t2v",
        "t2v_2_2",
        "shotplan_t2v",
        "shotplan_t2v_2_2",
        "alpha",
        "alpha2",
        "lynx",
        "vista4d",
        "bernini",
        "bernini_1.3B",
    }
)

_WAN_BERNINI = frozenset({"bernini", "bernini_1.3B"})

_WAN_MULTITALK = frozenset(
    {
        "multitalk",
        "vace_multitalk_14B",
        "i2v_2_2_multitalk",
        "infinitetalk",
    }
)

_WAN_ARCHITECTURES = _WAN_I2V | _WAN_VACE | frozenset(
    {
        "t2v",
        "t2v_1.3B",
        "t2v_2_2",
        "alpha",
        "alpha2",
        "alpha_lynx",
        "lynx",
        "lynx_lite",
        "vista4d",
        "bernini",
        "bernini_1.3B",
        "shotplan_t2v",
        "shotplan_t2v_2_2",
        "ti2v_2_2",
        "lucy_edit",
        "kiwi_edit",
        "phantom_1.3B",
        "phantom_14B",
        "recam_1.3B",
        "standin",
        "mocha",
    }
)


def _wan_injected_image_prompt_types(architecture: str) -> str | None:
    """``image_prompt_types_allowed`` posé par ``wan_handler.query_model_def``.

    ``None`` si l'architecture n'est pas celle de ce handler : la clé
    manque alors vraiment, et ``update_default_settings`` lève KeyError
    au lieu de sauter la tâche.
    """
    if architecture not in _WAN_ARCHITECTURES:
        return None
    if architecture in {"bernini", "bernini_1.3B"}:
        return "T"
    if architecture == "animate2":
        return "TSVL"
    if architecture in _WAN_VACE or architecture in {
        "animate",
        "t2v",
        "t2v_2_2",
        "shotplan_t2v",
        "shotplan_t2v_2_2",
        "lynx",
    }:
        return "TVL"
    if architecture == "infinitetalk":
        return "TSVL"
    if architecture == "ti2v_2_2":
        return "TSVL"
    if architecture in {"lucy_edit", "kiwi_edit"}:
        return "TVL"
    if architecture in {"scail2_14B", "scail2_1.3B"}:
        return "TSV"
    if architecture == "fun_camera_1.3B":
        return "TS"
    if architecture == "uni3c":
        return "S"
    if architecture in _WAN_MULTITALK or architecture in {"fantasy", "steadydancer", "scail"}:
        return "SVL"
    if architecture in _WAN_I2V:
        return "SEVL"
    return ""


_QUERY_INJECTED_IMAGE_MODES = frozenset(
    {
        "qwen_image_20B",
        "qwen_image_edit_20B",
        "qwen_image_edit_plus_20B",
        "qwen_image_edit_plus2_20B",
        "qwen_image_21_7B",
        "flux",
        "flux_schnell",
        "flux_dev_kontext",
        "flux2_dev",
        "flux2_klein_4b",
        "flux2_klein_9b",
        "krea2_raw",
        "krea2_turbo",
        "krea2_raw_edit",
        "krea2_turbo_edit",
    }
)


def _architecture_name(model: dict[str, object], model_type: str) -> str:
    """Architecture vue par ``init_model_def`` quand la clé manque.

    Le nom du fichier (``t2v_1.3B``) sert de repli, comme
    ``get_base_model_type`` quand le modèle n'est pas encore enregistré
    et qu'il fait partie des handlers.
    """
    raw = model.get("architecture")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return model_type


def _file_opened_from_clone(path: Path, root: Path) -> Path | None:
    """Fichier que ``open`` atteint quand le cwd est le clone.

    ``None`` si ce processus ne peut pas ouvrir un fichier. Un lien vers
    ``/proc/self`` n'a pas la même cible depuis le pipeline.
    """
    try:
        absolute = str(path.absolute())
    except OSError:
        return None
    if os.name != "nt" and _path_depends_on_proc_cwd(absolute):
        opened = _realpath_from_cwd(path, root)
        if opened is None:
            return None
        try:
            if opened.is_file():
                return opened
        except OSError:
            return None
        return None
    try:
        if path.is_file():
            return path
    except OSError:
        return None
    return None


def _directory_opened_from_clone(path: Path, root: Path) -> Path | None:
    """Dossier que ``glob`` parcourt quand le cwd est le clone.

    ``None`` si ce processus n'ouvre pas un dossier. ``defaults`` vers
    ``/proc/self/cwd`` liste la racine du clone, pas le cwd du pipeline.
    """
    try:
        absolute = str(path.absolute())
    except OSError:
        return None
    if os.name != "nt" and _path_depends_on_proc_cwd(absolute):
        opened = _realpath_from_cwd(path, root)
        if opened is None:
            return None
        try:
            if opened.is_dir():
                return opened
        except OSError:
            return None
        return None
    try:
        if path.is_dir():
            return path
    except OSError:
        return None
    return None


def _model_def_where(logical: Path, opened: Path) -> str:
    """Emplacement cité, avec le JSON vu par le clone si ``/proc/self`` s'en mêle."""
    where = str(logical)
    try:
        absolute = str(logical.absolute())
    except OSError:
        return where
    if os.name == "nt" or not _path_depends_on_proc_cwd(absolute):
        return where
    return (
        f"{logical} (ouvert depuis le cwd du clone : {opened}). "
        "iterdir et Path.is_file() depuis le pipeline suivent /proc/self de ce processus. "
        "WanGP a pour cwd le clone et lit ce JSON. "
        "Un image_outputs vrai force image_mode à 1 : un JPEG est écrit "
        "et le CLI sort avec le code 0"
    )


# Plugins toujours pris par discover_plugin_model_extensions, même sans enabled_plugins.
_SYSTEM_PLUGINS = (
    "video_mask_creator",
    "guides",
    "configuration",
    "plugin_manager",
    "about",
)


def _clone_abspath(text: str, root: Path) -> str:
    """``os.path.abspath`` quand le cwd est le clone, pas celui du pipeline."""
    normalized = text.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)
    base = _probe_cwd(root) or str(root.absolute())
    return os.path.normpath(os.path.join(base, normalized))


def _plugin_metadata_text(value: object) -> str:
    """Champ ``defaults`` ou ``profiles`` tel que le normalise PluginManager."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _plugin_string_list(value: object) -> list[str]:
    """``model_handlers`` : une chaîne ou une liste de chaînes, comme WanGP."""
    if isinstance(value, str):
        values: list[object] = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text:
            cleaned.append(text)
    return cleaned


def _plugin_handler_path(plugin_id: str, handler_path: str) -> str | None:
    """Chemin de module accepté par ``_resolve_plugin_module_path``.

    Le module n'a pas besoin d'exister au moment du glob des définitions.
    Un texte vide est ignoré. Un chemin relatif (``.handler``) est préfixé
    par l'identifiant du plugin.
    """
    text = handler_path.strip()
    if not text:
        return None
    is_relative = text.startswith(".")
    text = text.replace("\\", "/")
    if text.endswith(".py"):
        text = text[:-3]
    if is_relative:
        text = text.lstrip(".").lstrip("/")
    text = text.strip("/").replace("/", ".").strip(".")
    if not text:
        return None
    if is_relative:
        return f"{plugin_id}.{text}"
    return text


def _plugin_folder_path(plugin_id: str, path_value: str, root: Path) -> str:
    """Dossier ``defaults`` ou ``profiles`` résolu comme WanGP.

    Un chemin qui commence par ``.`` est relatif au plugin. Les autres
    chemins relatifs le sont au cwd du clone. ``os.path.abspath`` ne suit
    pas ``/proc/self`` : le texte ``/proc/self/cwd/bag`` reste tel quel,
    et ``glob`` l'ouvre plus tard avec le cwd du processus.
    """
    path_text = path_value.strip()
    if not path_text:
        return ""
    path_text = path_text.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(path_text):
        return os.path.normpath(path_text)
    if path_text.startswith("."):
        return _clone_abspath(os.path.join("plugins", plugin_id, path_text), root)
    return _clone_abspath(path_text, root)


def _enabled_plugin_ids(root: Path) -> set[str]:
    """Plugins dont WanGP charge les définitions au démarrage.

    ``initialize_plugins`` ajoute ``motion_designer`` après
    ``refresh_model_defs``. Seuls ``enabled_plugins`` déjà dans
    ``wgp_config.json``, plus les plugins système, comptent ici.
    Le JSON est celui que ``open`` atteint quand le cwd est le clone.
    """
    enabled = set(_SYSTEM_PLUGINS)
    opened = _wan_config_file(root)
    if opened is None:
        return enabled
    try:
        payload = json.loads(opened.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return enabled
    if not isinstance(payload, dict):
        return enabled
    raw = payload.get("enabled_plugins") or []
    if not isinstance(raw, list):
        return enabled
    for item in raw:
        if isinstance(item, str) and item:
            enabled.add(item)
    return enabled


def _plugin_metadata(root: Path, plugin_id: str) -> dict[str, object] | None:
    """``plugin_info.json`` ouvert depuis le cwd du clone.

    Un lien vers ``/proc/self/cwd/meta.json`` n'est pas le fichier que
    ``Path.is_file()`` lit depuis le pipeline. ``None`` si WanGP ignore
    le plugin (fichier absent ou JSON inutilisable).
    """
    logical = root / "plugins" / plugin_id / "plugin_info.json"
    opened = _file_opened_from_clone(logical, root)
    if opened is None:
        return None
    try:
        payload = json.loads(opened.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _discovered_plugin_ids(root: Path) -> list[str]:
    """Dossiers de ``plugins/`` que ``discover_plugins`` retiendrait.

    ``os.listdir("plugins")`` suit le cwd du clone. ``plugins`` vers
    ``/proc/self/cwd`` ou ``/proc/self/cwd/only`` ne liste pas le dossier
    du pipeline.
    """
    logical = root / "plugins"
    opened = _directory_opened_from_clone(logical, root)
    if opened is None:
        return []
    try:
        names = [item.name for item in opened.iterdir()]
    except OSError:
        return []
    found: list[str] = []
    for name in names:
        if name.endswith(".bak"):
            continue
        # Le dossier déjà ouvert depuis le clone suffit pour un fichier
        # régulier. Un enfant qui est lui-même un lien vers /proc/self
        # se résout encore par le chemin logique : is_dir() ici suivrait
        # le cwd du pipeline.
        candidate = opened / name
        try:
            candidate_path = str(candidate.absolute())
        except OSError:
            continue
        if os.name != "nt" and _path_depends_on_proc_cwd(candidate_path):
            plugin = logical / name
            if _directory_opened_from_clone(plugin, root) is None:
                continue
            has_code = _file_opened_from_clone(plugin / "plugin.py", root) is not None
            has_package = _file_opened_from_clone(plugin / "__init__.py", root) is not None
        else:
            try:
                if not candidate.is_dir():
                    continue
                has_code = (candidate / "plugin.py").is_file()
                has_package = (candidate / "__init__.py").is_file()
            except OSError:
                continue
        if has_code or has_package:
            found.append(name)
    return sorted(found)


def _proc_open_note(path: Path, root: Path, what: str) -> str:
    """Phrase si ``path`` n'ouvre pas la même cible depuis le pipeline."""
    try:
        absolute = str(path.absolute())
    except OSError:
        return ""
    if os.name == "nt" or not _path_depends_on_proc_cwd(absolute):
        return ""
    opened = _realpath_from_cwd(path, root)
    return (
        f"{what} {path} suit /proc/self (ouvert depuis le cwd du clone : {opened}). "
        "Path.is_file() ou iterdir depuis le pipeline suivent /proc/self de ce processus."
    )


def _plugin_model_default_dirs(root: Path) -> list[tuple[str, str]]:
    """Dossiers ``defaults`` des plugins, en chemins que ``glob`` reçoit.

    Le chemin est absolu, comme ``_resolve_plugin_folder_path``. Un plugin
    sans ``model_handlers``, sans ``defaults`` ou sans ``profiles`` est
    ignoré : WanGP ne l'ajoute pas à ``model_definition_sources``. La
    note décrit un lien ``plugins`` ou ``plugin_info.json`` vers
    ``/proc/self``, quand le JSON du modèle lui-même n'est pas ce lien.
    """
    enabled = _enabled_plugin_ids(root)
    folders: list[tuple[str, str]] = []
    plugins_note = _proc_open_note(root / "plugins", root, "Le dossier plugins")
    for plugin_id in _discovered_plugin_ids(root):
        if plugin_id not in enabled:
            continue
        metadata = _plugin_metadata(root, plugin_id)
        if metadata is None:
            continue
        handlers = _plugin_string_list(metadata.get("model_handlers"))
        if not any(_plugin_handler_path(plugin_id, item) for item in handlers):
            continue
        defaults = _plugin_folder_path(plugin_id, _plugin_metadata_text(metadata.get("defaults")), root)
        profiles = _plugin_folder_path(plugin_id, _plugin_metadata_text(metadata.get("profiles")), root)
        if not defaults or not profiles:
            continue
        info_note = _proc_open_note(
            root / "plugins" / plugin_id / "plugin_info.json",
            root,
            "plugin_info.json",
        )
        note = " ".join(part for part in (plugins_note, info_note) if part)
        folders.append((defaults, note))
    return folders


def _json_model_files(
    logical_dir: Path,
    root: Path,
    model_type: str,
    sort_dir: str,
    note: str = "",
) -> list[tuple[str, str, Path, Path, str]]:
    """JSON du modèle dans un dossier, avec la clé de tri de WanGP.

    ``glob`` ne descend pas. Seul ``<modèle>.json`` change ``image_mode``.
    """
    opened_dir = _directory_opened_from_clone(logical_dir, root)
    if opened_dir is None:
        return []
    try:
        names = [item.name for item in opened_dir.iterdir()]
    except OSError:
        return []
    found: list[tuple[str, str, Path, Path, str]] = []
    prefix = sort_dir.rstrip("/\\")
    for name in names:
        if not name.endswith(".json") or Path(name).stem != model_type:
            continue
        logical = logical_dir / name
        opened = _file_opened_from_clone(logical, root)
        if opened is None:
            continue
        found.append((f"{prefix}/{name}", f"{prefix}/{name}", logical, opened, note))
    return found


def _model_definition_entries(root: Path, model_type: str) -> list[tuple[str, Path, Path, str]]:
    """``(libellé, chemin logique, fichier ouvert, note /proc)`` dans l'ordre de WanGP.

    ``refresh_model_defs`` réunit ``defaults/*.json``, les ``defaults`` des
    plugins et ``finetunes/*.json``, puis trie ces chemins. Les dossiers
    de plugin sont absolus (``os.path.abspath``), donc ils passent avant
    ``defaults/`` et ``finetunes/``. Le fichier suivant met à jour le
    précédent : une clé absente ne retire pas ``image_outputs``.
    Seul le modèle demandé est lu.
    """
    found: list[tuple[str, str, Path, Path, str]] = []
    for folder in _MODEL_DEF_FOLDERS:
        found.extend(_json_model_files(root / folder, root, model_type, folder))
    for folder, note in _plugin_model_default_dirs(root):
        found.extend(_json_model_files(Path(folder), root, model_type, folder, note))
    found.sort(key=lambda item: item[0])
    return [(label, logical, opened, note) for _sort_key, label, logical, opened, note in found]


def _effective_model_mode(settings: dict[str, object], queue_model_mode: object) -> object:
    """``model_mode`` lu par ``validate_generative_settings``.

    ``models/_settings.json`` part de ``null``. Les clés hors ``model``
    deviennent ``model_def["settings"]`` et remplacent ce défaut.
    ``clean_settings`` ne laisse une valeur nulle de la file écraser ce
    réglage que s'il n'est pas déjà là : il l'est toujours.
    """
    mode: object = None
    if "model_mode" in settings:
        mode = settings["model_mode"]
    if queue_model_mode is _QUEUE_MODEL_MODE_UNSET or queue_model_mode is None:
        return mode
    return queue_model_mode


def _chrono_model_mode_cleared(model_modes: object, model_mode: object) -> bool:
    """``validate_settings`` remet ``model_mode`` à None avant le handler.

    La liste de départ est ``[0, 1, 2]``. ``model_modes.image_modes``, si la
    clé est présente, la remplace. La file de ce pipeline demande
    ``image_mode`` 0. S'il n'est pas dans cette liste et que ``model_mode``
    n'est pas déjà None, WanGP l'efface. ``chrono_edit`` fait alors
    ``image_mode = 1`` : un JPEG, et ``process_tasks_cli`` sort 0.
    Une clé absente laisse ``[0, 1, 2]``. Un ``image_modes`` qui n'est pas
    un conteneur fait lever ``TypeError`` : le processus ne sort pas 0.
    """
    if model_mode is None or not isinstance(model_modes, dict):
        return False
    if "image_modes" not in model_modes:
        return False
    visibility = model_modes["image_modes"]
    try:
        return 0 not in visibility
    except TypeError:
        return False


def _require_chrono_final_architecture(
    merged: dict[str, object],
    settings: dict[str, object],
    where: str,
    model_modes_where: str,
    queue_model_mode: object,
    model_type: str,
    first_architecture: str | None = None,
    first_where: str = "",
) -> None:
    """Refuse ``chrono_edit`` comme architecture finale hors ``model_mode`` 2.

    ``init_model_def`` ne voit que le premier JSON. Un finetune peut
    pourtant remplacer ``architecture`` sans rappeler ``query_model_def`` :
    ``image_outputs`` reste faux. ``get_base_model_type`` renvoie cette
    architecture finale. ``validate_generative_settings`` fait alors
    ``image_mode = 0 if model_mode == 2 else 1`` et raccourcit
    ``video_length``. Le défaut ``model_mode`` est ``null``, donc un JPEG
    est écrit et ``process_tasks_cli`` sort avec le code 0.
    Un ``model_mode`` 2 est annulé avant ce test si ``model_modes.image_modes``
    ne contient pas 0 : le handler voit None et écrit quand même le JPEG.
    La clé peut manquer de tous les JSON et rester injectée par
    ``query_model_def`` du premier fichier (``image_modes`` ``[2]``).
    """
    if merged.get("architecture") != "chrono_edit":
        return
    mode = _effective_model_mode(settings, queue_model_mode)
    injected = (
        not model_modes_where
        and first_architecture in _QUERY_INJECTED_IMAGE_MODES
    )
    modes = {"image_modes": [2]} if injected else merged.get("model_modes")
    if mode == 2 and _chrono_model_mode_cleared(modes, mode):
        if injected:
            shown = first_where or where
            raise Wan2GPError(
                "image_mode est forcé à 1. query_model_def du premier JSON injecte "
                f"model_modes.image_modes [2] pour l'architecture {first_architecture} "
                f"(init_model_def, seulement ce premier fichier : {shown}). "
                "Aucun JSON n'écrit model_modes : la clé reste absente des fichiers, "
                "mais WanGP conserve l'injection. default_model_def.update ne la retire "
                "que si un JSON contient la clé. Un fichier suivant passe l'architecture "
                "à chrono_edit, met image_outputs à faux et model_mode à 2. "
                "validate_settings voit image_modes [2]. La file demande image_mode 0, "
                f"qui n'est pas dans cette liste : model_mode={mode!r} devient None "
                "avant validate_generative_settings (models/wan/wan_handler.py). "
                "None n'est pas 2 : image_mode devient 1, un JPEG est écrit et "
                "process_tasks_cli quitte avec le code 0, sans MP4. "
                f"Architecture finale lue dans {where}. "
                f"Écrire model_modes dans un JSON de {model_type} "
                "(avec 0 dans image_modes, ou sans la clé image_modes), "
                "ou retirer chrono_edit."
            )
        shown = model_modes_where or where
        raise Wan2GPError(
            "image_mode est forcé à 1. validate_settings annule model_mode avant "
            "validate_generative_settings (models/wan/wan_handler.py). "
            "model_modes.image_modes ne contient pas 0, et la file demande image_mode 0. "
            f"model_mode={mode!r} devient None. Pour l'architecture finale chrono_edit, "
            "None n'est pas 2 : image_mode devient 1, un JPEG est écrit et "
            "process_tasks_cli quitte avec le code 0, sans MP4. "
            "image_outputs peut rester faux. "
            f"model_modes lu dans {shown}. "
            f"Remettre 0 dans image_modes, ou retirer chrono_edit de la définition de {model_type}. "
            "Sans la clé image_modes, la liste reste [0, 1, 2] et model_mode 2 écrit une vidéo."
        )
    if mode == 2:
        return
    raise Wan2GPError(
        "image_mode est forcé à 1. validate_generative_settings (models/wan/wan_handler.py) "
        "lit l'architecture finale chrono_edit, pas seulement celle du premier JSON. "
        f"model_mode={mode!r} n'est pas 2 : image_mode devient 1, video_length passe à 5 ou 29, "
        "un JPEG est écrit et process_tasks_cli quitte avec le code 0, sans MP4. "
        "query_model_def n'est pas rappelé, donc image_outputs peut rester faux. "
        f"Architecture lue dans {where}. "
        f"Retirer chrono_edit de la définition de {model_type}. "
        "model_mode 2 garde image_mode à 0 et écrit une vidéo, "
        "sauf si model_modes.image_modes ne contient pas 0."
    )


def _require_fix_settings_image_mode(
    merged: dict[str, object],
    where: str,
    model_type: str,
) -> None:
    """Refuse ``qwen_image_21_7B`` comme architecture finale.

    ``clean_settings`` applique d'abord ``image_mode`` 0 de la file, puis
    ``fix_settings`` du handler de l'architecture finale. Pour
    ``qwen_image_21_7B``, ``models/qwen/qwen_handler.py`` délègue à
    ``models/qwen21/qwen21_handler.py``, qui remplace 0, ``None`` et
    ``"0"`` par 1. ``image_outputs`` peut rester faux : ``query_model_def``
    n'est appelé que sur le premier JSON. Un JPEG est écrit et
    ``process_tasks_cli`` sort avec le code 0.
    """
    architecture = merged.get("architecture")
    if architecture not in _FIX_SETTINGS_FORCED_IMAGE:
        return
    shown = where or model_type
    raise Wan2GPError(
        "image_mode est forcé à 1. fix_settings (models/qwen/qwen_handler.py, "
        "qui appelle models/qwen21/qwen21_handler.py) lit l'architecture finale "
        f"{architecture}, après que clean_settings a appliqué image_mode 0 de la file. "
        "0, None et \"0\" redeviennent 1. Un JPEG est écrit et "
        "process_tasks_cli quitte avec le code 0, sans MP4. "
        "query_model_def n'est pas rappelé : image_outputs peut rester faux ou absent. "
        f"Architecture lue dans {shown}. "
        f"Retirer {architecture} de la définition de {model_type}."
    )


def _model_modes_abort_before_start_image(
    model_modes: object,
    model_mode: object,
) -> bool:
    """``image_mode not in image_modes`` lève TypeError avant l'image de départ.

    La validation ne revient pas : le processus ne sort pas 0. Ce n'est pas
    le saut silencieux d'une tâche sans MP4.
    """
    if model_mode is None or not isinstance(model_modes, dict):
        return False
    if "image_modes" not in model_modes:
        return False
    try:
        0 in model_modes["image_modes"]
    except TypeError:
        return True
    return False


def _require_i2v_start_image(
    merged: dict[str, object],
    settings: dict[str, object],
    queue_model_mode: object,
    queue_image_prompt_type: object,
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    prompt_types_explicit: bool,
    black_frame_explicit: bool,
) -> None:
    """Refuse un I2V final dont la validation saute faute d'image de départ.

    ``update_default_settings`` de ``models/wan/wan_handler.py`` pose
    ``image_prompt_type`` à ``S`` quand l'architecture finale est I2V et
    que ``image_prompt_types_allowed`` contient ``S``. La file de ce
    pipeline n'envoie pas cette clé, ni d'image. ``validate_settings``
    renvoie une erreur. ``process_tasks_cli`` compte la tâche sautée
    (``completed == total - skipped``) et sort 0, sans MP4.
    ``query_model_def`` du premier JSON injecte la chaîne si elle manque
    (``SEVL`` pour ``fun_inp_1.3B`` et ``chrono_edit``). ``black_frame``
    fabrique une image noire à la place : ``i2v`` et ``flf2v_720p`` ne
    sont pas sautés. Un ``image_modes`` non itérable lève TypeError avant
    ce contrôle.
    """
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or architecture not in _WAN_I2V:
        return
    if prompt_types_explicit:
        allowed: object = merged.get("image_prompt_types_allowed")
        allowed_note = ""
    else:
        if first_architecture is None:
            return
        injected = _wan_injected_image_prompt_types(first_architecture)
        if injected is None:
            return
        allowed = injected
        allowed_note = (
            f" La clé est absente du JSON. query_model_def({first_architecture!r}) "
            f"l'injecte ({injected!r}) dans init_model_def, seulement pour le premier "
            f"fichier ({first_where})."
        )
    if not isinstance(allowed, str) or "S" not in allowed:
        return
    if (
        queue_image_prompt_type is not _QUEUE_MODEL_MODE_UNSET
        and queue_image_prompt_type is not None
    ):
        prompt_type = queue_image_prompt_type
    elif "image_prompt_type" in settings:
        prompt_type = settings["image_prompt_type"]
    else:
        prompt_type = "S"
    if not isinstance(prompt_type, str):
        return
    if "S" not in prompt_type:
        return
    if black_frame_explicit:
        black_frame = bool(merged.get("black_frame"))
    else:
        black_frame = first_architecture in _WAN_BLACK_FRAME
    if black_frame and "S" in allowed:
        return
    mode = _effective_model_mode(settings, queue_model_mode)
    modes = merged.get("model_modes")
    if _model_modes_abort_before_start_image(modes, mode):
        return
    shown = architecture_where or first_where or model_type
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. update_default_settings "
        "(models/wan/wan_handler.py) pose image_prompt_type à S pour "
        f"l'architecture finale {architecture}, parce que image_prompt_types_allowed "
        f"({allowed!r}) contient S. La file n'envoie pas d'image de départ. "
        "validate_settings renvoie une erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        "black_frame n'est pas vrai : aucune image noire ne remplace le fichier manquant."
        f"{allowed_note} "
        f"Architecture lue dans {shown}. "
        f"Retirer {architecture}, écrire image_prompt_types_allowed sans S, "
        "ou poser image_prompt_type sans S dans les réglages du JSON."
    )


# Architectures dont query_model_def pose un guide_custom_choices non nul.
# fix_settings ne retire alors ni I ni K de video_prompt_type (WanGP 18.84).
_WAN_GUIDE_CUSTOM = frozenset(
    {
        "i2v_2_2",
        "i2v_2_2_multitalk",
        "i2v_2_2_svi2pro",
        "t2v",
        "t2v_2_2",
        "shotplan_t2v",
        "shotplan_t2v_2_2",
        "lynx",
        "vista4d",
        "bernini",
        "bernini_1.3B",
        "steadydancer",
        "scail",
        "scail2_14B",
        "scail2_1.3B",
        "infinitetalk",
        "kiwi_edit",
        "animate2",
        "animate",
        "lynx_lite",
        "alpha_lynx",
        "mocha",
    }
)

_WAN_ONE_IMAGE_REF = frozenset({"standin", "lynx", "lynx_lite", "alpha_lynx"})


def _kiwi_embedder_from_urls(model: dict[str, object]) -> bool:
    """Variante Kiwi lue par ``detect_kiwi_variant`` sur le premier JSON.

    ``instruct_only`` dans l'URL désactive l'embedder. Le défaut
    ``instruct_reference`` et ``reference_only`` le laissent actif.
    """
    urls = model.get("URLs", [])
    primary = ""
    if isinstance(urls, list) and urls:
        primary = str(urls[0])
    elif isinstance(urls, str):
        primary = urls
    if "instruct_only" in Path(primary).name.lower():
        return False
    return True


def _factory_video_prompt(architecture: str, kiwi_embedder: bool) -> str:
    """``video_prompt_type`` posé par ``wan_handler.update_default_settings``."""
    if architecture == "infinitetalk":
        return "0KI"
    if architecture == "standin":
        return "I"
    if architecture in {"lynx_lite", "lynx", "alpha_lynx", "phantom_1.3B", "phantom_14B"}:
        return "I"
    if architecture == "kiwi_edit":
        return "UVI" if kiwi_embedder else "UV"
    if architecture in {"bernini", "bernini_1.3B"}:
        return "VI"
    if architecture in {"recam_1.3B", "lucy_edit", "animate2"}:
        return "UV"
    if architecture == "animate":
        return "PVBKI"
    if architecture == "vace_ditto_14B":
        return "V"
    if architecture == "mocha":
        return "VAI"
    if architecture == "steadydancer":
        return "VA"
    if architecture == "scail":
        return "V#1#"
    if architecture in {"scail2_14B", "scail2_1.3B"}:
        return "V1"
    return ""


def _factory_audio_prompt(architecture: str) -> str:
    """``audio_prompt_type`` qui contient ``A`` dans ``update_default_settings``."""
    if architecture in {"fantasy", "multitalk"}:
        return "A"
    return ""


def _injected_image_ref(architecture: str) -> tuple[str, str, bool]:
    """Forme de ``image_ref_choices`` après ``query_model_def`` du premier JSON.

    ``absent`` : la clé manque, ``fix_settings`` retire ``I`` et ``K``.
    ``choices`` : le premier libellé et le fait qu'il n'y en ait qu'un.
    ``phantom`` commence par ``I``. ``vace_ditto_14B`` efface la clé VACE.
    """
    if architecture in {"phantom_1.3B", "phantom_14B"}:
        return ("choices", "I", True)
    if architecture == "vace_ditto_14B":
        return ("absent", "", False)
    if architecture in _WAN_VACE or architecture in {
        "standin",
        "lynx",
        "lynx_lite",
        "alpha_lynx",
        "bernini",
        "bernini_1.3B",
        "animate2",
        "i2v_2_2_svi2pro",
        "scail2_14B",
        "scail2_1.3B",
    }:
        return ("choices", "", False)
    return ("absent", "", False)


def _explicit_image_ref(value: object) -> tuple[str, str, bool] | None:
    """Forme écrite dans le JSON. ``None`` si ``fix_settings`` lève avant le saut."""
    if not isinstance(value, dict):
        return None
    choices = value.get("choices", [])
    if not isinstance(choices, list):
        return None
    if len(choices) == 0:
        return ("absent", "", False)
    first_item = choices[0]
    if not isinstance(first_item, (list, tuple)) or len(first_item) < 2:
        return ("choices", "", len(choices) == 1)
    first = first_item[1]
    if not isinstance(first, str):
        return None
    return ("choices", first, len(choices) == 1)


def _adjust_video_prompt(video: str, guide_present: bool, ref: tuple[str, str, bool]) -> str:
    """Lettres de ``video_prompt_type`` après le filtre IK de ``fix_settings``."""
    if guide_present:
        return video
    kind, first, single = ref
    if kind == "absent":
        return video.replace("I", "").replace("K", "")
    if "I" in first and "I" not in video:
        video += "I"
    if single and "K" in first and "K" not in video:
        video += "K"
    return video


def _queue_or_settings(queue: object, settings: dict[str, object], key: str, factory: object) -> object:
    """Valeur vue par ``clean_settings`` : la file, puis le JSON, puis l'usine.

    Une valeur nulle de la file ne remplace pas une clé déjà présente.
    ``get_factory_settings`` part de ``models/_settings.json``, qui contient
    ces clés : null en file laisse donc l'usine ou le JSON.
    """
    if queue is not _QUEUE_MODEL_MODE_UNSET and queue is not None:
        return queue
    if key in settings:
        return settings[key]
    return factory


def _reference_missing(value: object) -> bool | None:
    """``True`` si l'image de référence manque. ``None`` si le test lèverait."""
    if value is None:
        return True
    if isinstance(value, str):
        return len(value) == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return None


def _resolved_prompt_letters(
    merged: dict[str, object],
    settings: dict[str, object],
    first_architecture: str | None,
    first_model: dict[str, object],
    image_ref_explicit: bool,
    guide_custom_explicit: bool,
    kiwi_ref_explicit: bool,
    queue_video_prompt_type: object,
    queue_audio_prompt_type: object,
) -> tuple[str, str, str, tuple[str, str, bool]] | None:
    """Lettres vues par ``validate_settings`` après ``fix_settings``.

    ``None`` si l'architecture ou une chaîne n'est pas du texte : le
    processus lève alors avant de compter la tâche comme sautée.
    Le troisième terme est la chaîne avant le filtre IK, le quatrième
    la forme de ``image_ref_choices``.
    """
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        return None
    if kiwi_ref_explicit:
        kiwi_embedder = bool(merged.get("kiwi_ref_embedder"))
    elif first_architecture == "kiwi_edit":
        kiwi_embedder = _kiwi_embedder_from_urls(first_model)
    else:
        kiwi_embedder = True
    video_raw = _queue_or_settings(
        queue_video_prompt_type,
        settings,
        "video_prompt_type",
        _factory_video_prompt(architecture, kiwi_embedder),
    )
    audio_raw = _queue_or_settings(
        queue_audio_prompt_type,
        settings,
        "audio_prompt_type",
        _factory_audio_prompt(architecture),
    )
    if not isinstance(video_raw, str) or not isinstance(audio_raw, str):
        return None
    if guide_custom_explicit:
        guide_present = merged.get("guide_custom_choices") is not None
    else:
        guide_present = first_architecture in _WAN_GUIDE_CUSTOM
    if image_ref_explicit:
        ref = _explicit_image_ref(merged.get("image_ref_choices"))
        if ref is None and not guide_present:
            return None
        if ref is None:
            ref = ("absent", "", False)
    elif first_architecture is None:
        ref = ("absent", "", False)
    else:
        ref = _injected_image_ref(first_architecture)
    video = _adjust_video_prompt(video_raw, guide_present, ref)
    return video, audio_raw, str(video_raw), ref


def _custom_guide_visible(guide: dict[str, object], video: str, audio: str) -> bool:
    """``custom_setting_visible`` de WanGP 18.84 pour le guide personnalisé."""
    video_flags = str(guide.get("video_prompt_type") or "")
    audio_flags = str(guide.get("audio_prompt_type") or "")
    video_absent = str(guide.get("video_prompt_type_not") or "")
    audio_absent = str(guide.get("audio_prompt_type_not") or "")
    if video_absent and any(letter in video for letter in video_absent):
        return False
    if audio_absent and any(letter in audio for letter in audio_absent):
        return False
    if not video_flags and not audio_flags:
        return True
    return any(letter in video for letter in video_flags) or any(letter in audio for letter in audio_flags)


def _effective_custom_guide(
    merged: dict[str, object],
    custom_guide_explicit: bool,
    first_architecture: str | None,
) -> dict[str, object] | None:
    """Définition de ``custom_guide`` après ``init_model_def``.

    Seul le premier JSON appelle ``query_model_def``. ``wanmove`` y pose
    un fichier ``.npy`` obligatoire. Un JSON suivant qui change
    l'architecture ne retire pas cette clé. Un JSON qui l'écrit
    remplace l'injection.
    """
    if custom_guide_explicit:
        value = merged.get("custom_guide")
        if isinstance(value, dict):
            return value
        return None
    if first_architecture == "wanmove":
        return {
            "id": "custom_guide",
            "name": "Trajectory",
            "label": "Trajectory File (.npy)",
            "type": "file",
            "default": None,
            "required": True,
            "file_types": [".npy"],
        }
    return None


def _require_custom_guide(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_model: dict[str, object],
    first_where: str,
    architecture_where: str,
    custom_guide_explicit: bool,
    image_ref_explicit: bool,
    guide_custom_explicit: bool,
    kiwi_ref_explicit: bool,
    queue_video_prompt_type: object,
    queue_audio_prompt_type: object,
    queue_custom_guide: object,
) -> None:
    """Refuse un ``custom_guide`` obligatoire que la file n'envoie pas.

    ``validate_settings`` le contrôle avant l'image de départ et avant
    la vidéo de contrôle. Sans fichier, la tâche est sautée et
    ``process_tasks_cli`` sort 0. Un guide invisible (lettres de
    ``video_prompt_type`` absentes) n'est pas exigé.
    """
    guide = _effective_custom_guide(merged, custom_guide_explicit, first_architecture)
    if guide is None or not guide.get("required", False):
        return
    letters = _resolved_prompt_letters(
        merged,
        settings,
        first_architecture,
        first_model,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_video_prompt_type,
        queue_audio_prompt_type,
    )
    if letters is None:
        return
    video, audio, _video_raw, _ref = letters
    if not _custom_guide_visible(guide, video, audio):
        return
    provided = _queue_or_settings(queue_custom_guide, settings, "custom_guide", None)
    if provided is not None:
        return
    architecture = merged.get("architecture")
    shown = architecture_where or first_where or model_type
    label = guide.get("label", "Custom Guide")
    if custom_guide_explicit:
        origin = (
            f"custom_guide.required est vrai ({label!r}), lu dans {shown}. "
            "La clé vient d'un JSON, pas seulement de l'injection."
        )
    else:
        origin = (
            "query_model_def du premier JSON pose custom_guide "
            f"({label!r}, required vrai) pour {first_architecture} "
            f"({first_where}). init_model_def n'est appelé que pour ce fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas la clé."
        )
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. "
        f"{origin} La file n'envoie pas ce fichier. validate_settings renvoie "
        "une erreur avant l'image de départ. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Écrire custom_guide avec required faux, fournir le fichier dans les réglages, "
        "ou retirer wanmove du premier JSON si la clé n'est pas écrite."
    )


def _require_factory_media(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_model: dict[str, object],
    first_where: str,
    architecture_where: str,
    one_image_ref_explicit: bool,
    image_ref_explicit: bool,
    guide_custom_explicit: bool,
    kiwi_ref_explicit: bool,
    queue_video_prompt_type: object,
    queue_audio_prompt_type: object,
    queue_image_refs: object,
    queue_video_guide: object,
    queue_audio_guide: object,
) -> None:
    """Refuse un média d'usine dont l'absence saute la tâche.

    ``update_default_settings`` pose ``video_prompt_type`` ou
    ``audio_prompt_type`` selon l'architecture finale. La file de ce
    pipeline n'envoie ni image de référence, ni vidéo de contrôle, ni
    audio. ``validate_settings`` renvoie une erreur, ``process_tasks_cli``
    compte la tâche sautée (``completed == total - skipped``) et sort 0.
    ``fix_settings`` retire ``I`` et ``K`` seulement sans
    ``guide_custom_choices`` et sans choix d'image. Le premier choix
    ``I`` de Phantom remet la lettre. ``one_image_ref_needed`` est posé
    par ``query_model_def`` du premier JSON pour StandIn et Lynx, et
    pour Kiwi tant que l'URL n'est pas ``instruct_only``.
    """
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        return
    letters = _resolved_prompt_letters(
        merged,
        settings,
        first_architecture,
        first_model,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_video_prompt_type,
        queue_audio_prompt_type,
    )
    if letters is None:
        return
    video, audio_raw, video_raw, ref = letters
    if one_image_ref_explicit:
        one_image = bool(merged.get("one_image_ref_needed"))
    elif first_architecture is None:
        one_image = False
    else:
        one_image = _injected_one_image_ref(first_architecture, first_model)
    image_refs = _queue_or_settings(queue_image_refs, settings, "image_refs", None)
    video_guide = _queue_or_settings(queue_video_guide, settings, "video_guide", None)
    audio_guide = _queue_or_settings(queue_audio_guide, settings, "audio_guide", None)
    shown = architecture_where or first_where or model_type
    if "A" in audio_raw and audio_guide is None and not bool(merged.get("auto_null_audio")):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. update_default_settings "
            "(models/wan/wan_handler.py) pose audio_prompt_type à "
            f"{audio_raw!r} pour l'architecture finale {architecture}. "
            "Cette chaîne contient A et la file n'envoie pas d'audio. "
            "validate_settings renvoie une erreur. process_tasks_cli compte la tâche "
            "comme sautée et quitte avec le code 0, sans MP4. "
            f"Architecture lue dans {shown}. "
            f"Retirer {architecture}, ou poser audio_prompt_type sans A dans les réglages du JSON."
        )
    reference_missing = _reference_missing(image_refs)
    if one_image and image_refs is None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. query_model_def du premier JSON "
            f"pose one_image_ref_needed pour {first_architecture} "
            f"({first_where or shown}). La file n'envoie pas d'image de référence. "
            "validate_settings renvoie une erreur avant l'image de départ. "
            "process_tasks_cli compte la tâche comme sautée et quitte avec le code 0, "
            "sans MP4. image_outputs peut rester absent. "
            f"Architecture finale {architecture}, lue dans {shown}. "
            "Écrire one_image_ref_needed à faux, ou retirer cette architecture."
        )
    if bool(merged.get("at_least_one_image_ref_needed")) and image_refs is None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. at_least_one_image_ref_needed "
            f"est vrai pour {architecture} et la file n'envoie pas d'image de référence. "
            "validate_settings renvoie une erreur. process_tasks_cli quitte avec le code 0, "
            f"sans MP4. Réglage lu dans {shown}."
        )
    if "I" in video:
        if reference_missing is None:
            return
        if reference_missing:
            added = ""
            if "I" not in video_raw and ref[0] == "choices" and "I" in ref[1]:
                added = (
                    " fix_settings ajoute I : le premier choix de image_ref_choices "
                    f"est {ref[1]!r}, injecté par query_model_def du premier JSON "
                    "quand la clé manque."
                )
            raise Wan2GPError(
                "La tâche est sautée avant toute vidéo. update_default_settings "
                "(models/wan/wan_handler.py) pose video_prompt_type à "
                f"{video_raw!r} pour l'architecture finale {architecture}. "
                f"Après fix_settings la chaîne est {video!r} et contient I. "
                "La file n'envoie pas d'image de référence. validate_settings renvoie "
                "une erreur. process_tasks_cli compte la tâche comme sautée et quitte "
                "avec le code 0, sans MP4."
                f"{added} "
                f"Architecture lue dans {shown}. "
                f"Retirer {architecture}, écrire video_prompt_type sans I, "
                "ou un image_ref_choices dont le premier choix ne contient pas I "
                "si la lettre ne vient que de ce choix."
            )
    if "V" in video and video_guide is None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. update_default_settings "
            "(models/wan/wan_handler.py) pose video_prompt_type à "
            f"{video_raw!r} pour l'architecture finale {architecture}. "
            f"Après fix_settings la chaîne est {video!r} et contient V. "
            "La file n'envoie pas de vidéo de contrôle. validate_settings renvoie "
            "une erreur. process_tasks_cli compte la tâche comme sautée et quitte "
            "avec le code 0, sans MP4. "
            f"Architecture lue dans {shown}. "
            f"Retirer {architecture}, ou poser video_prompt_type sans V dans les réglages du JSON."
        )


def _injected_one_image_ref(architecture: str, first_model: dict[str, object]) -> bool:
    """``one_image_ref_needed`` posé par ``query_model_def`` du premier JSON."""
    if architecture in _WAN_ONE_IMAGE_REF:
        return True
    if architecture == "kiwi_edit":
        return _kiwi_embedder_from_urls(first_model)
    return False


def _require_video_model_definition(
    root: Path | None,
    model_type: str,
    queue_model_mode: object = _QUEUE_MODEL_MODE_UNSET,
    queue_image_prompt_type: object = _QUEUE_MODEL_MODE_UNSET,
    queue_video_prompt_type: object = _QUEUE_MODEL_MODE_UNSET,
    queue_audio_prompt_type: object = _QUEUE_MODEL_MODE_UNSET,
    queue_image_refs: object = _QUEUE_MODEL_MODE_UNSET,
    queue_video_guide: object = _QUEUE_MODEL_MODE_UNSET,
    queue_audio_guide: object = _QUEUE_MODEL_MODE_UNSET,
    queue_custom_guide: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sliding_window_overlap: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sliding_window_size: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sub_parallel_window_overlap: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sub_parallel_window_size: object = _QUEUE_MODEL_MODE_UNSET,
    queue_color_correction_strength: object = _QUEUE_MODEL_MODE_UNSET,
    queue_overlap_noise: object = _QUEUE_MODEL_MODE_UNSET,
    queue_discard_last_frames: object = _QUEUE_MODEL_MODE_UNSET,
    queue_trim_first_frames: object = _QUEUE_MODEL_MODE_UNSET,
    queue_top_p: object = _QUEUE_MODEL_MODE_UNSET,
    queue_top_k: object = _QUEUE_MODEL_MODE_UNSET,
    queue_nag_scale: object = _QUEUE_MODEL_MODE_UNSET,
    queue_nag_tau: object = _QUEUE_MODEL_MODE_UNSET,
    queue_nag_alpha: object = _QUEUE_MODEL_MODE_UNSET,
    queue_custom_settings: object = _QUEUE_MODEL_MODE_UNSET,
    queue_custom_sigmas: object = _QUEUE_MODEL_MODE_UNSET,
    queue_guidance_phases: object = _QUEUE_MODEL_MODE_UNSET,
    queue_sample_solver: object = _QUEUE_MODEL_MODE_UNSET,
) -> None:
    """Refuse une définition qui fait écrire autre chose qu'un MP4.

    La file de ce pipeline met ``image_mode`` à 0. WanGP 18.84 repasse à 1
    si ``model.image_outputs`` est vrai, puis sort avec le code 0. Le JSON
    est celui que le clone ouvre, y compris le ``defaults`` d'un plugin.
    Un lien ``/proc/self`` ne doit pas faire lire le leurre du pipeline,
    ni ignorer le fichier du clone. Le premier JSON décide aussi de
    l'architecture passée à ``query_model_def``. Une architecture image,
    audio, mesh ou world pose le drapeau sans clé. Les fichiers suivants
    ne le retirent que s'ils l'écrivent.     L'architecture finale, elle,
    alimente ``validate_generative_settings`` : ``chrono_edit`` force
    quand même le JPEG si ``model_mode`` n'est pas 2. Un ``model_mode`` 2
    est aussi annulé avant ce test quand ``model_modes.image_modes`` ne
    contient pas 0 : le handler voit None et écrit le JPEG.
    ``fix_settings`` de l'architecture finale ``qwen_image_21_7B`` remet
    aussi ``image_mode`` à 1 après la file, même si ``image_outputs`` est
    faux ou absent. ``query_model_def`` du premier JSON peut en plus
    injecter ``model_modes.image_modes`` ``[2]``. Tant qu'aucun JSON
    n'écrit ``model_modes``, un ``chrono_edit`` final avec ``model_mode``
    2 voit cette liste, ``model_mode`` devient None et le JPEG sort.
    Une architecture I2V finale dont ``image_prompt_types_allowed`` contient
    ``S`` pose ``image_prompt_type`` à ``S``. Sans image de départ, la tâche
    est sautée et le CLI sort 0. La chaîne vient du premier
    ``query_model_def`` si aucun JSON ne l'écrit.
    ``update_default_settings`` pose aussi ``video_prompt_type`` ou
    ``audio_prompt_type``. Sans le média demandé, la tâche est sautée
    de la même façon. ``one_image_ref_needed`` vient du premier JSON.
    ``query_model_def("wanmove")`` pose un ``custom_guide`` obligatoire.
    Un finetune qui change l'architecture ne l'efface pas. Sans le
    fichier, ``validate_settings`` saute la tâche avant l'image de départ
    et le CLI sort 0.
    ``query_model_def`` d'Ovi pose ``sliding_window`` et ``overlap_max`` 1.
    Le défaut ``sliding_window_overlap`` reste 5. ``extra_settings`` refuse
    la valeur, la tâche est sautée et le CLI sort 0. Un JSON suivant qui
    change l'architecture ne retire pas ce plafond.
    Le même validateur refuse ``sliding_window_size`` avant ce recouvrement.
    MiniMax H3 pose ``window_min`` à 124. L'architecture finale
    ``ti2v_2_2`` laisse la taille à 121 : la tâche est sautée et le CLI
    sort 0. Un JSON qui écrit ``window_min`` ou ``window_max`` hors de
    la taille retenue fait pareil.
    ``sub_parallel_windows`` rend ensuite visibles
    ``sub_parallel_window_size`` puis ``sub_parallel_window_overlap``.
    La taille d'usine est 0 et le minimum est fixe : 0 passe. Une valeur
    300 dépasse le ``window_max`` d'usine 257 avant le recouvrement, même
    si celui-ci vaut 8. ``sliding_window_size`` 129 reste sous 257. La
    tâche est sautée et le CLI sort 0. Le recouvrement d'usine est 17.
    Un ``overlap_max`` de 8 laisse le recouvrement 5 et refuse 17.
    ``scail2`` injecte le drapeau sans changer ces plafonds (257 et 97) :
    la taille 0 et le recouvrement 17 restent acceptés.
    ``color_correction`` rend ensuite visible
    ``sliding_window_color_correction_strength``. Le défaut d'usine est 0.
    Une valeur 2 ou 1.01 est refusée, la tâche est sautée et le CLI sort 0.
    ``query_model_def`` de Wan pose le drapeau à vrai. ``animate2`` le
    remet à faux dans ce même appel. Un JSON qui l'écrit à faux l'efface.
    ``vace_class`` ou un ``model_type`` qui contient ``sky_df`` ou
    ``diffusion_forcing`` rend ensuite visible
    ``sliding_window_overlap_noise``. Le défaut d'usine est 0. Une
    valeur 151 est refusée, la tâche est sautée et le CLI sort 0.
    ``query_model_def`` pose ``vace_class`` pour ``test_vace`` au premier
    JSON seulement. ``df_handler`` ne pose pas ``color_correction`` :
    une force 2 y est ignorée et le bruit 151 est l'erreur.
    ``sliding_window`` vrai, sans ``sky_df`` ni ``diffusion_forcing`` dans
    ``model_type``, rend ensuite visible ``sliding_window_discard_last_frames``.
    Le défaut d'usine est 0. Une valeur 21 est refusée, la tâche est sautée
    et le CLI sort 0. ``query_model_def`` de ``t2v_1.3B`` pose
    ``sliding_window`` à faux. ``multitalk`` pose la valeur à 4, dans la borne.
    Une chaîne que le ``int()`` du frame scheduler refuse (``large``,
    ``21.5``) lève ``ValueError`` avant ``extra_settings`` : le processus
    ne sort pas 0.
    ``sliding_window`` vrai rend ensuite visible
    ``sliding_window_trim_first_frames``, y compris si ``model_type``
    contient ``sky_df`` ou ``diffusion_forcing``. Le défaut d'usine est 0.
    Une valeur 11 est refusée, la tâche est sautée et le CLI sort 0.
    ``wan_handler`` ne réécrit pas cette clé. Si le rejet 21 est visible,
    cette erreur-là sort avant. ``large`` pour ce trim n'est pas un
    ``int()`` du scheduler : ``extra_settings`` renvoie une erreur et le
    CLI sort 0. Un ``discard_step`` 0, ou un rejet que le ``int()`` refuse,
    lève avant ce contrôle.
    ``top_p_slider`` rend ensuite visible ``top_p``. Le défaut d'usine
    est 0,9 (minimum 0, maximum 1). ``query_model_def`` de Wan ne pose
    pas le drapeau. Un JSON qui l'écrit rend 1,01 ou 2 refusé, même si
    ``sliding_window`` est faux et que le trim vaut 0. La tâche est
    sautée et le CLI sort 0. ``top_k_slider`` seul ne suffit pas.
    ``temperature`` n'est visible qu'avec ``audio_only``, déjà refusé.
    Si le trim 11 est visible, cette erreur-là sort avant.
    ``top_k_slider`` rend ensuite visible ``top_k``. Le défaut d'usine
    est 50 (entier, minimum 0, maximum 100). ``_trim_label`` coupe le
    libellé au premier ``(`` : l'erreur dit ``Top-k``. ``query_model_def``
    de Wan ne pose pas le drapeau. Un JSON qui l'écrit rend 101 refusé,
    même si ``top_p_slider`` est faux. La tâche est sautée et le CLI
    sort 0. Si ``top_p`` 2 est déjà visible, cette erreur-là sort avant.
    ``NAG`` rend ensuite visible ``NAG_scale`` (nombre, minimum 1,
    maximum 20, sauf objet ``model.NAG_scale``). Le défaut d'usine est 1.
    ``query_model_def`` pose le drapeau pour ``test_class_t2v``,
    ``test_class_i2v`` et ``test_vace``, sauf ``bernini`` et ``animate2``.
    ``t2v_1.3B`` n'est pas dans ``test_class_t2v``. Une valeur 21 est
    refusée, la tâche est sautée et le CLI sort 0. Si ``top_k`` 101 est
    visible, cette erreur-là sort avant. ``NAG`` rend ensuite visible
    ``NAG_tau`` (nombre, minimum 1, maximum 5, sauf objet
    ``model.NAG_tau``). Le défaut d'usine est 3,5. Une valeur 5,01 est
    refusée, la tâche est sautée et le CLI sort 0. Si ``NAG_scale`` 21
    est visible, cette erreur-là sort avant. Un objet illisible sur
    ``NAG_scale``, ``NAG_tau`` ou ``NAG_alpha`` lève ``AttributeError``
    dans ``get_def`` avant cette erreur : le processus ne sort pas 0.
    ``NAG`` rend ensuite visible ``NAG_alpha`` (nombre, minimum 0,
    maximum 2, sauf objet ``model.NAG_alpha``). Le défaut d'usine est
    0,5. Une valeur 2,01 est refusée, la tâche est sautée et le CLI
    sort 0. Si ``NAG_tau`` 5,01 est visible, cette erreur-là sort avant.
    ``pace`` est contrôlé ensuite, seulement si ``custom_settings``
    contient l'id ``pace`` parmi les cinq premiers objets. Il n'y a pas
    de défaut d'usine. La valeur est lue dans le dict ``custom_settings``
    des entrées, pas à la racine. 9, ``9`` et 1,01 sont refusés. La tâche
    est sautée et le CLI sort 0. Un objet illisible sur ``NAG_scale``,
    ``NAG_tau``, ``NAG_alpha``, ``switch_threshold`` ou
    ``temperature_slider`` lève ``AttributeError`` avant cette erreur.
    ``exaggeration`` est contrôlé ensuite, même mécanisme, minimum 0,25
    et maximum 2. 3, ``3`` et 2,01 sont refusés. Si ``pace`` 9 est
    visible, cette erreur-là sort avant. Le pré-parse qui transforme un
    type ``integer`` en texte n'est pas ce saut. Un type ``dropdown``
    dont ``choices`` n'est pas vide refuse une valeur hors liste avant
    ce pré-parse numérique (« Expected one of the dropdown choices. »).
    ``_setting_kind`` voit une chaîne et laisserait passer.
    ``parse_custom_sigmas`` tourne ensuite, seulement si
    ``model.custom_sigmas`` est vrai. ``query_model_def`` de Wan pose
    ce drapeau pour toute architecture Wan, y compris ``t2v_1.3B``.
    ``models/_settings.json`` laisse la chaîne vide et
    ``guidance_phases`` à 1. Une chaîne vide ou des espaces passent.
    ``nope``, un entier 12, ``0.2,0.9``, ``1,0.5``, ``0`` seul ou
    ``0.9;0.2`` avec une phase sautent la tâche. Le CLI sort 0, sans
    MP4. Un menu hors liste ou un type ``int`` sort avant. ``NAG``
    sort après. Si un créneau reste défini et que
    ``custom_sigmas_schedulers`` n'est pas ``None``, un
    ``sample_solver`` hors de cette liste saute aussi. Wan injecte
    ``["", "unipc", "euler", "dpm++"]``. ``lcm`` et ``causvid`` avec
    ``0.9,0.2`` sont refusés. L'usine est ``unipc``, ``euler`` pour
    ``multitalk`` et ``infinitetalk``, ``dpm++`` pour ``animate2``.
    Une chaîne vide ignore le solveur. Le contrôle continu vient après.
    """
    if root is None:
        return
    merged: dict[str, object] = {}
    origin: dict[str, str] = {}
    settings: dict[str, object] = {}
    first_architecture: str | None = None
    first_where = ""
    architecture_where = ""
    model_modes_where = ""
    prompt_types_explicit = False
    black_frame_explicit = False
    one_image_ref_explicit = False
    image_ref_explicit = False
    guide_custom_explicit = False
    kiwi_ref_explicit = False
    custom_guide_explicit = False
    sliding_window_explicit = False
    sliding_defaults_explicit = False
    sub_parallel_explicit = False
    color_correction_explicit = False
    vace_class_explicit = False
    top_p_slider_explicit = False
    top_k_slider_explicit = False
    nag_explicit = False
    custom_sigmas_explicit = False
    custom_sigmas_schedulers_explicit = False
    custom_settings_where = ""
    first_model: dict[str, object] = {}
    for _relative, logical, opened, note in _model_definition_entries(root, model_type):
        try:
            loaded = json.loads(opened.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if _relative.startswith("finetunes/"):
                continue
            raise Wan2GPError(
                f"Définition {_relative} illisible ({opened}) : {exc}. "
                "WanGP interrompt le chargement des définitions de modèle et ne produit pas de MP4."
            ) from exc
        if not isinstance(loaded, dict):
            if _relative.startswith("finetunes/"):
                continue
            raise Wan2GPError(f"Définition {_relative} doit être un objet ({opened}).")
        model = loaded.get("model")
        if not isinstance(model, dict):
            if _relative.startswith("finetunes/"):
                continue
            raise Wan2GPError(
                f"Définition {_relative} n'a pas d'objet model ({opened}). "
                "WanGP lève une exception au chargement et ne produit pas de MP4."
            )
        where = _model_def_where(logical, opened)
        if note and "/proc/self" not in where:
            where = f"{where}. {note}"
        if first_architecture is None:
            first_architecture = _architecture_name(model, model_type)
            first_where = where
            first_model = dict(model)
        for flag, _message in _NON_VIDEO_MODEL_FLAGS:
            if flag in model:
                origin[flag] = where
        settings.update({key: value for key, value in loaded.items() if key != "model"})
        if "architecture" in model:
            architecture_where = where
        if "model_modes" in model:
            model_modes_where = where
        if "image_prompt_types_allowed" in model:
            prompt_types_explicit = True
        if "black_frame" in model:
            black_frame_explicit = True
        if "one_image_ref_needed" in model:
            one_image_ref_explicit = True
        if "image_ref_choices" in model:
            image_ref_explicit = True
        if "guide_custom_choices" in model:
            guide_custom_explicit = True
        if "kiwi_ref_embedder" in model:
            kiwi_ref_explicit = True
        if "custom_guide" in model:
            custom_guide_explicit = True
        if "sliding_window" in model:
            sliding_window_explicit = True
        if "sliding_window_defaults" in model:
            sliding_defaults_explicit = True
        if "sub_parallel_windows" in model:
            sub_parallel_explicit = True
        if "color_correction" in model:
            color_correction_explicit = True
        if "vace_class" in model:
            vace_class_explicit = True
        if "top_p_slider" in model:
            top_p_slider_explicit = True
        if "top_k_slider" in model:
            top_k_slider_explicit = True
        if "NAG" in model:
            nag_explicit = True
        if "custom_sigmas" in model:
            custom_sigmas_explicit = True
        if "custom_sigmas_schedulers" in model:
            custom_sigmas_schedulers_explicit = True
        if "custom_settings" in model:
            custom_settings_where = where
        merged.update(model)
    forced = (
        _HANDLER_FORCED_FLAGS.get(first_architecture, frozenset())
        if first_architecture is not None
        else frozenset()
    )
    for flag, message in _NON_VIDEO_MODEL_FLAGS:
        explicit = flag in origin
        active = bool(merged.get(flag)) if explicit else flag in forced
        if not active:
            continue
        if explicit:
            where = origin[flag]
            raise Wan2GPError(
                f"{message} Réglage lu dans {where}. "
                "Retirer ce drapeau de defaults/, de finetunes/, ou du dossier defaults "
                "d'un plugin activé, ou reprendre la définition d'origine de "
                f"{model_type}."
            )
        raise Wan2GPError(
            f"{message} La clé est absente du JSON. query_model_def({first_architecture!r}) "
            f"la met à vrai dans init_model_def, appelé seulement pour le premier fichier "
            f"({first_where}). Les JSON suivants ne rappellent pas query_model_def : "
            "une architecture réécrite sans le drapeau, y compris le defaults officiel "
            f"de {model_type} qui ne contient pas image_outputs, le laisse vrai. "
            "process_tasks_cli quitte avec le code 0, sans MP4. "
            f"Écrire {flag} à faux dans un JSON de {model_type}, ou retirer "
            f"l'architecture {first_architecture} de ce premier fichier."
        )
    _require_custom_dropdowns(
        merged,
        settings,
        model_type,
        first_architecture,
        first_model,
        first_where,
        architecture_where,
        custom_settings_where,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        sliding_window_explicit,
        queue_model_mode,
        queue_video_prompt_type,
        queue_audio_prompt_type,
        queue_discard_last_frames,
        queue_custom_settings,
    )
    _require_custom_sigmas(
        merged,
        settings,
        model_type,
        first_architecture,
        first_model,
        first_where,
        architecture_where,
        custom_sigmas_explicit,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        sliding_window_explicit,
        queue_model_mode,
        queue_video_prompt_type,
        queue_audio_prompt_type,
        queue_discard_last_frames,
        queue_custom_settings,
        queue_custom_sigmas,
        queue_guidance_phases,
        queue_sample_solver,
        custom_sigmas_schedulers_explicit,
    )
    _require_chrono_final_architecture(
        merged,
        settings,
        architecture_where,
        model_modes_where,
        queue_model_mode,
        model_type,
        first_architecture,
        first_where,
    )
    _require_fix_settings_image_mode(merged, architecture_where, model_type)
    _require_custom_guide(
        merged,
        settings,
        model_type,
        first_architecture,
        first_model,
        first_where,
        architecture_where,
        custom_guide_explicit,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_video_prompt_type,
        queue_audio_prompt_type,
        queue_custom_guide,
    )
    _require_i2v_start_image(
        merged,
        settings,
        queue_model_mode,
        queue_image_prompt_type,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        prompt_types_explicit,
        black_frame_explicit,
    )
    _require_factory_media(
        merged,
        settings,
        model_type,
        first_architecture,
        first_model,
        first_where,
        architecture_where,
        one_image_ref_explicit,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_video_prompt_type,
        queue_audio_prompt_type,
        queue_image_refs,
        queue_video_guide,
        queue_audio_guide,
    )
    _require_sliding_overlap(
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        sliding_window_explicit,
        sliding_defaults_explicit,
        queue_sliding_window_overlap,
    )
    _require_sliding_size(
        root,
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        sliding_window_explicit,
        sliding_defaults_explicit,
        queue_sliding_window_size,
    )
    _require_sub_parallel_size(
        root,
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        sub_parallel_explicit,
        sliding_defaults_explicit,
        queue_sub_parallel_window_size,
    )
    _require_sub_parallel_overlap(
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        sub_parallel_explicit,
        sliding_defaults_explicit,
        queue_sub_parallel_window_overlap,
    )
    _require_color_correction(
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        color_correction_explicit,
        queue_color_correction_strength,
    )
    _require_overlap_noise(
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        vace_class_explicit,
        queue_overlap_noise,
    )
    _require_discard_last_frames(
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        sliding_window_explicit,
        queue_discard_last_frames,
    )
    _require_trim_first_frames(
        merged,
        settings,
        model_type,
        first_architecture,
        first_where,
        architecture_where,
        sliding_window_explicit,
        queue_discard_last_frames,
        queue_trim_first_frames,
    )
    _require_top_p(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        top_p_slider_explicit,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_top_p,
    )
    _require_top_k(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        top_k_slider_explicit,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_top_k,
    )
    _require_nag_scale(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        nag_explicit,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_nag_scale,
    )
    _require_nag_tau(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        nag_explicit,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_nag_tau,
    )
    _require_nag_alpha(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        nag_explicit,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_nag_alpha,
    )
    _require_pace(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        custom_settings_where,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_custom_settings,
    )
    _require_exaggeration(
        merged,
        settings,
        model_type,
        first_where,
        architecture_where,
        custom_settings_where,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
        queue_custom_settings,
    )


_PRIMARY_SLIDING_OVERLAP = 5
_DEFAULT_OVERLAP_MAX = 97
_DEFAULT_OVERLAP_MIN = 1


def _oneframe_overlap_architecture(architecture: str) -> bool:
    """``test_oneframe_overlap`` de ``models/wan/wan_handler.py``."""
    if architecture in {"ti2v_2_2", "lucy_edit", "kiwi_edit"}:
        return True
    if architecture not in _WAN_I2V:
        return False
    if architecture in _WAN_MULTITALK:
        return False
    if architecture in {"animate", "scail", "scail2_14B", "scail2_1.3B", "i2v_2_2_svi2pro"}:
        return False
    return True


def _factory_sliding_overlap(architecture: str) -> int:
    """``sliding_window_overlap`` après ``update_default_settings``.

    ``models/_settings.json`` part de 5. Le handler de l'architecture
    finale peut le remplacer. Ovi ne le fait pas.
    """
    if architecture in {"magi_human", "magi_human_distill"}:
        return 1
    if architecture in {"hunyuan_i2v", "hunyuan_1_5_i2v", "hunyuan_1_5_upsampler"}:
        return 1
    if architecture == "i2v_2_2_svi2pro":
        return 4
    if architecture == "infinitetalk":
        return 9
    if architecture in {"standin", "lynx_lite", "lynx", "alpha_lynx"}:
        return 9
    if architecture in {"fantasy", "scail"}:
        return 1
    if _oneframe_overlap_architecture(architecture):
        return 1
    return _PRIMARY_SLIDING_OVERLAP


def _overlap_bound(value: object, fallback: int) -> int:
    """Borne lue comme ``int(defaults.get(bound, fallback))``."""
    try:
        if isinstance(value, str):
            return int(value.strip())
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _sliding_overlap_number(value: object) -> int | None:
    """Entier accepté par ``validate_setting_value``.

    ``None`` si la valeur est vide (pas d'erreur) ou si ce n'est pas un
    entier (``extra_settings`` renvoie alors une erreur et la tâche saute).
    Le booléen est distingué : ``True`` n'est pas un entier pour WanGP.
    """
    if value is None:
        return None
    if isinstance(value, str) and len(value.strip()) == 0:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None


def _overlap_is_rejected(value: object) -> bool:
    """``True`` quand ``validate_setting_value`` renvoie une erreur."""
    if value is None:
        return False
    if isinstance(value, str) and len(value.strip()) == 0:
        return False
    return _sliding_overlap_number(value) is None


def _require_sliding_overlap(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    sliding_window_explicit: bool,
    sliding_defaults_explicit: bool,
    queue_sliding_window_overlap: object,
) -> None:
    """Refuse un recouvrement de fenêtre hors de la borne du modèle.

    ``models/wan/ovi_handler.py`` (WanGP 18.84) met ``sliding_window`` à
    vrai et ``overlap_max`` à 1. ``models/_settings.json`` vaut 5, et
    ``update_default_settings`` d'Ovi ne le change pas.
    ``extra_settings.validate_inputs`` renvoie une erreur.
    ``validate_settings`` saute la tâche. ``process_tasks_cli`` sort 0
    (``completed == total - skipped``), sans MP4. ``init_model_def``
    n'injecte ces clés que pour le premier JSON.
    """
    if sliding_window_explicit:
        enabled = bool(merged.get("sliding_window"))
    else:
        enabled = first_architecture == "ovi"
    if not enabled:
        return
    if sliding_defaults_explicit:
        raw_defaults = merged.get("sliding_window_defaults")
        defaults = raw_defaults if isinstance(raw_defaults, dict) else {}
        overlap_max = _overlap_bound(defaults.get("overlap_max", _DEFAULT_OVERLAP_MAX), _DEFAULT_OVERLAP_MAX)
        overlap_min = _overlap_bound(defaults.get("overlap_min", _DEFAULT_OVERLAP_MIN), _DEFAULT_OVERLAP_MIN)
        bounds_note = "Les bornes viennent du JSON (sliding_window_defaults)."
    elif first_architecture == "ovi":
        overlap_max = 1
        overlap_min = 1
        bounds_note = (
            "query_model_def d'ovi pose overlap_max et overlap_min à 1 "
            f"({first_where}). init_model_def n'est appelé que pour ce premier fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas ces bornes."
        )
    else:
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_sliding_window_overlap is not _QUEUE_MODEL_MODE_UNSET
        and queue_sliding_window_overlap is not None
    ):
        raw_overlap: object = queue_sliding_window_overlap
        overlap_note = "La file envoie sliding_window_overlap."
    elif "sliding_window_overlap" in settings:
        raw_overlap = settings["sliding_window_overlap"]
        overlap_note = "Le JSON de réglages envoie sliding_window_overlap."
    else:
        raw_overlap = _factory_sliding_overlap(architecture)
        if architecture == "ovi":
            overlap_note = (
                "update_default_settings d'ovi ne remplace pas sliding_window_overlap. "
                f"models/_settings.json le laisse à {raw_overlap}."
            )
        else:
            overlap_note = (
                f"update_default_settings de {architecture} laisse sliding_window_overlap "
                f"à {raw_overlap}. models/_settings.json part de 5."
            )
    shown = architecture_where or first_where or model_type
    if _overlap_is_rejected(raw_overlap):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
            f"sliding_window_overlap vaut {raw_overlap!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{overlap_note} {bounds_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    overlap = _sliding_overlap_number(raw_overlap)
    if overlap is None:
        return
    if overlap_min <= overlap <= overlap_max:
        return
    if overlap > overlap_max:
        limit = f"au-dessus de {overlap_max}"
        wan_message = f"must be at most {overlap_max}"
    else:
        limit = f"en dessous de {overlap_min}"
        wan_message = f"must be at least {overlap_min}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
        f"sliding_window_overlap vaut {overlap}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"(Windows Frames Overlap {wan_message}). "
        "validate_settings renvoie cette erreur avant le guide et l'image de départ. "
        "process_tasks_cli compte la tâche comme sautée et quitte avec le code 0, "
        "sans MP4. "
        f"{overlap_note} {bounds_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sliding_window_overlap dans la borne, écrire sliding_window à faux, "
        "ou relever overlap_max."
    )


_PRIMARY_SLIDING_SIZE = 129
_DEFAULT_WINDOW_MIN = 5
_DEFAULT_WINDOW_MAX = 257

# query_model_def de MiniMax H3 (WanGP 18.84). window_min reste 124.
# Le contrôle et Viggle ne changent que window_max.
_H3_WINDOW_BOUNDS: dict[str, tuple[int, int]] = {
    "minimax_h3_fl2va": (124, 481),
    "minimax_h3_fl2va_pruned": (124, 481),
    "minimax_h3_ref2va": (124, 481),
    "minimax_h3_ref2va_pruned": (124, 481),
    "minimax_h3_control": (124, 362),
    "minimax_h3_control_pruned": (124, 362),
    "viggle_animate": (124, 124),
}

# Architectures dont le premier query_model_def pose sliding_window sans
# window_min ni window_max. extra_settings retombe alors sur 5 et 257.
_SLIDING_WINDOW_INJECTED = (
    frozenset(
        {
            "ovi",
            "t2v",
            "t2v_2_2",
            "lynx",
            "ti2v_2_2",
            "lucy_edit",
            "kiwi_edit",
        }
    )
    | _WAN_I2V
    | _WAN_VACE
    | frozenset(_H3_WINDOW_BOUNDS)
)

_H3_FACTORY_SIZE = frozenset(
    {
        "minimax_h3_fl2va",
        "minimax_h3_fl2va_pruned",
        "minimax_h3_ref2va",
        "minimax_h3_ref2va_pruned",
        "minimax_h3_control",
        "minimax_h3_control_pruned",
    }
)


def _factory_sliding_size(architecture: str) -> int:
    """``sliding_window_size`` après ``update_default_settings``.

    ``models/_settings.json`` part de 129. Le handler de l'architecture
    finale peut le remplacer. MiniMax le passe à 362, Viggle à 124,
    Wan 5B et Ovi à 121.
    """
    if architecture == "infinitetalk":
        return 81
    if architecture == "animate2":
        return 65
    if architecture in {"scail", "scail2_14B", "scail2_1.3B", "i2v_2_2_svi2pro"}:
        return 81
    if architecture in {"ti2v_2_2", "lucy_edit", "kiwi_edit", "ovi"}:
        return 121
    if architecture == "viggle_animate":
        return 124
    if architecture in _H3_FACTORY_SIZE:
        return 362
    return _PRIMARY_SLIDING_SIZE


def _max_frames_multiplier(root: Path | None) -> int:
    """``get_max_frames`` de WanGP. Défaut 1 : ``window_max`` ne change pas."""
    if root is None:
        return 1
    opened = _wan_config_file(root)
    if opened is None:
        return 1
    try:
        loaded = json.loads(opened.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 1
    if not isinstance(loaded, dict):
        return 1
    raw = loaded.get("max_frames_multiplier", 1)
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _sliding_size_enabled(
    merged: dict[str, object],
    sliding_window_explicit: bool,
    first_architecture: str | None,
) -> bool:
    """``sliding_window`` vu par ``extra_settings`` après ``init_model_def``."""
    if sliding_window_explicit:
        return bool(merged.get("sliding_window"))
    return first_architecture in _SLIDING_WINDOW_INJECTED


def _window_bounds(
    merged: dict[str, object],
    sliding_defaults_explicit: bool,
    first_architecture: str | None,
) -> tuple[int, int, str]:
    """``window_min`` / ``window_max`` avant ``get_max_frames``."""
    if sliding_defaults_explicit:
        raw_defaults = merged.get("sliding_window_defaults")
        defaults = raw_defaults if isinstance(raw_defaults, dict) else {}
        window_min = _overlap_bound(defaults.get("window_min", _DEFAULT_WINDOW_MIN), _DEFAULT_WINDOW_MIN)
        window_max = _overlap_bound(defaults.get("window_max", _DEFAULT_WINDOW_MAX), _DEFAULT_WINDOW_MAX)
        return window_min, window_max, "Les bornes viennent du JSON (sliding_window_defaults)."
    injected = None if first_architecture is None else _H3_WINDOW_BOUNDS.get(first_architecture)
    if injected is not None:
        window_min, window_max = injected
        return (
            window_min,
            window_max,
            f"query_model_def({first_architecture!r}) pose window_min {window_min} et "
            f"window_max {window_max}. init_model_def n'est appelé que pour ce premier fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas ces bornes.",
        )
    return (
        _DEFAULT_WINDOW_MIN,
        _DEFAULT_WINDOW_MAX,
        "Sans sliding_window_defaults, extra_settings part de window_min 5 et window_max 257.",
    )


def _require_sliding_size(
    root: Path | None,
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    sliding_window_explicit: bool,
    sliding_defaults_explicit: bool,
    queue_sliding_window_size: object,
) -> None:
    """Refuse une taille de fenêtre hors de ``window_min`` / ``window_max``.

    ``extra_settings.validate_inputs`` contrôle ``sliding_window_size``
    avant ``sliding_window_overlap``. MiniMax H3 pose ``window_min`` à 124.
    ``update_default_settings`` de ``ti2v_2_2`` laisse la taille à 121.
    La tâche est sautée, ``process_tasks_cli`` sort 0, sans MP4.
    ``get_max_frames`` n'élargit que la borne haute.
    """
    if not _sliding_size_enabled(merged, sliding_window_explicit, first_architecture):
        return
    window_min, window_max, bounds_note = _window_bounds(
        merged,
        sliding_defaults_explicit,
        first_architecture,
    )
    multiplier = _max_frames_multiplier(root)
    widened_max = (window_max - 1) * multiplier + 1
    if multiplier != 1:
        bounds_note += (
            f" get_max_frames élargit window_max avec max_frames_multiplier {multiplier} : "
            f"la borne haute devient {widened_max}."
        )
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_sliding_window_size is not _QUEUE_MODEL_MODE_UNSET
        and queue_sliding_window_size is not None
    ):
        raw_size: object = queue_sliding_window_size
        size_note = "La file envoie sliding_window_size."
    elif "sliding_window_size" in settings:
        raw_size = settings["sliding_window_size"]
        size_note = "Le JSON de réglages envoie sliding_window_size."
    else:
        raw_size = _factory_sliding_size(architecture)
        if architecture == "ti2v_2_2":
            size_note = (
                "update_default_settings de ti2v_2_2 (test_wan_5B) pose "
                f"sliding_window_size à {raw_size}. models/_settings.json part de 129."
            )
        elif architecture in _H3_FACTORY_SIZE:
            size_note = (
                "update_default_settings de MiniMax pose "
                f"sliding_window_size à {raw_size}. models/_settings.json part de 129."
            )
        elif architecture == "ovi":
            size_note = (
                "update_default_settings d'ovi pose "
                f"sliding_window_size à {raw_size}. models/_settings.json part de 129."
            )
        else:
            size_note = (
                f"update_default_settings de {architecture} laisse sliding_window_size "
                f"à {raw_size}. models/_settings.json part de 129."
            )
    shown = architecture_where or first_where or model_type
    if _overlap_is_rejected(raw_size):
        kind = "a integer" if isinstance(raw_size, bool) else "an integer"
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
            f"sliding_window_size vaut {raw_size!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"(Sliding Window Size must be {kind}). "
            "validate_settings la propage avant sliding_window_overlap. "
            "process_tasks_cli compte la tâche comme sautée et quitte avec le code 0, "
            "sans MP4. "
            f"{size_note} {bounds_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    size = _sliding_overlap_number(raw_size)
    if size is None:
        return
    if window_min <= size <= widened_max:
        return
    if size > widened_max:
        limit = f"au-dessus de {widened_max}"
        wan_message = f"must be at most {widened_max}"
    else:
        limit = f"en dessous de {window_min}"
        wan_message = f"must be at least {window_min}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
        f"sliding_window_size vaut {size}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"(Sliding Window Size {wan_message}) avant sliding_window_overlap. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{size_note} {bounds_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sliding_window_size dans la borne, écrire sliding_window à faux, "
        "ou élargir window_min et window_max."
    )


_SCAIL2_ARCHITECTURES = frozenset({"scail2_14B", "scail2_1.3B"})
_FACTORY_SUB_PARALLEL_OVERLAP = 17
_FACTORY_SUB_PARALLEL_SIZE = 0
_SUB_PARALLEL_SIZE_MIN = 0


def _sub_parallel_enabled(
    merged: dict[str, object],
    sub_parallel_explicit: bool,
    first_architecture: str | None,
) -> bool:
    """``sub_parallel_windows`` vu par ``extra_settings`` après ``init_model_def``.

    ``query_model_def`` de Wan pose le drapeau à vrai seulement pour
    ``scail2``. ``animate2`` le remet à faux dans le même appel. Un JSON
    qui écrit la clé remplace l'injection.
    """
    if sub_parallel_explicit:
        return bool(merged.get("sub_parallel_windows"))
    return first_architecture in _SCAIL2_ARCHITECTURES


def _sub_parallel_bounds(
    merged: dict[str, object],
    sliding_defaults_explicit: bool,
) -> tuple[int, int, str]:
    """``overlap_min`` / ``overlap_max`` du recouvrement parallèle.

    Le réglage partage ``sliding_window_defaults`` avec
    ``sliding_window_overlap``. Sans cette clé, le plafond reste 97.
    ``scail2`` n'écrit pas de bornes : 17 passe.
    """
    if not sliding_defaults_explicit:
        return (
            _DEFAULT_OVERLAP_MIN,
            _DEFAULT_OVERLAP_MAX,
            "Sans sliding_window_defaults, extra_settings part de overlap_min 1 "
            "et overlap_max 97. scail2 injecte sub_parallel_windows sans resserrer "
            "ce plafond : le défaut 17 passe.",
        )
    raw_defaults = merged.get("sliding_window_defaults")
    defaults = raw_defaults if isinstance(raw_defaults, dict) else {}
    overlap_min = _overlap_bound(defaults.get("overlap_min", _DEFAULT_OVERLAP_MIN), _DEFAULT_OVERLAP_MIN)
    overlap_max = _overlap_bound(defaults.get("overlap_max", _DEFAULT_OVERLAP_MAX), _DEFAULT_OVERLAP_MAX)
    return overlap_min, overlap_max, "Les bornes viennent du JSON (sliding_window_defaults)."


def _sub_parallel_size_cap(
    root: Path | None,
    merged: dict[str, object],
    sliding_defaults_explicit: bool,
    first_architecture: str | None,
) -> tuple[int, str]:
    """Plafond de ``sub_parallel_window_size``.

    ``extra_settings`` fixe le minimum à 0. Le maximum est
    ``_sliding_window_size_max`` : ``window_max``, puis ``get_max_frames``.
    ``window_min`` ne s'applique pas à cette taille.
    """
    window_min, window_max, bounds_note = _window_bounds(
        merged,
        sliding_defaults_explicit,
        first_architecture,
    )
    multiplier = _max_frames_multiplier(root)
    widened_max = (window_max - 1) * multiplier + 1
    note = (
        f"{bounds_note} Le minimum de sub_parallel_window_size est "
        f"{_SUB_PARALLEL_SIZE_MIN}, pas window_min {window_min}. "
        f"Le maximum reprend window_max {window_max}."
    )
    if multiplier != 1:
        note += (
            f" get_max_frames élargit cette borne avec max_frames_multiplier {multiplier} : "
            f"le plafond devient {widened_max}."
        )
    return widened_max, note


def _require_sub_parallel_size(
    root: Path | None,
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    sub_parallel_explicit: bool,
    sliding_defaults_explicit: bool,
    queue_sub_parallel_window_size: object,
) -> None:
    """Refuse une taille parallèle hors de 0 et ``window_max``.

    ``extra_settings.validate_inputs`` contrôle ``sub_parallel_window_size``
    après ``sliding_window_size`` et avant ``sub_parallel_window_overlap``,
    dès que ``sub_parallel_windows`` est vrai. ``models/_settings.json``
    vaut 0 : cette taille passe. Une valeur 300 avec le ``window_max``
    d'usine 257 est refusée même si le recouvrement parallèle est dans
    la borne. ``sliding_window_size`` 129 reste sous 257, donc ce
    contrôle-là ne voit pas le saut. La tâche est sautée,
    ``process_tasks_cli`` sort 0, sans MP4.
    """
    if not _sub_parallel_enabled(merged, sub_parallel_explicit, first_architecture):
        return
    widened_max, bounds_note = _sub_parallel_size_cap(
        root,
        merged,
        sliding_defaults_explicit,
        first_architecture,
    )
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_sub_parallel_window_size is not _QUEUE_MODEL_MODE_UNSET
        and queue_sub_parallel_window_size is not None
    ):
        raw_size: object = queue_sub_parallel_window_size
        size_note = "La file envoie sub_parallel_window_size."
    elif "sub_parallel_window_size" in settings:
        raw_size = settings["sub_parallel_window_size"]
        size_note = "Le JSON de réglages envoie sub_parallel_window_size."
    else:
        raw_size = _FACTORY_SUB_PARALLEL_SIZE
        size_note = (
            "update_default_settings ne remplace pas sub_parallel_window_size. "
            f"models/_settings.json le laisse à {raw_size}."
        )
    shown = architecture_where or first_where or model_type
    if first_architecture in _SCAIL2_ARCHITECTURES and not sub_parallel_explicit:
        bounds_note += (
            f" query_model_def({first_architecture!r}) pose sub_parallel_windows "
            f"({first_where}). init_model_def n'est appelé que pour ce premier fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas le drapeau."
        )
    if _overlap_is_rejected(raw_size):
        kind = "a integer" if isinstance(raw_size, bool) else "an integer"
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sub_parallel_windows est vrai et "
            f"sub_parallel_window_size vaut {raw_size!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"(Sub Parallel Window Size must be {kind}) avant le recouvrement parallèle. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{size_note} {bounds_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    size = _sliding_overlap_number(raw_size)
    if size is None:
        return
    if _SUB_PARALLEL_SIZE_MIN <= size <= widened_max:
        return
    if size > widened_max:
        limit = f"au-dessus de {widened_max}"
        wan_message = f"must be at most {widened_max}"
    else:
        limit = f"en dessous de {_SUB_PARALLEL_SIZE_MIN}"
        wan_message = f"must be at least {_SUB_PARALLEL_SIZE_MIN}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sub_parallel_windows est vrai et "
        f"sub_parallel_window_size vaut {size}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"(Sub Parallel Window Size {wan_message}) avant le recouvrement parallèle. "
        "Le minimum est 0, pas window_min. sliding_window_size peut rester dans "
        "sa borne (129 sous un window_max de 257) : ce contrôle-là ne voit pas le saut. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{size_note} {bounds_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sub_parallel_window_size entre 0 et window_max, écrire "
        "sub_parallel_windows à faux, ou relever window_max."
    )


def _require_sub_parallel_overlap(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    sub_parallel_explicit: bool,
    sliding_defaults_explicit: bool,
    queue_sub_parallel_window_overlap: object,
) -> None:
    """Refuse un recouvrement parallèle hors de ``overlap_max``.

    ``extra_settings.validate_inputs`` contrôle ``sub_parallel_window_overlap``
    après ``sliding_window_size`` et ``sliding_window_overlap``, dès que
    ``sub_parallel_windows`` est vrai. ``models/_settings.json`` vaut 17.
    ``update_default_settings`` ne le change pas. Un ``overlap_max`` de 8
    laisse le recouvrement de fenêtre (5) et refuse 17. La tâche est
    sautée, ``process_tasks_cli`` sort 0, sans MP4. Le drapeau de
    ``scail2`` seul garde le plafond 97.
    """
    if not _sub_parallel_enabled(merged, sub_parallel_explicit, first_architecture):
        return
    overlap_min, overlap_max, bounds_note = _sub_parallel_bounds(merged, sliding_defaults_explicit)
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_sub_parallel_window_overlap is not _QUEUE_MODEL_MODE_UNSET
        and queue_sub_parallel_window_overlap is not None
    ):
        raw_overlap: object = queue_sub_parallel_window_overlap
        overlap_note = "La file envoie sub_parallel_window_overlap."
    elif "sub_parallel_window_overlap" in settings:
        raw_overlap = settings["sub_parallel_window_overlap"]
        overlap_note = "Le JSON de réglages envoie sub_parallel_window_overlap."
    else:
        raw_overlap = _FACTORY_SUB_PARALLEL_OVERLAP
        overlap_note = (
            "update_default_settings ne remplace pas sub_parallel_window_overlap. "
            f"models/_settings.json le laisse à {raw_overlap}."
        )
    shown = architecture_where or first_where or model_type
    if first_architecture in _SCAIL2_ARCHITECTURES and not sub_parallel_explicit:
        bounds_note += (
            f" query_model_def({first_architecture!r}) pose sub_parallel_windows "
            f"({first_where}). init_model_def n'est appelé que pour ce premier fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas le drapeau."
        )
    if _overlap_is_rejected(raw_overlap):
        kind = "a integer" if isinstance(raw_overlap, bool) else "an integer"
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sub_parallel_windows est vrai et "
            f"sub_parallel_window_overlap vaut {raw_overlap!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"(Sub Parallel Window Overlap must be {kind}) après sliding_window_overlap. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{overlap_note} {bounds_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    overlap = _sliding_overlap_number(raw_overlap)
    if overlap is None:
        return
    if overlap_min <= overlap <= overlap_max:
        return
    if overlap > overlap_max:
        limit = f"au-dessus de {overlap_max}"
        wan_message = f"must be at most {overlap_max}"
    else:
        limit = f"en dessous de {overlap_min}"
        wan_message = f"must be at least {overlap_min}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sub_parallel_windows est vrai et "
        f"sub_parallel_window_overlap vaut {overlap}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"(Sub Parallel Window Overlap {wan_message}) après sliding_window_overlap. "
        "Le recouvrement de fenêtre, 5 par défaut, peut rester dans la borne. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{overlap_note} {bounds_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sub_parallel_window_overlap dans la borne, écrire sub_parallel_windows "
        "à faux, ou relever overlap_max."
    )


_FACTORY_COLOR_STRENGTH = 0
_COLOR_STRENGTH_MIN = 0.0
_COLOR_STRENGTH_MAX = 1.0


def _color_correction_enabled(
    merged: dict[str, object],
    color_correction_explicit: bool,
    first_architecture: str | None,
) -> bool:
    """``color_correction`` vu par ``extra_settings`` après ``init_model_def``.

    ``query_model_def`` de Wan pose le drapeau à vrai pour toute
    architecture de ce handler. ``animate2`` le remet à faux dans le
    même appel. Un JSON qui écrit la clé remplace l'injection. Un
    finetune qui change l'architecture ne rappelle pas ``query_model_def``.
    """
    if color_correction_explicit:
        return bool(merged.get("color_correction"))
    if first_architecture == "animate2":
        return False
    return first_architecture in _WAN_ARCHITECTURES


def _color_strength_number(value: object) -> float | None:
    """Nombre accepté par ``validate_setting_value`` pour un type ``number``.

    ``None`` si la valeur est vide (pas d'erreur) ou si ce n'est pas un
    nombre. Le booléen est distingué : ``True`` n'est pas un nombre pour
    WanGP, même s'il est une sous-classe de ``int``.
    """
    if value is None:
        return None
    if isinstance(value, str) and len(value.strip()) == 0:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _color_strength_rejected(value: object) -> bool:
    """``True`` quand ``validate_setting_value`` refuse le type."""
    if value is None:
        return False
    if isinstance(value, str) and len(value.strip()) == 0:
        return False
    return _color_strength_number(value) is None


def _format_color_strength(value: float) -> str:
    """Même rendu que ``_format_number`` de ``extra_settings``."""
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _require_color_correction(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    color_correction_explicit: bool,
    queue_color_correction_strength: object,
) -> None:
    """Refuse une force de correction couleur hors de 0 et 1.

    ``extra_settings.validate_inputs`` contrôle
    ``sliding_window_color_correction_strength`` après
    ``sub_parallel_window_overlap``, dès que ``color_correction`` est
    vrai. ``models/_settings.json`` vaut 0 : cette force passe. Une
    valeur 2 ou 1.01 est refusée même si ``sub_parallel_window_size``
    vaut 0 et que le recouvrement parallèle tient dans la borne.
    ``validate_settings`` saute la tâche. ``process_tasks_cli`` sort 0,
    sans MP4. Un drapeau faux ignore une force 2. Si le recouvrement
    parallèle 17 dépasse ``overlap_max`` 8, cette erreur-là sort avant.
    """
    if not _color_correction_enabled(merged, color_correction_explicit, first_architecture):
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_color_correction_strength is not _QUEUE_MODEL_MODE_UNSET
        and queue_color_correction_strength is not None
    ):
        raw_strength: object = queue_color_correction_strength
        strength_note = "La file envoie sliding_window_color_correction_strength."
    elif "sliding_window_color_correction_strength" in settings:
        raw_strength = settings["sliding_window_color_correction_strength"]
        strength_note = "Le JSON de réglages envoie sliding_window_color_correction_strength."
    else:
        raw_strength = _FACTORY_COLOR_STRENGTH
        strength_note = (
            "update_default_settings ne sort pas sliding_window_color_correction_strength "
            f"de la borne. models/_settings.json le laisse à {raw_strength}. "
            "test_oneframe_overlap le réécrit à 0."
        )
    shown = architecture_where or first_where or model_type
    if color_correction_explicit:
        flag_note = "Le JSON écrit color_correction."
    else:
        flag_note = (
            f"query_model_def({first_architecture!r}) pose color_correction à vrai "
            f"({first_where}). animate2 le remet à faux dans ce même appel, pas dans "
            "un JSON suivant. init_model_def n'est appelé que pour ce premier fichier."
        )
    if _color_strength_rejected(raw_strength):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. color_correction est vrai et "
            f"sliding_window_color_correction_strength vaut {raw_strength!r}, qui n'est pas un nombre. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            "(Color Correction Strength must be a number) après sub_parallel_window_overlap. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{strength_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    strength = _color_strength_number(raw_strength)
    if strength is None:
        return
    if not (strength < _COLOR_STRENGTH_MIN or strength > _COLOR_STRENGTH_MAX):
        return
    shown_strength = _format_color_strength(strength)
    if strength > _COLOR_STRENGTH_MAX:
        limit = f"au-dessus de {_format_color_strength(_COLOR_STRENGTH_MAX)}"
        wan_message = f"must be at most {_format_color_strength(_COLOR_STRENGTH_MAX)}"
    else:
        limit = f"en dessous de {_format_color_strength(_COLOR_STRENGTH_MIN)}"
        wan_message = f"must be at least {_format_color_strength(_COLOR_STRENGTH_MIN)}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. color_correction est vrai et "
        f"sliding_window_color_correction_strength vaut {shown_strength}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"(Color Correction Strength {wan_message}) après sub_parallel_window_overlap. "
        "Le défaut d'usine 0 passe. sub_parallel_window_size 0 et un recouvrement "
        "parallèle dans la borne ne voient pas ce saut. Si ce recouvrement vaut 17 "
        "avec overlap_max 8, cette erreur-là sort avant. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{strength_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sliding_window_color_correction_strength entre 0 et 1, ou écrire "
        "color_correction à faux."
    )


_FACTORY_OVERLAP_NOISE = 0
_OVERLAP_NOISE_MIN = 0
_OVERLAP_NOISE_MAX = 150
_OVERLAP_NOISE_LABEL = "Noise to be added to overlapped frames to reduce blur effect"


def _vace_class_enabled(
    merged: dict[str, object],
    vace_class_explicit: bool,
    first_architecture: str | None,
) -> bool:
    """``vace_class`` vu par ``extra_settings`` après ``init_model_def``.

    ``query_model_def`` de Wan pose le drapeau pour ``test_vace``
    (``vace_14B``, ``vace_1.3B`` et la famille). Un JSON qui écrit la clé
    remplace l'injection. Un finetune qui change l'architecture ne
    rappelle pas ``query_model_def``.
    """
    if vace_class_explicit:
        return bool(merged.get("vace_class"))
    return first_architecture in _WAN_VACE


def _overlap_noise_visible(
    merged: dict[str, object],
    vace_class_explicit: bool,
    first_architecture: str | None,
    model_type: str,
) -> bool:
    """Visibilité de ``sliding_window_overlap_noise``.

    Le drapeau ``vace_class`` suffit. Sinon, ``validate_inputs`` lit
    ``model_type`` dans les réglages de la file (pas le champ
    ``architecture`` du JSON) et le rend visible s'il contient
    ``sky_df`` ou ``diffusion_forcing``.
    """
    if _vace_class_enabled(merged, vace_class_explicit, first_architecture):
        return True
    lowered = model_type.lower()
    return "sky_df" in lowered or "diffusion_forcing" in lowered


def _extra_integer(value: object) -> tuple[str | None, int | None]:
    """Entier accepté par ``validate_setting_value``.

    Le premier élément est ``a integer`` ou ``an integer`` quand le type
    est refusé, sinon None. Le second est l'entier retenu. ``int`` tronque
    un flottant : 150.5 devient 150 et reste dans la borne. Une chaîne
    n'est un entier que si ``float`` tombe juste (``151.0``). Un booléen
    n'est pas un entier. Vide et None ne sont pas une erreur.
    """
    if value is None:
        return None, None
    if isinstance(value, str) and len(value.strip()) == 0:
        return None, None
    if isinstance(value, bool):
        return "a integer", None
    if isinstance(value, int):
        return None, value
    if isinstance(value, float):
        return None, int(value)
    if isinstance(value, str):
        try:
            return None, int(value)
        except ValueError:
            try:
                parsed = float(value.strip())
            except ValueError:
                return "an integer", None
            if not parsed.is_integer():
                return "an integer", None
            return None, int(parsed)
    try:
        return None, int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "an integer", None


def _require_overlap_noise(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    vace_class_explicit: bool,
    queue_overlap_noise: object,
) -> None:
    """Refuse un bruit de recouvrement hors de 0 et 150.

    ``extra_settings.validate_inputs`` contrôle
    ``sliding_window_overlap_noise`` après
    ``sliding_window_color_correction_strength``, dès que ``vace_class``
    est vrai ou que ``model_type`` contient ``sky_df`` ou
    ``diffusion_forcing``. ``models/_settings.json`` vaut 0 : ce bruit
    passe. Une valeur 151 est refusée même si la force de couleur vaut
    0, ``sub_parallel_window_size`` vaut 0 et le recouvrement parallèle
    tient dans la borne. ``validate_settings`` saute la tâche.
    ``process_tasks_cli`` sort 0, sans MP4. Si ``color_correction`` est
    vrai et que la force vaut 2, cette erreur-là sort avant.
    """
    if not _overlap_noise_visible(merged, vace_class_explicit, first_architecture, model_type):
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if queue_overlap_noise is not _QUEUE_MODEL_MODE_UNSET and queue_overlap_noise is not None:
        raw_noise: object = queue_overlap_noise
        noise_note = "La file envoie sliding_window_overlap_noise."
    elif "sliding_window_overlap_noise" in settings:
        raw_noise = settings["sliding_window_overlap_noise"]
        noise_note = "Le JSON de réglages envoie sliding_window_overlap_noise."
    else:
        raw_noise = _FACTORY_OVERLAP_NOISE
        noise_note = (
            "update_default_settings ne remplace pas sliding_window_overlap_noise. "
            f"models/_settings.json le laisse à {raw_noise}."
        )
    shown = architecture_where or first_where or model_type
    if _vace_class_enabled(merged, vace_class_explicit, first_architecture):
        if vace_class_explicit:
            flag_note = "Le JSON écrit vace_class."
        else:
            flag_note = (
                f"query_model_def({first_architecture!r}) pose vace_class à vrai "
                f"({first_where}). init_model_def n'est appelé que pour ce premier fichier. "
                "Un JSON suivant qui change l'architecture ne retire pas le drapeau."
            )
    else:
        flag_note = (
            f"model_type {model_type!r} contient sky_df ou diffusion_forcing. "
            "Le réglage est visible sans vace_class. L'architecture du JSON ne compte pas."
        )
    kind, noise = _extra_integer(raw_noise)
    if kind is not None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sliding_window_overlap_noise "
            f"vaut {raw_noise!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({_OVERLAP_NOISE_LABEL} must be {kind}) après "
            "sliding_window_color_correction_strength. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{noise_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    if noise is None:
        return
    if _OVERLAP_NOISE_MIN <= noise <= _OVERLAP_NOISE_MAX:
        return
    if noise > _OVERLAP_NOISE_MAX:
        limit = f"au-dessus de {_OVERLAP_NOISE_MAX}"
        wan_message = f"must be at most {_OVERLAP_NOISE_MAX}"
    else:
        limit = f"en dessous de {_OVERLAP_NOISE_MIN}"
        wan_message = f"must be at least {_OVERLAP_NOISE_MIN}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sliding_window_overlap_noise "
        f"vaut {noise}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({_OVERLAP_NOISE_LABEL} {wan_message}) après "
        "sliding_window_color_correction_strength. "
        "Le défaut d'usine 0 passe. La force de couleur 0, sub_parallel_window_size 0 "
        "et un recouvrement parallèle dans la borne ne voient pas ce saut. "
        "Si color_correction est vrai et que la force vaut 2, cette erreur-là sort avant. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{noise_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sliding_window_overlap_noise entre 0 et 150, ou retirer vace_class "
        "et un model_type sky_df ou diffusion_forcing."
    )


_FACTORY_DISCARD_LAST = 0
_MULTITALK_DISCARD_LAST = 4
_DISCARD_LAST_MIN = 0
_DISCARD_LAST_MAX = 20
_DISCARD_LAST_LABEL = "Discard Last Frames of a Window"


def _discard_last_frames_visible(
    merged: dict[str, object],
    sliding_window_explicit: bool,
    first_architecture: str | None,
    model_type: str,
) -> bool:
    """Visibilité de ``sliding_window_discard_last_frames``.

    ``sliding_window`` doit être vrai. ``model_type`` (la file, pas le
    champ ``architecture``) cache le réglage s'il contient ``sky_df`` ou
    ``diffusion_forcing``. ``query_model_def`` de ``t2v_1.3B`` pose le
    drapeau à faux : il faut l'écrire, ou un premier JSON qui l'injecte.
    """
    if not _sliding_size_enabled(merged, sliding_window_explicit, first_architecture):
        return False
    lowered = model_type.lower()
    return "sky_df" not in lowered and "diffusion_forcing" not in lowered


def _discard_scheduler_int_crashes(value: object) -> bool:
    """``int(value or 0)`` du frame scheduler, avant ``extra_settings``.

    ``validate_settings`` l'exécute quand ``image_mode`` est 0 et
    ``sliding_window`` est vrai. ``large``, ``21.5`` et ``21.0`` lèvent
    ``ValueError``. Le processus ne sort pas 0. ``extra_settings`` n'est
    pas atteint.
    """
    try:
        int(value or 0)
    except (TypeError, ValueError):
        return True
    return False


def _factory_discard_last_frames(architecture: str) -> int:
    """``sliding_window_discard_last_frames`` après ``update_default_settings``.

    ``models/_settings.json`` part de 0. ``multitalk`` le pose à 4, dans
    la borne. ``vace_14B`` et ``vace_multitalk_14B`` réécrivent 0.
    """
    if architecture == "multitalk":
        return _MULTITALK_DISCARD_LAST
    return _FACTORY_DISCARD_LAST


def _require_discard_last_frames(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    sliding_window_explicit: bool,
    queue_discard_last_frames: object,
) -> None:
    """Refuse un rejet de fenêtres hors de 0 et 20.

    ``extra_settings.validate_inputs`` contrôle
    ``sliding_window_discard_last_frames`` après
    ``sliding_window_overlap_noise``, dès que ``sliding_window`` est vrai
    et que ``model_type`` ne contient ni ``sky_df`` ni
    ``diffusion_forcing``. ``models/_settings.json`` vaut 0 : ce rejet
    passe. Une valeur 21 est refusée même si le bruit vaut 0, la force
    de couleur vaut 0 et le recouvrement parallèle tient dans la borne.
    ``validate_settings`` saute la tâche. ``process_tasks_cli`` sort 0,
    sans MP4. Si le bruit 151 est visible, cette erreur-là sort avant.
    Une chaîne que le ``int()`` du frame scheduler refuse n'est pas ce
    saut : le processus ne sort pas 0.
    """
    if not _discard_last_frames_visible(
        merged,
        sliding_window_explicit,
        first_architecture,
        model_type,
    ):
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_discard_last_frames is not _QUEUE_MODEL_MODE_UNSET
        and queue_discard_last_frames is not None
    ):
        raw_discard: object = queue_discard_last_frames
        discard_note = "La file envoie sliding_window_discard_last_frames."
    elif "sliding_window_discard_last_frames" in settings:
        raw_discard = settings["sliding_window_discard_last_frames"]
        discard_note = "Le JSON de réglages envoie sliding_window_discard_last_frames."
    else:
        raw_discard = _factory_discard_last_frames(architecture)
        if architecture == "multitalk":
            discard_note = (
                "update_default_settings de multitalk pose "
                f"sliding_window_discard_last_frames à {raw_discard}. "
                "Cette valeur est dans la borne."
            )
        else:
            discard_note = (
                "update_default_settings ne remplace pas "
                "sliding_window_discard_last_frames. "
                f"models/_settings.json le laisse à {raw_discard}."
            )
    shown = architecture_where or first_where or model_type
    if sliding_window_explicit:
        flag_note = "Le JSON écrit sliding_window."
    else:
        flag_note = (
            f"query_model_def({first_architecture!r}) pose sliding_window à vrai "
            f"({first_where}). init_model_def n'est appelé que pour ce premier fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas le drapeau. "
            "query_model_def de t2v_1.3B le laisse faux."
        )
    if _discard_scheduler_int_crashes(raw_discard):
        return
    kind, discarded = _extra_integer(raw_discard)
    if kind is not None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
            f"sliding_window_discard_last_frames vaut {raw_discard!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({_DISCARD_LAST_LABEL} must be {kind}) après sliding_window_overlap_noise. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{discard_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    if discarded is None:
        return
    if _DISCARD_LAST_MIN <= discarded <= _DISCARD_LAST_MAX:
        return
    if discarded > _DISCARD_LAST_MAX:
        limit = f"au-dessus de {_DISCARD_LAST_MAX}"
        wan_message = f"must be at most {_DISCARD_LAST_MAX}"
    else:
        limit = f"en dessous de {_DISCARD_LAST_MIN}"
        wan_message = f"must be at least {_DISCARD_LAST_MIN}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
        f"sliding_window_discard_last_frames vaut {discarded}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({_DISCARD_LAST_LABEL} {wan_message}) après sliding_window_overlap_noise. "
        "Le défaut d'usine 0 passe. Le bruit 0, la force de couleur 0, "
        "sub_parallel_window_size 0 et un recouvrement parallèle dans la borne "
        "ne voient pas ce saut. Si le bruit 151 est visible, cette erreur-là sort avant. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{discard_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sliding_window_discard_last_frames entre 0 et 20, ou écrire "
        "sliding_window à faux. sky_df ou diffusion_forcing dans model_type cache le contrôle."
    )


_FACTORY_TRIM_FIRST = 0
_TRIM_FIRST_MIN = 0
_TRIM_FIRST_MAX = 10
_TRIM_FIRST_LABEL = "Trim First Frames in First Window or if there is no Overlap Frames"


def _scheduler_discard_step(merged: dict[str, object]) -> object:
    """``discard_step`` du frame scheduler, avant ``extra_settings``.

    ``validate_settings`` lit ``sliding_window_defaults.discard_step``,
    sinon ``latent_size``, sinon ``frames_steps`` (4). Le JSON gagne sur
    l'injection : ``init_model_def`` fait ``default_model_def.update``.
    """
    defaults = merged.get("sliding_window_defaults")
    if isinstance(defaults, dict) and "discard_step" in defaults:
        return defaults["discard_step"]
    frames_steps = merged.get("frames_steps", 4)
    return merged.get("latent_size", frames_steps)


def _discard_step_breaks_scheduler(step: object) -> bool:
    """``int(rejet) // discard_step`` lève avant ``extra_settings``.

    0 lève ``ZeroDivisionError``. Un texte ou ``None`` lève ``TypeError``.
    ``False`` vaut 0. Le processus ne sort pas 0.
    """
    if isinstance(step, bool):
        return not step
    if isinstance(step, (int, float)):
        return step == 0
    return True


def _raw_discard_for_scheduler(
    merged: dict[str, object],
    settings: dict[str, object],
    queue_discard_last_frames: object,
) -> object:
    """Rejet vu par ``int(inputs.get(...) or 0)`` du frame scheduler."""
    if (
        queue_discard_last_frames is not _QUEUE_MODEL_MODE_UNSET
        and queue_discard_last_frames is not None
    ):
        return queue_discard_last_frames
    if "sliding_window_discard_last_frames" in settings:
        return settings["sliding_window_discard_last_frames"]
    architecture = merged.get("architecture")
    if isinstance(architecture, str) and architecture:
        return _factory_discard_last_frames(architecture)
    return _FACTORY_DISCARD_LAST


def _frame_scheduler_skips_extra_settings(
    merged: dict[str, object],
    settings: dict[str, object],
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
) -> bool:
    """Le frame scheduler lève avant le contrôle du trim.

    Il tourne quand ``image_mode`` est 0 et ``sliding_window`` est vrai.
    ``int(rejet or 0)`` refuse ``large`` et ``21.5``. La division par
    ``discard_step`` 0 lève aussi. ``extra_settings`` n'est pas atteint :
    un trim 11 n'est pas un code 0 dans ces cas.
    """
    if not _sliding_size_enabled(merged, sliding_window_explicit, first_architecture):
        return False
    raw_discard = _raw_discard_for_scheduler(merged, settings, queue_discard_last_frames)
    if _discard_scheduler_int_crashes(raw_discard):
        return True
    return _discard_step_breaks_scheduler(_scheduler_discard_step(merged))


def _require_trim_first_frames(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_where: str,
    architecture_where: str,
    sliding_window_explicit: bool,
    queue_discard_last_frames: object,
    queue_trim_first_frames: object,
) -> None:
    """Refuse un rognage des premières images hors de 0 et 10.

    ``extra_settings.validate_inputs`` contrôle
    ``sliding_window_trim_first_frames`` après
    ``sliding_window_discard_last_frames``, dès que ``sliding_window``
    est vrai. ``sky_df`` et ``diffusion_forcing`` cachent le rejet, pas
    ce rognage. ``models/_settings.json`` vaut 0 : ce trim passe.
    ``models/wan/wan_handler.py`` ne réécrit pas la clé. Une valeur 11
    est refusée même si le rejet vaut 0, le bruit vaut 0 et la force de
    couleur vaut 0. ``validate_settings`` saute la tâche.
    ``process_tasks_cli`` sort 0, sans MP4. Si le rejet 21 est visible,
    cette erreur-là sort avant. ``large`` n'est pas lu par le ``int()``
    du frame scheduler : ``extra_settings`` renvoie ``must be an integer``.
    """
    if not _sliding_size_enabled(merged, sliding_window_explicit, first_architecture):
        return
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if (
        queue_trim_first_frames is not _QUEUE_MODEL_MODE_UNSET
        and queue_trim_first_frames is not None
    ):
        raw_trim: object = queue_trim_first_frames
        trim_note = "La file envoie sliding_window_trim_first_frames."
    elif "sliding_window_trim_first_frames" in settings:
        raw_trim = settings["sliding_window_trim_first_frames"]
        trim_note = "Le JSON de réglages envoie sliding_window_trim_first_frames."
    else:
        raw_trim = _FACTORY_TRIM_FIRST
        trim_note = (
            "update_default_settings ne remplace pas sliding_window_trim_first_frames. "
            f"models/_settings.json le laisse à {raw_trim}."
        )
    shown = architecture_where or first_where or model_type
    if sliding_window_explicit:
        flag_note = "Le JSON écrit sliding_window."
    else:
        flag_note = (
            f"query_model_def({first_architecture!r}) pose sliding_window à vrai "
            f"({first_where}). init_model_def n'est appelé que pour ce premier fichier. "
            "Un JSON suivant qui change l'architecture ne retire pas le drapeau. "
            "query_model_def de t2v_1.3B le laisse faux."
        )
    kind, trimmed = _extra_integer(raw_trim)
    if kind is not None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
            f"sliding_window_trim_first_frames vaut {raw_trim!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({_TRIM_FIRST_LABEL} must be {kind}) après sliding_window_discard_last_frames. "
            "Ce trim n'est pas passé au int() du frame scheduler : large et 11.5 "
            "en chaîne ne lèvent pas ValueError. validate_settings propage l'erreur. "
            "process_tasks_cli compte la tâche comme sautée et quitte avec le code 0, "
            "sans MP4. "
            f"{trim_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    if trimmed is None:
        return
    if _TRIM_FIRST_MIN <= trimmed <= _TRIM_FIRST_MAX:
        return
    if trimmed > _TRIM_FIRST_MAX:
        limit = f"au-dessus de {_TRIM_FIRST_MAX}"
        wan_message = f"must be at most {_TRIM_FIRST_MAX}"
    else:
        limit = f"en dessous de {_TRIM_FIRST_MIN}"
        wan_message = f"must be at least {_TRIM_FIRST_MIN}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. sliding_window est vrai et "
        f"sliding_window_trim_first_frames vaut {trimmed}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({_TRIM_FIRST_LABEL} {wan_message}) après sliding_window_discard_last_frames. "
        "Le défaut d'usine 0 passe. Le rejet 0, le bruit 0, la force de couleur 0, "
        "sub_parallel_window_size 0 et un recouvrement parallèle dans la borne "
        "ne voient pas ce saut. sky_df et diffusion_forcing cachent le rejet, pas ce trim. "
        "Si le rejet 21 est visible, cette erreur-là sort avant. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{trim_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser sliding_window_trim_first_frames entre 0 et 10, ou écrire "
        "sliding_window à faux."
    )


_FACTORY_TOP_P = 0.9
_TOP_P_MIN = 0.0
_TOP_P_MAX = 1.0


def _top_p_enabled(merged: dict[str, object], explicit: bool) -> bool:
    """``top_p_slider`` vu par ``extra_settings`` après ``init_model_def``.

    ``query_model_def`` de Wan ne pose pas le drapeau. Un JSON qui écrit
    la clé décide. ``bool`` suit WanGP : ``1`` et la chaîne ``false`` sont
    vrais, ``0`` est faux. ``top_k_slider`` seul laisse ``top_p`` invisible.
    """
    if not explicit:
        return False
    return bool(merged.get("top_p_slider"))


def _require_top_p(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    top_p_slider_explicit: bool,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_top_p: object,
) -> None:
    """Refuse un top-p hors de 0 et 1.

    ``extra_settings.validate_inputs`` contrôle ``top_p`` après
    ``temperature``, dès que ``top_p_slider`` est vrai. Le conteneur
    ``top_pk_row`` l'est aussi dans ce cas. ``models/_settings.json``
    vaut 0,9 : cette valeur passe. Une valeur 1,01 ou 2 est refusée
    même si ``sliding_window`` est faux et que le trim vaut 0.
    ``validate_settings`` saute la tâche. ``process_tasks_cli`` sort 0,
    sans MP4. ``temperature`` n'est visible qu'avec ``audio_only``, déjà
    refusé. Si le trim 11 est visible, cette erreur-là sort avant.
    Le frame scheduler qui lève n'atteint pas ``extra_settings``.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if not _top_p_enabled(merged, top_p_slider_explicit):
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if queue_top_p is not _QUEUE_MODEL_MODE_UNSET and queue_top_p is not None:
        raw_top_p: object = queue_top_p
        top_p_note = "La file envoie top_p."
    elif "top_p" in settings:
        raw_top_p = settings["top_p"]
        top_p_note = "Le JSON de réglages envoie top_p."
    else:
        raw_top_p = _FACTORY_TOP_P
        top_p_note = (
            "update_default_settings de Wan ne remplace pas top_p. "
            f"models/_settings.json le laisse à {raw_top_p}."
        )
    shown = architecture_where or first_where or model_type
    flag_note = (
        "Le JSON écrit top_p_slider. query_model_def de Wan ne pose pas ce drapeau. "
        "top_k_slider seul ne rend pas top_p visible. "
        "t2v_1.3B sans la clé laisse le réglage ignoré."
    )
    if _color_strength_rejected(raw_top_p):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. top_p_slider est vrai et "
            f"top_p vaut {raw_top_p!r}, qui n'est pas un nombre. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            "(Top-p must be a number) après temperature. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{top_p_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    top_p = _color_strength_number(raw_top_p)
    if top_p is None:
        return
    if not (top_p < _TOP_P_MIN or top_p > _TOP_P_MAX):
        return
    shown_top_p = _format_color_strength(top_p)
    if top_p > _TOP_P_MAX:
        limit = f"au-dessus de {_format_color_strength(_TOP_P_MAX)}"
        wan_message = f"must be at most {_format_color_strength(_TOP_P_MAX)}"
    else:
        limit = f"en dessous de {_format_color_strength(_TOP_P_MIN)}"
        wan_message = f"must be at least {_format_color_strength(_TOP_P_MIN)}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. top_p_slider est vrai et "
        f"top_p vaut {shown_top_p}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"(Top-p {wan_message}) après temperature. "
        "Le défaut d'usine 0.9 passe. Le trim 0 et sliding_window faux ne voient "
        "pas ce saut. Si le trim 11 est visible, cette erreur-là sort avant. "
        "Une température hors borne est ignorée tant qu'audio_only est faux. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{top_p_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser top_p entre 0 et 1, ou écrire top_p_slider à faux."
    )


_FACTORY_TOP_K = 50
_TOP_K_MIN = 0
_TOP_K_MAX = 100
_TOP_K_LABEL = "Top-k"


def _top_k_enabled(merged: dict[str, object], explicit: bool) -> bool:
    """``top_k_slider`` vu par ``extra_settings`` après ``init_model_def``.

    ``query_model_def`` de Wan ne pose pas le drapeau. Un JSON qui écrit
    la clé décide. ``bool`` suit WanGP : ``1`` et la chaîne ``false`` sont
    vrais, ``0`` est faux. ``top_p_slider`` seul laisse ``top_k`` invisible.
    Le conteneur ``top_pk_row`` suit les deux drapeaux, mais le réglage
    lui-même n'est visible qu'avec ``top_k_slider``.
    """
    if not explicit:
        return False
    return bool(merged.get("top_k_slider"))


def _require_top_k(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    top_k_slider_explicit: bool,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_top_k: object,
) -> None:
    """Refuse un top-k hors de 0 et 100.

    ``extra_settings.validate_inputs`` contrôle ``top_k`` après ``top_p``,
    dès que ``top_k_slider`` est vrai. Le libellé d'usine est
    ``Top-k (0 = disabled)``. ``_trim_label`` le coupe au premier ``(`` :
    l'erreur dit ``Top-k``. ``models/_settings.json`` vaut 50 : cette
    valeur passe. Une valeur 101 est refusée même si ``top_p`` vaut 0,9
    et que ``top_p_slider`` est faux. ``int`` tronque un flottant :
    100,9 reste dans la borne, 101,5 devient 101. ``validate_settings``
    saute la tâche. ``process_tasks_cli`` sort 0, sans MP4. Si ``top_p``
    2 est visible, cette erreur-là sort avant. ``NAG_scale`` vient après,
    seulement si ``NAG`` est vrai. Le frame scheduler qui lève n'atteint
    pas ``extra_settings``.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if not _top_k_enabled(merged, top_k_slider_explicit):
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if queue_top_k is not _QUEUE_MODEL_MODE_UNSET and queue_top_k is not None:
        raw_top_k: object = queue_top_k
        top_k_note = "La file envoie top_k."
    elif "top_k" in settings:
        raw_top_k = settings["top_k"]
        top_k_note = "Le JSON de réglages envoie top_k."
    else:
        raw_top_k = _FACTORY_TOP_K
        top_k_note = (
            "update_default_settings de Wan ne remplace pas top_k. "
            f"models/_settings.json le laisse à {raw_top_k}."
        )
    shown = architecture_where or first_where or model_type
    flag_note = (
        "Le JSON écrit top_k_slider. query_model_def de Wan ne pose pas ce drapeau. "
        "top_p_slider seul ne rend pas top_k visible. "
        "t2v_1.3B sans la clé laisse le réglage ignoré."
    )
    kind, top_k = _extra_integer(raw_top_k)
    if kind is not None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. top_k_slider est vrai et "
            f"top_k vaut {raw_top_k!r}, qui n'est pas un entier. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({_TOP_K_LABEL} must be {kind}) après top_p. "
            "_trim_label coupe le libellé au premier parenthèse : le message dit Top-k. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{top_k_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    if top_k is None:
        return
    if _TOP_K_MIN <= top_k <= _TOP_K_MAX:
        return
    if top_k > _TOP_K_MAX:
        limit = f"au-dessus de {_TOP_K_MAX}"
        wan_message = f"must be at most {_TOP_K_MAX}"
    else:
        limit = f"en dessous de {_TOP_K_MIN}"
        wan_message = f"must be at least {_TOP_K_MIN}"
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. top_k_slider est vrai et "
        f"top_k vaut {top_k}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({_TOP_K_LABEL} {wan_message}) après top_p. "
        "_trim_label coupe le libellé au premier parenthèse : le message dit Top-k. "
        "Le défaut d'usine 50 passe. int tronque un flottant : 100.9 reste dans la borne, "
        "101.5 devient 101. top_p_slider faux et un top_p 2 ne voient pas ce saut. "
        "Si top_p 2 est visible, cette erreur-là sort avant. NAG_scale vient après, "
        "seulement si NAG est vrai : une valeur 21 n'est pas ce saut. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{top_k_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser top_k entre 0 et 100, ou écrire top_k_slider à faux."
    )


_FACTORY_NAG_SCALE = 1.0
_NAG_SCALE_MIN = 1.0
_NAG_SCALE_MAX = 20.0
_NAG_SCALE_LABEL = "NAG Scale"
_FACTORY_NAG_TAU = 3.5
_NAG_TAU_MIN = 1.0
_NAG_TAU_MAX = 5.0
_NAG_TAU_LABEL = "NAG Tau"
_FACTORY_NAG_ALPHA = 0.5
_NAG_ALPHA_MIN = 0.0
_NAG_ALPHA_MAX = 2.0
_NAG_ALPHA_LABEL = "NAG Alpha"


def _injected_nag(architecture: str | None) -> bool:
    """``NAG`` posé par ``wan_handler.query_model_def`` (ligne 973).

    La formule finale écrase les ``NAG: False`` de ``bernini`` et
    ``animate2`` plus haut dans le même appel. ``vista4d`` est dans
    ``test_class_t2v`` : le faux écrit pour la caméra ne tient pas.
    ``t2v_1.3B`` n'est pas dans cette classe. ``init_model_def`` n'est
    appelé que pour le premier JSON, puis ``default_model_def.update``
    laisse une clé écrite remplacer l'injection.
    """
    if architecture is None or architecture not in _WAN_ARCHITECTURES:
        return False
    visible = (
        architecture in _WAN_VACE
        or architecture in _WAN_T2V_CLASS
        or architecture in _WAN_I2V
    )
    blocked = architecture in _WAN_BERNINI or architecture == "animate2"
    return visible and not blocked


def _nag_enabled(
    merged: dict[str, object],
    explicit: bool,
    first_architecture: str | None,
) -> bool:
    """``NAG`` vu par le conteneur ``NAG_col`` après ``init_model_def``.

    ``bool`` suit WanGP : ``1`` et la chaîne ``false`` sont vrais, ``0``
    est faux. Un JSON qui écrit la clé remplace l'injection. Un finetune
    qui change l'architecture ne rappelle pas ``query_model_def``.
    """
    if explicit:
        return bool(merged.get("NAG"))
    return _injected_nag(first_architecture)


def _trim_setting_label(label: str, fallback: str) -> str:
    """``_trim_label`` de ``shared/extra_settings.py``."""
    collapsed = " ".join(str(label or "").strip().split())
    trimmed = collapsed.split("(", 1)[0].strip()
    return trimmed or fallback


def _normalize_setting_number(value: object) -> float | None:
    """``_normalize_number`` de ``shared/extra_settings.py``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed


def _nag_number_spec(
    merged: dict[str, object],
    key: str,
    fallback_label: str,
    default_min: float,
    default_max: float,
) -> tuple[str, float | None, float | None] | None:
    """Libellé et bornes d'un réglage NAG, ou None si ``get_def`` lève.

    ``_model_setting_config`` fait ``.get`` sur la valeur du modèle. Un
    objet qui n'est pas un dict, ou un ``label`` qui n'est pas du texte,
    lève ``AttributeError``. ``iter_defs`` résout chaque définition avant
    de valider : un objet illisible sur ``NAG_alpha`` empêche aussi
    l'erreur de ``NAG_tau``. Le processus ne sort pas 0. Un dict peut
    remplacer ``min``, ``max`` et ``label``. Une borne illisible devient
    None et le contrôle correspondant disparaît.
    """
    if key not in merged:
        config: object = {}
    else:
        config = merged[key]
    if not isinstance(config, dict):
        return None
    if "label" in config:
        raw_label = config["label"]
        if not isinstance(raw_label, str):
            return None
        label_text = raw_label.strip() or fallback_label
    else:
        label_text = fallback_label
    label = _trim_setting_label(label_text, key)
    minimum = _normalize_setting_number(config.get("min", default_min))
    maximum = _normalize_setting_number(config.get("max", default_max))
    return label, minimum, maximum


def _nag_scale_spec(
    merged: dict[str, object],
) -> tuple[str, float | None, float | None] | None:
    """Libellé et bornes de ``NAG_scale``, ou None si WanGP lève avant."""
    return _nag_number_spec(
        merged,
        "NAG_scale",
        _NAG_SCALE_LABEL,
        _NAG_SCALE_MIN,
        _NAG_SCALE_MAX,
    )


def _require_nag_scale(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    nag_explicit: bool,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_nag_scale: object,
) -> None:
    """Refuse un NAG scale hors de 1 et 20.

    ``extra_settings.validate_inputs`` contrôle ``NAG_scale`` après
    ``top_k``, dès que le conteneur ``NAG_col`` est visible, donc dès
    que ``NAG`` est vrai. Le type est un nombre. ``models/_settings.json``
    vaut 1, ``NAG_tau`` 3,5 et ``NAG_alpha`` 0,5 : ces valeurs passent.
    21 et ``20.01`` sont refusés. ``0.5`` est sous 1. Un objet
    ``model.NAG_scale`` peut relever le maximum (``{"max": 30}`` laisse
    passer 21) ou le baisser (``{"max": 10}`` refuse 11) et remplacer
    le libellé. ``validate_settings`` saute la tâche.
    ``process_tasks_cli`` sort 0, sans MP4. Si ``top_k`` 101 est visible,
    cette erreur-là sort avant. ``NAG_tau`` vient ensuite et n'est pas
    ce saut. Le frame scheduler qui lève n'atteint pas ``extra_settings``.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if not _nag_enabled(merged, nag_explicit, first_architecture):
        return
    spec = _nag_scale_spec(merged)
    if spec is None:
        return
    label, minimum, maximum = spec
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if queue_nag_scale is not _QUEUE_MODEL_MODE_UNSET and queue_nag_scale is not None:
        raw_scale: object = queue_nag_scale
        scale_note = "La file envoie NAG_scale."
    elif "NAG_scale" in settings:
        raw_scale = settings["NAG_scale"]
        scale_note = "Le JSON de réglages envoie NAG_scale."
    else:
        raw_scale = _FACTORY_NAG_SCALE
        scale_note = (
            "update_default_settings de Wan ne remplace pas NAG_scale. "
            f"models/_settings.json le laisse à {int(_FACTORY_NAG_SCALE)}."
        )
    shown = architecture_where or first_where or model_type
    if nag_explicit:
        flag_note = (
            "Le JSON écrit NAG. bool de WanGP : 1 et la chaîne false sont vrais, "
            "0 est faux."
        )
    else:
        flag_note = (
            f"query_model_def({first_architecture!r}) pose NAG à vrai "
            f"({first_where}). La formule est (vace_class ou test_class_t2v ou "
            "test_class_i2v) et pas (bernini ou animate2). t2v_1.3B n'est pas "
            "dans test_class_t2v. init_model_def n'est appelé que pour ce premier "
            "fichier. Un JSON suivant qui change l'architecture ne retire pas le drapeau."
        )
    if isinstance(merged.get("NAG_scale"), dict):
        flag_note += " Les bornes viennent de l'objet model.NAG_scale."
    else:
        flag_note += " Les bornes d'usine sont 1 et 20."
    if _color_strength_rejected(raw_scale):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. NAG est vrai et "
            f"NAG_scale vaut {raw_scale!r}, qui n'est pas un nombre. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({label} must be a number) après top_k. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{scale_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    scale = _color_strength_number(raw_scale)
    if scale is None:
        return
    too_low = minimum is not None and scale < minimum
    too_high = maximum is not None and scale > maximum
    if not too_low and not too_high:
        return
    shown_scale = _format_color_strength(scale)
    if too_low and minimum is not None:
        shown_limit = _format_color_strength(minimum)
        limit = f"en dessous de {shown_limit}"
        wan_message = f"must be at least {shown_limit}"
    elif too_high and maximum is not None:
        shown_limit = _format_color_strength(maximum)
        limit = f"au-dessus de {shown_limit}"
        wan_message = f"must be at most {shown_limit}"
    else:
        return
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. NAG est vrai et "
        f"NAG_scale vaut {shown_scale}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({label} {wan_message}) après top_k. "
        "Le défaut d'usine 1 passe, ainsi que 20. 20.01 non. 0.5 est sous 1. "
        "Un objet model.NAG_scale peut changer le libellé et les bornes : "
        "max 30 laisse passer 21, max 10 refuse 11. "
        "top_k_slider faux et un top_k 101 ne voient pas ce saut. "
        "Si top_k 101 est visible, cette erreur-là sort avant. "
        "NAG_tau et NAG_alpha viennent après : une valeur hors borne n'est pas ce saut. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{scale_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser NAG_scale dans la borne, écrire NAG à faux, ou relever max dans "
        "l'objet model.NAG_scale."
    )


def _require_nag_tau(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    nag_explicit: bool,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_nag_tau: object,
) -> None:
    """Refuse un NAG tau hors de 1 et 5.

    ``extra_settings.validate_inputs`` contrôle ``NAG_tau`` après
    ``NAG_scale``, dès que le conteneur ``NAG_col`` est visible, donc dès
    que ``NAG`` est vrai. Le type est un nombre. ``models/_settings.json``
    vaut 3,5. 5,01 est refusé. 5 passe. ``0,5`` est sous 1. Un objet
    ``model.NAG_tau`` peut relever le maximum (``{"max": 6}`` laisse
    passer 5,01) ou le baisser (``{"max": 4}`` refuse 5) et remplacer
    le libellé. ``validate_settings`` saute la tâche.
    ``process_tasks_cli`` sort 0, sans MP4. Si ``NAG_scale`` 21 est
    visible, cette erreur-là sort avant. Un objet illisible sur
    ``NAG_scale``, ``NAG_tau`` ou ``NAG_alpha`` lève ``AttributeError``
    dans ``get_def`` avant toute erreur de borne. ``NAG_alpha`` vient
    ensuite et n'est pas ce saut. Le frame scheduler qui lève n'atteint
    pas ``extra_settings``.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if not _nag_enabled(merged, nag_explicit, first_architecture):
        return
    if _nag_scale_spec(merged) is None or _nag_number_spec(
        merged,
        "NAG_alpha",
        _NAG_ALPHA_LABEL,
        _NAG_ALPHA_MIN,
        _NAG_ALPHA_MAX,
    ) is None:
        return
    spec = _nag_number_spec(
        merged,
        "NAG_tau",
        _NAG_TAU_LABEL,
        _NAG_TAU_MIN,
        _NAG_TAU_MAX,
    )
    if spec is None:
        return
    label, minimum, maximum = spec
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if queue_nag_tau is not _QUEUE_MODEL_MODE_UNSET and queue_nag_tau is not None:
        raw_tau: object = queue_nag_tau
        tau_note = "La file envoie NAG_tau."
    elif "NAG_tau" in settings:
        raw_tau = settings["NAG_tau"]
        tau_note = "Le JSON de réglages envoie NAG_tau."
    else:
        raw_tau = _FACTORY_NAG_TAU
        tau_note = (
            "update_default_settings de Wan ne remplace pas NAG_tau. "
            f"models/_settings.json le laisse à {_format_color_strength(_FACTORY_NAG_TAU)}."
        )
    shown = architecture_where or first_where or model_type
    if nag_explicit:
        flag_note = (
            "Le JSON écrit NAG. bool de WanGP : 1 et la chaîne false sont vrais, "
            "0 est faux."
        )
    else:
        flag_note = (
            f"query_model_def({first_architecture!r}) pose NAG à vrai "
            f"({first_where}). La formule est (vace_class ou test_class_t2v ou "
            "test_class_i2v) et pas (bernini ou animate2). t2v_1.3B n'est pas "
            "dans test_class_t2v. init_model_def n'est appelé que pour ce premier "
            "fichier. Un JSON suivant qui change l'architecture ne retire pas le drapeau."
        )
    if isinstance(merged.get("NAG_tau"), dict):
        flag_note += " Les bornes viennent de l'objet model.NAG_tau."
    else:
        flag_note += " Les bornes d'usine sont 1 et 5."
    if _color_strength_rejected(raw_tau):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. NAG est vrai et "
            f"NAG_tau vaut {raw_tau!r}, qui n'est pas un nombre. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({label} must be a number) après NAG_scale. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{tau_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    tau = _color_strength_number(raw_tau)
    if tau is None:
        return
    too_low = minimum is not None and tau < minimum
    too_high = maximum is not None and tau > maximum
    if not too_low and not too_high:
        return
    shown_tau = _format_color_strength(tau)
    if too_low and minimum is not None:
        shown_limit = _format_color_strength(minimum)
        limit = f"en dessous de {shown_limit}"
        wan_message = f"must be at least {shown_limit}"
    elif too_high and maximum is not None:
        shown_limit = _format_color_strength(maximum)
        limit = f"au-dessus de {shown_limit}"
        wan_message = f"must be at most {shown_limit}"
    else:
        return
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. NAG est vrai et "
        f"NAG_tau vaut {shown_tau}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({label} {wan_message}) après NAG_scale. "
        "Le défaut d'usine 3.5 passe, ainsi que 1 et 5. 5.01 non. 0.5 est sous 1. "
        "Un objet model.NAG_tau peut changer le libellé et les bornes : "
        "max 6 laisse passer 5.01, max 4 refuse 5. "
        "Si NAG_scale 21 est visible, cette erreur-là sort avant. "
        "NAG_alpha vient après : une valeur 3 n'est pas ce saut. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{tau_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser NAG_tau dans la borne, écrire NAG à faux, ou relever max dans "
        "l'objet model.NAG_tau."
    )


def _require_nag_alpha(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    nag_explicit: bool,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_nag_alpha: object,
) -> None:
    """Refuse un NAG alpha hors de 0 et 2.

    ``extra_settings.validate_inputs`` contrôle ``NAG_alpha`` après
    ``NAG_tau``, dès que le conteneur ``NAG_col`` est visible, donc dès
    que ``NAG`` est vrai. Le type est un nombre. ``models/_settings.json``
    vaut 0,5. 2,01 est refusé. 0 et 2 passent. ``-0,01`` est sous 0. Un
    objet ``model.NAG_alpha`` peut relever le maximum (``{"max": 3}``
    laisse passer 2,01) ou le baisser (``{"max": 1}`` refuse 2) et
    remplacer le libellé. ``validate_settings`` saute la tâche.
    ``process_tasks_cli`` sort 0, sans MP4. Si ``NAG_tau`` 5,01 est
    visible, cette erreur-là sort avant. Si ``NAG_scale`` 21 est visible,
    elle sort encore avant le tau. Un objet illisible sur ``NAG_scale``,
    ``NAG_tau`` ou ``NAG_alpha`` lève ``AttributeError`` dans ``get_def``
    avant toute erreur de borne. ``pace`` vient ensuite et n'est pas ce
    saut. Le frame scheduler qui lève n'atteint pas ``extra_settings``.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if not _nag_enabled(merged, nag_explicit, first_architecture):
        return
    if _nag_scale_spec(merged) is None or _nag_number_spec(
        merged,
        "NAG_tau",
        _NAG_TAU_LABEL,
        _NAG_TAU_MIN,
        _NAG_TAU_MAX,
    ) is None:
        return
    spec = _nag_number_spec(
        merged,
        "NAG_alpha",
        _NAG_ALPHA_LABEL,
        _NAG_ALPHA_MIN,
        _NAG_ALPHA_MAX,
    )
    if spec is None:
        return
    label, minimum, maximum = spec
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    if queue_nag_alpha is not _QUEUE_MODEL_MODE_UNSET and queue_nag_alpha is not None:
        raw_alpha: object = queue_nag_alpha
        alpha_note = "La file envoie NAG_alpha."
    elif "NAG_alpha" in settings:
        raw_alpha = settings["NAG_alpha"]
        alpha_note = "Le JSON de réglages envoie NAG_alpha."
    else:
        raw_alpha = _FACTORY_NAG_ALPHA
        alpha_note = (
            "update_default_settings de Wan ne remplace pas NAG_alpha. "
            f"models/_settings.json le laisse à {_format_color_strength(_FACTORY_NAG_ALPHA)}."
        )
    shown = architecture_where or first_where or model_type
    if nag_explicit:
        flag_note = (
            "Le JSON écrit NAG. bool de WanGP : 1 et la chaîne false sont vrais, "
            "0 est faux."
        )
    else:
        flag_note = (
            f"query_model_def({first_architecture!r}) pose NAG à vrai "
            f"({first_where}). La formule est (vace_class ou test_class_t2v ou "
            "test_class_i2v) et pas (bernini ou animate2). t2v_1.3B n'est pas "
            "dans test_class_t2v. init_model_def n'est appelé que pour ce premier "
            "fichier. Un JSON suivant qui change l'architecture ne retire pas le drapeau."
        )
    if isinstance(merged.get("NAG_alpha"), dict):
        flag_note += " Les bornes viennent de l'objet model.NAG_alpha."
    else:
        flag_note += " Les bornes d'usine sont 0 et 2."
    if _color_strength_rejected(raw_alpha):
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. NAG est vrai et "
            f"NAG_alpha vaut {raw_alpha!r}, qui n'est pas un nombre. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({label} must be a number) après NAG_tau. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{alpha_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    alpha = _color_strength_number(raw_alpha)
    if alpha is None:
        return
    too_low = minimum is not None and alpha < minimum
    too_high = maximum is not None and alpha > maximum
    if not too_low and not too_high:
        return
    shown_alpha = _format_color_strength(alpha)
    if too_low and minimum is not None:
        shown_limit = _format_color_strength(minimum)
        limit = f"en dessous de {shown_limit}"
        wan_message = f"must be at least {shown_limit}"
    elif too_high and maximum is not None:
        shown_limit = _format_color_strength(maximum)
        limit = f"au-dessus de {shown_limit}"
        wan_message = f"must be at most {shown_limit}"
    else:
        return
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. NAG est vrai et "
        f"NAG_alpha vaut {shown_alpha}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({label} {wan_message}) après NAG_tau. "
        "Le défaut d'usine 0.5 passe, ainsi que 0 et 2. 2.01 non. -0.01 est sous 0. "
        "Un objet model.NAG_alpha peut changer le libellé et les bornes : "
        "max 3 laisse passer 2.01, max 1 refuse 2. "
        "Si NAG_tau 5.01 est visible, cette erreur-là sort avant. "
        "Si NAG_scale 21 est visible, elle sort encore avant le tau. "
        "pace vient après : une valeur 9 n'est pas ce saut. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{alpha_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser NAG_alpha dans la borne, écrire NAG à faux, ou relever max dans "
        "l'objet model.NAG_alpha."
    )


_CUSTOM_DROPDOWN_MAX = 5
_PACE_ID = "pace"
_PACE_LABEL = "Pace"
_PACE_MIN = 0.2
_PACE_MAX = 1.0
_CUSTOM_SETTINGS_SCAN = 5


def _setting_kind(raw: object) -> str:
    """``_normalize_type`` de ``shared/extra_settings.py``."""
    kind = str(raw or "number").strip().lower()
    if kind in {"int", "integer"}:
        return "integer"
    if kind in {"float", "number"}:
        return "number"
    return "string"


def _pace_setting(merged: dict[str, object]) -> dict[str, object] | None:
    """Premier objet dont l'id est exactement ``pace``, dans les cinq premiers.

    ``Pace`` ne compte pas. Un élément qui n'est pas un dict est sauté.
    Une valeur qui n'est pas une liste, et le sixième objet, sont ignorés.
    """
    raw = merged.get("custom_settings")
    if not isinstance(raw, list):
        return None
    for item in raw[:_CUSTOM_SETTINGS_SCAN]:
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "") or "").strip() == _PACE_ID:
            return item
    return None


def _pace_label(setting: dict[str, object]) -> str:
    """Libellé de ``pace`` après ``_trim_label``.

    ``label`` gagne sur ``name``. Une chaîne vide retombe sur ``Pace``.
    Un libellé qui ne laisse que ``(`` retombe sur la clé ``pace``.
    """
    if "label" in setting:
        raw = setting["label"]
    else:
        raw = setting.get("name", _PACE_LABEL)
    return _trim_setting_label(str(raw or _PACE_LABEL), _PACE_ID)


def _pace_bound(setting: dict[str, object], bound: str, fallback: float) -> float | None:
    """Borne de l'objet, ou le défaut. Une borne illisible disparaît."""
    raw = setting.get(bound, fallback)
    return _normalize_setting_number(raw)


def _iter_defs_breaks_before_pace(merged: dict[str, object]) -> bool:
    """``iter_defs`` lève avant de renvoyer l'erreur de ``pace``.

    ``get_def`` résout chaque définition, visible ou non. Un objet qui
    n'est pas un dict sur ``switch_threshold``, ``temperature_slider``,
    ``NAG_scale``, ``NAG_tau`` ou ``NAG_alpha``, ou un ``label`` NAG qui
    n'est pas du texte, lève ``AttributeError``. Le processus ne sort
    pas 0, même si ``NAG`` est faux.
    """
    for key in ("switch_threshold", "temperature_slider"):
        if key in merged and not isinstance(merged[key], dict):
            return True
    for key, label, low, high in (
        ("NAG_scale", _NAG_SCALE_LABEL, _NAG_SCALE_MIN, _NAG_SCALE_MAX),
        ("NAG_tau", _NAG_TAU_LABEL, _NAG_TAU_MIN, _NAG_TAU_MAX),
        ("NAG_alpha", _NAG_ALPHA_LABEL, _NAG_ALPHA_MIN, _NAG_ALPHA_MAX),
    ):
        if key in merged and _nag_number_spec(merged, key, label, low, high) is None:
            return True
    return False


_COLLECT_PARSE_TYPES = frozenset({"int", "float", "text", "dropdown"})


def _collect_parse_type(setting: dict[str, object]) -> str:
    """Type vu par ``get_model_custom_settings``, pas par extra_settings.

    ``CUSTOM_SETTING_TYPES`` vaut ``int``, ``float``, ``text`` et
    ``dropdown``. ``integer``, ``number`` et une clé absente deviennent
    ``text``. ``int`` reste ``int``.
    """
    raw = setting["type"] if "type" in setting else "text"
    parsed = str(raw or "text").strip().lower()
    if parsed in _COLLECT_PARSE_TYPES:
        return parsed
    return "text"


def _collect_slot_index(merged: dict[str, object], setting: dict[str, object]) -> int:
    """Index d'énumération, y compris les éléments qui ne sont pas des dict."""
    raw = merged.get("custom_settings")
    if isinstance(raw, list):
        for index, item in enumerate(raw[:_CUSTOM_SETTINGS_SCAN]):
            if item is setting:
                return index
    return 0


def _collect_untrimmed_label(setting: dict[str, object], index: int) -> str:
    """Libellé de la copie utilisée par le parse, sans ``_trim_label``."""
    if "label" in setting:
        return str(setting["label"])
    return f"Custom Setting {index + 1}"


def _parse_collected_int(raw: object) -> tuple[object, str | None]:
    """``parse_custom_setting_typed_value`` pour le type ``int``.

    Un flottant entier devient un ``int``. Un flottant 1,5 renvoie
    « Expected an integer value. ». ``None`` si la valeur est vide.
    """
    if raw is None:
        return None, None
    if isinstance(raw, str):
        raw = raw.strip()
        if len(raw) == 0:
            return None, None
    if isinstance(raw, bool):
        return None, "Expected an integer value."
    if isinstance(raw, int):
        return raw, None
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw), None
        return None, "Expected an integer value."
    try:
        return int(str(raw).strip()), None
    except (TypeError, ValueError):
        try:
            parsed = float(str(raw).strip())
        except (TypeError, ValueError):
            return None, "Expected an integer value."
        if not parsed.is_integer():
            return None, "Expected an integer value."
        return int(parsed), None


def _rewrite_custom_value(
    raw: object,
    setting: dict[str, object],
    merged: dict[str, object],
    kind: str,
) -> tuple[object, str | None, str | None]:
    """Valeur vue après ``collect_custom_settings_from_inputs``.

    Le troisième élément est l'erreur de parse du type ``int``. Un
    flottant non entier de type ``integer`` (collecté comme ``text``)
    est remplacé par sa chaîne : extra_settings ne le tronque plus.
    """
    collect_type = _collect_parse_type(setting)
    if collect_type == "int":
        parsed, error = _parse_collected_int(raw)
        if error is not None:
            label = _collect_untrimmed_label(setting, _collect_slot_index(merged, setting))
            return raw, label, error
        if parsed is None:
            return raw, None, None
        return parsed, None, None
    if (
        collect_type == "text"
        and kind == "integer"
        and isinstance(raw, float)
        and not raw.is_integer()
    ):
        return str(raw), None, None
    return raw, None, None


def _raise_collected_int(
    setting_name: str,
    raw: object,
    collect_label: str,
    parse_error: str,
    after: str,
    note: str,
    flag_note: str,
    architecture: str,
    shown: str,
) -> Never:
    """Le type ``int`` saute la tâche avant ``extra_settings``."""
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. "
        f"custom_settings contient l'id {setting_name} de type int et "
        f"{setting_name} vaut {raw!r}. "
        "collect_custom_settings_from_inputs (wgp.py, strict=True) appelle "
        "parse_custom_setting_typed_value avant extra_settings.validate_inputs. "
        "CUSTOM_SETTING_TYPES garde int. "
        f"Le message est « {collect_label} {parse_error} ». "
        "Le libellé de la copie n'est pas tronqué au premier « ( ». "
        "Sans label, il vaut Custom Setting et l'index d'énumération à partir de 1, "
        "y compris les éléments qui ne sont pas des dict. "
        "Un flottant entier est converti, puis les bornes d'extra_settings s'appliquent. "
        "Un entier hors borne reste l'erreur de borne, pas ce parse. "
        f"validate_settings renvoie cette erreur {after}. "
        "process_tasks_cli compte la tâche comme sautée et quitte avec le code 0, sans MP4. "
        f"{note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}."
    )


def _integer_setting_value(value: object) -> int | str:
    """Entier de ``validate_setting_value``, ou ``an integer`` si le type est refusé.

    ``int`` tronque un flottant entier : 2,0 devient 2. Une chaîne n'est
    un entier que si ``float`` tombe juste. Un flottant 1,5 de type
    ``integer`` a déjà été réécrit en chaîne par le pré-parse. Le booléen
    est traité avant cet appel.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return "an integer"
        if not parsed.is_integer():
            return "an integer"
        return int(parsed)


def _pace_values(
    settings: dict[str, object],
    queue_custom_settings: object,
) -> tuple[object, str]:
    """Valeur de ``pace`` et sa provenance.

    Le second élément vaut ``absent`` quand la clé n'est pas un nombre à
    contrôler. La file remplace tout le dict. Un dict absent de la file
    laisse le JSON. ``None`` dans la file ne remplace pas. Une valeur qui
    n'est pas un dict est lue comme un dict vide.
    """
    if queue_custom_settings is not _QUEUE_MODEL_MODE_UNSET and queue_custom_settings is not None:
        bag: object = queue_custom_settings
        note = "La file envoie custom_settings."
    elif "custom_settings" in settings:
        bag = settings["custom_settings"]
        note = "Le JSON de réglages envoie custom_settings."
    else:
        return _QUEUE_MODEL_MODE_UNSET, "Pas de dict custom_settings : la valeur est absente."
    if not isinstance(bag, dict) or "pace" not in bag:
        return _QUEUE_MODEL_MODE_UNSET, (
            f"{note} La clé pace est absente, ou custom_settings n'est pas un dict."
        )
    return bag["pace"], note


def _normalize_custom_setting_name(name: object) -> str:
    """``_normalize_custom_setting_name`` de ``wgp.py``."""
    return re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")


def _assign_custom_setting_id(
    setting: dict[str, object],
    index: int,
    used: set[str],
) -> str:
    """Id lu par ``collect_custom_settings_from_inputs``.

    L'id explicite est normalisé. Sinon ``name``, puis ``param``. Un nom
    absent devient ``Custom Setting N``, donc ``custom_setting_N``. Une
    collision reprend ``custom_setting_{index+1}`` sans second essai.
    """
    explicit = setting.get("id", None)
    setting_id = ""
    if explicit is not None and len(str(explicit).strip()) > 0:
        setting_id = _normalize_custom_setting_name(explicit)
    if not setting_id:
        for field in ("name", "param"):
            if field not in setting:
                continue
            setting_id = _normalize_custom_setting_name(setting.get(field, ""))
            if setting_id:
                break
    if not setting_id:
        setting_id = _normalize_custom_setting_name(f"Custom Setting {index + 1}")
    if not setting_id or setting_id in used:
        setting_id = f"custom_setting_{index + 1}"
    used.add(setting_id)
    return setting_id


def _dropdown_choice_pairs(setting: dict[str, object]) -> list[tuple[str, object]] | None:
    """``get_custom_setting_dropdown_choices`` une fois le type normalisé.

    Une liste vide, ou une valeur qui n'est pas une liste, vaut ``None`` :
    WanGP réécrit alors le type en ``text``.
    """
    choices = setting.get("choices", [])
    if not isinstance(choices, list):
        return None
    normalized: list[tuple[str, object]] = []
    for choice in choices:
        if isinstance(choice, (list, tuple)) and len(choice) >= 2:
            normalized.append((str(choice[0]), choice[1]))
        else:
            normalized.append((str(choice), choice))
    if not normalized:
        return None
    return normalized


def _dropdown_value_rejected(raw: object, choices: list[tuple[str, object]]) -> bool:
    """``parse_custom_setting_typed_value`` pour un type encore ``dropdown``.

    ``None`` et une chaîne vide reviennent avant les choix. Une valeur
    égale, ou égale après ``str``, est acceptée. La comparaison porte
    sur la valeur du couple, pas sur son libellé.
    """
    if raw is None:
        return False
    if isinstance(raw, str):
        raw = raw.strip()
        if len(raw) == 0:
            return False
    for _choice_label, value in choices:
        if raw == value or str(raw) == str(value):
            return False
    return True


def _custom_mode_visible(setting: dict[str, object], model_mode: object) -> bool:
    """``custom_setting_mode_visible``. Lève ``TypeError`` si l'appartenance lève."""
    for key, excluded in (("model_mode", False), ("model_mode_not", True)):
        if key not in setting:
            continue
        modes = setting[key]
        if not isinstance(modes, (list, tuple)):
            modes = [modes]
        if (model_mode in modes) == excluded:
            return False
    return True


def _custom_dropdown_visible(
    setting: dict[str, object],
    video: str,
    audio: str,
    model_mode: object,
) -> bool | None:
    """``custom_setting_visible`` au moment du parse strict.

    ``None`` si ``model_mode in modes`` lève : le processus ne sort pas 0.
    """
    try:
        if not _custom_mode_visible(setting, model_mode):
            return False
    except TypeError:
        return None
    return _custom_guide_visible(setting, video, audio)


def _custom_settings_bag(
    settings: dict[str, object],
    queue_custom_settings: object,
) -> tuple[dict[object, object] | None, str]:
    """Dict ``custom_settings`` lu par ``collect_custom_settings_from_inputs``.

    La file remplace tout le dict. ``None`` dans la file ne remplace pas.
    Une valeur qui n'est pas un dict est lue comme un dict vide.
    """
    if queue_custom_settings is not _QUEUE_MODEL_MODE_UNSET and queue_custom_settings is not None:
        bag: object = queue_custom_settings
        note = "La file envoie custom_settings."
    elif "custom_settings" in settings:
        bag = settings["custom_settings"]
        note = "Le JSON de réglages envoie custom_settings."
    else:
        return None, "Pas de dict custom_settings : la valeur est absente."
    if not isinstance(bag, dict):
        return None, f"{note} custom_settings n'est pas un dict."
    return bag, note


def _raise_dropdown_choice(
    setting_id: str,
    raw: object,
    collect_label: str,
    note: str,
    architecture: str,
    shown: str,
) -> Never:
    """Le menu saute la tâche avant ``extra_settings``."""
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. "
        f"custom_settings contient l'id {setting_id} de type dropdown et "
        f"la valeur vaut {raw!r}, hors de choices. "
        "collect_custom_settings_from_inputs (wgp.py, strict=True) appelle "
        "parse_custom_setting_typed_value avant extra_settings.validate_inputs. "
        "CUSTOM_SETTING_TYPES garde dropdown. "
        f"Le message est « {collect_label} Expected one of the dropdown choices. ». "
        "Le libellé de la copie n'est pas tronqué au premier « ( ». "
        "Sans label, il vaut Custom Setting et l'index d'énumération à partir de 1, "
        "y compris les éléments qui ne sont pas des dict. "
        "_setting_kind(dropdown) vaut string : extra_settings ignore la valeur. "
        "Le saut vient seulement de ce parse. "
        "Une valeur égale, ou égale après str, passe. None et une chaîne vide aussi. "
        "Une liste de choix vide, ou une valeur qui n'est pas une liste, devient text "
        "et ne saute pas. Le sixième objet est ignoré. "
        "CUSTOM_SETTING_DROPDOWN_MAX vaut 5, comme le scan : un sixième menu "
        "n'est pas lu. "
        "validate_settings renvoie cette erreur. "
        "process_tasks_cli compte la tâche comme sautée et quitte avec le code 0, sans MP4. "
        f"{note} "
        f"Architecture finale {architecture}, lue dans {shown}."
    )


def _require_custom_dropdowns(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_model: dict[str, object],
    first_where: str,
    architecture_where: str,
    custom_settings_where: str,
    image_ref_explicit: bool,
    guide_custom_explicit: bool,
    kiwi_ref_explicit: bool,
    sliding_window_explicit: bool,
    queue_model_mode: object,
    queue_video_prompt_type: object,
    queue_audio_prompt_type: object,
    queue_discard_last_frames: object,
    queue_custom_settings: object,
) -> None:
    """Refuse une valeur absente des choix d'un menu custom.

    ``collect_custom_settings_from_inputs`` tourne avec ``strict=True``
    avant ``extra_settings``. Un type ``dropdown`` dont ``choices`` n'est
    pas vide renvoie « Expected one of the dropdown choices. ». La tâche
    est sautée, ``process_tasks_cli`` sort 0, sans MP4.
    ``_setting_kind("dropdown")`` vaut ``string`` : les bornes de ``pace``
    et d'``exaggeration`` ne voient pas cette valeur. Une liste vide, ou
    une valeur qui n'est pas une liste, fait repasser le type en ``text``.
    ``None`` et une chaîne vide passent. Une valeur égale, ou égale après
    ``str``, aussi. Le sixième objet dépasse ``CUSTOM_SETTINGS_MAX``.
    Un réglage masqué par ``video_prompt_type``, ``audio_prompt_type`` ou
    ``model_mode`` n'est pas parsé. Si le frame scheduler lève, ce parse
    n'est pas atteint.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    letters = _resolved_prompt_letters(
        merged,
        settings,
        first_architecture,
        first_model,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_video_prompt_type,
        queue_audio_prompt_type,
    )
    if letters is None:
        return
    video, audio, _before_filter, _ref = letters
    model_mode = _effective_model_mode(settings, queue_model_mode)
    raw_list = merged.get("custom_settings")
    if not isinstance(raw_list, list):
        return
    bag, note = _custom_settings_bag(settings, queue_custom_settings)
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    shown = custom_settings_where or architecture_where or first_where or model_type
    used: set[str] = set()
    dropdown_count = 0
    for index, item in enumerate(raw_list[:_CUSTOM_SETTINGS_SCAN]):
        if not isinstance(item, dict):
            continue
        setting_id = _assign_custom_setting_id(item, index, used)
        collect_type = _collect_parse_type(item)
        kept_dropdown = False
        if collect_type == "dropdown":
            dropdown_count += 1
            pairs = _dropdown_choice_pairs(item)
            kept_dropdown = (
                dropdown_count <= _CUSTOM_DROPDOWN_MAX and pairs is not None
            )
        else:
            pairs = None
        visible = _custom_dropdown_visible(item, video, audio, model_mode)
        if visible is None:
            return
        if not visible:
            continue
        raw: object = None
        if bag is not None and setting_id in bag:
            raw = bag[setting_id]
        if collect_type == "int":
            _parsed, error = _parse_collected_int(raw)
            if error is not None and setting_id in {_PACE_ID, _EXAGGERATION_ID}:
                return
            continue
        if not kept_dropdown or pairs is None:
            continue
        if not _dropdown_value_rejected(raw, pairs):
            continue
        if "label" in item:
            collect_label = str(item["label"])
        else:
            collect_label = f"Custom Setting {index + 1}"
        _raise_dropdown_choice(
            setting_id,
            raw,
            collect_label,
            note,
            architecture,
            shown,
        )


def _injected_custom_sigmas(architecture: str | None) -> bool:
    """``custom_sigmas`` posé par ``wan_handler.query_model_def``.

    Le bloc ``extra_model_def.update`` met la clé à vrai pour chaque
    architecture de ce handler, ``t2v_1.3B`` compris. ``init_model_def``
    n'est appelé que pour le premier JSON. Un JSON qui écrit la clé
    remplace l'injection. Un finetune qui ne fait que changer
    l'architecture la laisse.
    """
    return architecture in _WAN_ARCHITECTURES


def _custom_sigmas_enabled(
    merged: dict[str, object],
    explicit: bool,
    first_architecture: str | None,
) -> bool:
    """Drapeau vu par ``if model_def.get("custom_sigmas", False)``.

    ``bool`` suit WanGP : ``1`` et la chaîne ``false`` sont vrais, ``0``
    et une chaîne vide sont faux. ``None`` aussi.
    """
    if explicit:
        return bool(merged.get("custom_sigmas"))
    return _injected_custom_sigmas(first_architecture)


def _sigma_phase_count(raw: object) -> tuple[int | None, str | None]:
    """Phases après ``normalize_phase_2_tiling_selection``, puis ``int``.

    ``2~`` devient 2 avant ``int``. Le second élément est le texte de
    ``ValueError`` : ``validate_settings`` le renvoie et le CLI sort 0.
    Une chaîne vide signale un ``TypeError`` : le processus ne sort pas 0.
    ``None`` comme second élément veut dire que le compte est utilisable.
    """
    phases = 2 if raw == "2~" else raw
    try:
        count = int(phases)  # type: ignore[arg-type]
    except ValueError as exc:
        return None, str(exc)
    except TypeError:
        return None, ""
    return max(1, count), None


def _parse_custom_sigmas_error(value: object, phases: int) -> str | None:
    """Message de ``parse_custom_sigmas``, ou None si le calendrier passe.

    Une chaîne vide ou des espaces vaut ``[None] * phases``. Le 0 final
    est retiré : ``0.9,0`` reste une valeur. ``0`` seul, après ce retrait,
    n'est plus un sigma positif.
    """
    if not isinstance(value, str):
        return "Custom Sigmas must be text: comma-separated numbers, with ';' between phases."
    if not value.strip():
        return None
    slots = value.split(";")
    if len(slots) != phases:
        return (
            f"Custom Sigmas requires {phases} phase slots separated by "
            f"{phases - 1} ';' characters. "
            "Leave a slot blank for its default schedule."
        )
    for phase, slot in enumerate(slots, 1):
        if not slot.strip():
            continue
        try:
            sigmas = [float(token.strip()) for token in slot.split(",")]
        except ValueError:
            return f"Custom Sigmas phase {phase}: enter numbers separated by commas."
        if any(not math.isfinite(sigma) or not 0 <= sigma < 1 for sigma in sigmas):
            return (
                f"Custom Sigmas phase {phase}: values must be finite numbers "
                "from 0 (inclusive) to 1 (exclusive)."
            )
        if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
            return f"Custom Sigmas phase {phase}: values must be in strictly descending order."
        if sigmas[-1] == 0:
            sigmas.pop()
        if not sigmas:
            return (
                f"Custom Sigmas phase {phase}: provide at least one positive sigma, "
                "or leave the slot blank."
            )
    return None


def _float_parse_rejected(raw: object) -> bool:
    """``parse_custom_setting_typed_value`` pour le type ``float``."""
    if raw is None:
        return False
    if isinstance(raw, str):
        raw = raw.strip()
        if len(raw) == 0:
            return False
    if isinstance(raw, bool):
        return True
    try:
        float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return True
    return False


def _collect_parse_stops_before_sigmas(
    merged: dict[str, object],
    settings: dict[str, object],
    first_architecture: str | None,
    first_model: dict[str, object],
    image_ref_explicit: bool,
    guide_custom_explicit: bool,
    kiwi_ref_explicit: bool,
    queue_model_mode: object,
    queue_video_prompt_type: object,
    queue_audio_prompt_type: object,
    queue_custom_settings: object,
) -> bool:
    """Le parse strict des réglages custom échoue avant ``parse_custom_sigmas``.

    ``True`` aussi si la visibilité lève ``TypeError`` ou si les lettres
    de prompt ne sont pas du texte : ce n'est pas un code 0 de sigmas.
    """
    letters = _resolved_prompt_letters(
        merged,
        settings,
        first_architecture,
        first_model,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_video_prompt_type,
        queue_audio_prompt_type,
    )
    if letters is None:
        return True
    video, audio, _before_filter, _ref = letters
    model_mode = _effective_model_mode(settings, queue_model_mode)
    raw_list = merged.get("custom_settings")
    if not isinstance(raw_list, list):
        return False
    bag, _note = _custom_settings_bag(settings, queue_custom_settings)
    used: set[str] = set()
    dropdown_count = 0
    for index, item in enumerate(raw_list[:_CUSTOM_SETTINGS_SCAN]):
        if not isinstance(item, dict):
            continue
        setting_id = _assign_custom_setting_id(item, index, used)
        collect_type = _collect_parse_type(item)
        kept_dropdown = False
        pairs: list[tuple[str, object]] | None = None
        if collect_type == "dropdown":
            dropdown_count += 1
            pairs = _dropdown_choice_pairs(item)
            kept_dropdown = dropdown_count <= _CUSTOM_DROPDOWN_MAX and pairs is not None
        visible = _custom_dropdown_visible(item, video, audio, model_mode)
        if visible is None:
            return True
        if not visible:
            continue
        raw: object = None
        if bag is not None and setting_id in bag:
            raw = bag[setting_id]
        if collect_type == "int":
            _parsed, error = _parse_collected_int(raw)
            if error is not None:
                return True
            continue
        if collect_type == "float":
            if _float_parse_rejected(raw):
                return True
            continue
        if kept_dropdown and pairs is not None and _dropdown_value_rejected(raw, pairs):
            return True
    return False


_WAN_SIGMA_SCHEDULERS: tuple[str, ...] = ("", "unipc", "euler", "dpm++")


def _factory_sample_solver(architecture: str) -> str:
    """``sample_solver`` après ``update_default_settings`` de Wan.

    Le bloc commun pose ``unipc``. ``multitalk`` et ``infinitetalk``
    le remplacent par ``euler``, ``animate2`` par ``dpm++``. Hors Wan,
    ``models/_settings.json`` laisse la chaîne vide.
    """
    if architecture in {"multitalk", "infinitetalk"}:
        return "euler"
    if architecture == "animate2":
        return "dpm++"
    if architecture in _WAN_ARCHITECTURES:
        return "unipc"
    return ""


def _resolved_sample_solver(
    queue: object,
    settings: dict[str, object],
    architecture: str,
) -> object:
    """Solveur vu par ``validate_settings`` après ``fix_settings``.

    La file, puis le JSON, puis l'usine. Une valeur nulle de la file
    ne remplace pas une clé déjà présente. ``wan_handler.fix_settings``
    remet une chaîne vide à ``unipc``. ``None`` n'est pas cette chaîne.
    """
    raw = _queue_or_settings(queue, settings, "sample_solver", _factory_sample_solver(architecture))
    if architecture in _WAN_ARCHITECTURES and raw == "":
        return "unipc"
    return raw


def _sigma_scheduler_allow_list(
    merged: dict[str, object],
    explicit: bool,
    first_architecture: str | None,
) -> object:
    """Liste de ``custom_sigmas_schedulers``, ou None si le contrôle saute.

    ``query_model_def`` ne tourne que pour le premier JSON. Un JSON qui
    écrit la clé, y compris ``null``, remplace l'injection.
    """
    if explicit:
        return merged.get("custom_sigmas_schedulers")
    if _injected_custom_sigmas(first_architecture):
        return list(_WAN_SIGMA_SCHEDULERS)
    return None


def _explicit_sigma_present(value: object, phases: int) -> bool:
    """Au moins un créneau de ``parse_custom_sigmas`` n'est pas None.

    Appelé seulement après un parse réussi. Le 0 final est retiré :
    ``0.9,0`` reste un sigma. Une chaîne vide laisse ``[None]``.
    """
    if not isinstance(value, str) or not value.strip():
        return False
    slots = value.split(";")
    if len(slots) != phases:
        return False
    for slot in slots:
        if not slot.strip():
            continue
        sigmas = [float(token.strip()) for token in slot.split(",")]
        if sigmas and sigmas[-1] == 0:
            sigmas.pop()
        if sigmas:
            return True
    return False


def _incompatible_scheduler_message(solver: object, allow: object) -> str | None:
    """Message de la ligne 1222, ou None si WanGP ne saute pas en code 0.

    ``None`` pour la liste saute le contrôle. Un objet non itérable lève
    ``TypeError`` : le processus ne sort pas 0. Un élément vrai qui n'est
    pas du texte fait pareil dans ``join``.
    """
    if allow is None:
        return None
    try:
        rejected = solver not in allow  # type: ignore[operator]
    except TypeError:
        return None
    if not rejected:
        return None
    try:
        names = ", ".join(item or "Default" for item in allow)  # type: ignore[union-attr]
    except TypeError:
        return None
    return (
        "The selected scheduler is not compatible with Custom Sigmas. "
        "Clear Custom Sigmas or select a compatible scheduler: "
        f"{names}."
    )


def _raise_incompatible_scheduler(
    raw_sigmas: object,
    solver: object,
    wan_message: str,
    solver_note: str,
    list_note: str,
    architecture: str,
    shown: str,
) -> None:
    """Le solveur n'est pas dans ``custom_sigmas_schedulers``."""
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. custom_sigmas est vrai, "
        f"le calendrier vaut {raw_sigmas!r} et sample_solver vaut {solver!r}. "
        "validate_settings (wgp.py, ligne 1221) refuse un solveur absent de "
        "custom_sigmas_schedulers après parse_custom_sigmas et avant "
        "le contrôle continu des phases, puis avant extra_settings. "
        f"Le message est « {wan_message} ». "
        "La liste injectée est '', unipc, euler, dpm++. "
        "lcm et causvid avec 0.9,0.2 sautent. "
        "Une chaîne de sigmas vide ou des espaces laisse tous les créneaux "
        "à None : le solveur n'est pas contrôlé. "
        "Une clé absente hors Wan saute aussi ce contrôle. "
        "Le contrôle continu (custom_sigmas_continuous) vient après. "
        "process_tasks_cli compte la tâche comme sautée et quitte avec le "
        "code 0, sans MP4. "
        f"{solver_note} {list_note} "
        f"Architecture finale {architecture}, lue dans {shown}."
    )


def _require_sigma_scheduler(
    merged: dict[str, object],
    settings: dict[str, object],
    raw_sigmas: object,
    phases: int,
    architecture: str,
    shown: str,
    first_architecture: str | None,
    queue_sample_solver: object,
    schedulers_explicit: bool,
) -> None:
    """Refuse ``sample_solver`` hors de la liste compatible."""
    if not _explicit_sigma_present(raw_sigmas, phases):
        return
    allow = _sigma_scheduler_allow_list(merged, schedulers_explicit, first_architecture)
    solver = _resolved_sample_solver(queue_sample_solver, settings, architecture)
    message = _incompatible_scheduler_message(solver, allow)
    if message is None:
        return
    if queue_sample_solver is not _QUEUE_MODEL_MODE_UNSET and queue_sample_solver is not None:
        solver_note = "La file envoie sample_solver."
    elif queue_sample_solver is None and "sample_solver" in settings:
        solver_note = (
            "La file envoie sample_solver nul : clean_settings ne remplace pas "
            "la valeur du JSON."
        )
    elif "sample_solver" in settings:
        solver_note = (
            "Le JSON de réglages envoie sample_solver. "
            "fix_settings de wan_handler remet une chaîne vide à unipc. "
            "None n'est pas remplacé."
        )
    else:
        solver_note = (
            "models/_settings.json laisse sample_solver à une chaîne vide. "
            f"update_default_settings de {architecture} pose "
            f"{_factory_sample_solver(architecture)!r}, puis fix_settings "
            "remet une chaîne vide à unipc."
        )
    if schedulers_explicit:
        list_note = (
            "Le JSON écrit custom_sigmas_schedulers. None saute le contrôle. "
            "Une liste vide refuse tout solveur. Un objet non itérable lève "
            "TypeError : le processus ne sort pas 0."
        )
    elif _injected_custom_sigmas(first_architecture):
        list_note = (
            "La clé est absente du JSON. query_model_def de wan_handler pose "
            f"custom_sigmas_schedulers pour {first_architecture}. "
            "Un finetune qui change seulement l'architecture laisse la liste."
        )
    else:
        list_note = "Le premier JSON n'injecte pas custom_sigmas_schedulers."
    _raise_incompatible_scheduler(
        raw_sigmas,
        solver,
        message,
        solver_note,
        list_note,
        architecture,
        shown,
    )


def _raise_custom_sigmas(
    raw: object,
    phases: object,
    wan_message: str,
    value_note: str,
    phase_note: str,
    flag_note: str,
    architecture: str,
    shown: str,
) -> None:
    """``parse_custom_sigmas`` saute la tâche avant ``extra_settings``."""
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. custom_sigmas est vrai et "
        f"la valeur vaut {raw!r} pour guidance_phases {phases!r}. "
        "parse_custom_sigmas (wgp.py, ligne 1090) lève ValueError après "
        "collect_custom_settings_from_inputs et avant extra_settings.validate_inputs. "
        f"Le message est « {wan_message} ». "
        "models/_settings.json laisse la chaîne vide et guidance_phases à 1 : "
        "une chaîne vide ou des espaces passe. 0.9,0.2 et 0.9,0 (le 0 final "
        "est retiré) passent aussi. Un menu hors liste ou un type int sort avant. "
        "NAG, pace et exaggeration sortent après. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{value_note} {phase_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}."
    )


def _require_custom_sigmas(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_architecture: str | None,
    first_model: dict[str, object],
    first_where: str,
    architecture_where: str,
    custom_sigmas_explicit: bool,
    image_ref_explicit: bool,
    guide_custom_explicit: bool,
    kiwi_ref_explicit: bool,
    sliding_window_explicit: bool,
    queue_model_mode: object,
    queue_video_prompt_type: object,
    queue_audio_prompt_type: object,
    queue_discard_last_frames: object,
    queue_custom_settings: object,
    queue_custom_sigmas: object,
    queue_guidance_phases: object,
    queue_sample_solver: object,
    custom_sigmas_schedulers_explicit: bool,
) -> None:
    """Refuse un calendrier de sigmas que ``parse_custom_sigmas`` rejette.

    Le contrôle n'a lieu que si ``model.custom_sigmas`` est vrai. Wan
    l'injecte pour ``t2v_1.3B``. La chaîne vient de la file, puis du
    JSON, puis de ``models/_settings.json`` (vide). ``guidance_phases``
    suit le même ordre, usine 1. ``2~`` compte pour 2. La tâche est
    sautée et le CLI sort 0, sans MP4. Un calendrier parsé qui a au
    moins un créneau défini est ensuite comparé à
    ``custom_sigmas_schedulers``. Wan injecte
    ``["", "unipc", "euler", "dpm++"]``. ``lcm`` et ``causvid``
    sautent. L'usine de ``sample_solver`` est ``unipc``, ``euler``
    pour ``multitalk`` et ``infinitetalk``, ``dpm++`` pour
    ``animate2``. ``fix_settings`` remet une chaîne vide à ``unipc``.
    Une chaîne de sigmas vide ignore le solveur.
    """
    if not _custom_sigmas_enabled(merged, custom_sigmas_explicit, first_architecture):
        return
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if _collect_parse_stops_before_sigmas(
        merged,
        settings,
        first_architecture,
        first_model,
        image_ref_explicit,
        guide_custom_explicit,
        kiwi_ref_explicit,
        queue_model_mode,
        queue_video_prompt_type,
        queue_audio_prompt_type,
        queue_custom_settings,
    ):
        return
    raw_sigmas = _queue_or_settings(queue_custom_sigmas, settings, "custom_sigmas", "")
    raw_phases = _queue_or_settings(queue_guidance_phases, settings, "guidance_phases", 1)
    if queue_custom_sigmas is not _QUEUE_MODEL_MODE_UNSET and queue_custom_sigmas is not None:
        value_note = "La file envoie custom_sigmas."
    elif "custom_sigmas" in settings:
        value_note = "Le JSON de réglages envoie custom_sigmas."
    else:
        value_note = "models/_settings.json laisse custom_sigmas à une chaîne vide."
    if queue_guidance_phases is not _QUEUE_MODEL_MODE_UNSET and queue_guidance_phases is not None:
        phase_note = "La file envoie guidance_phases."
    elif "guidance_phases" in settings:
        phase_note = "Le JSON de réglages envoie guidance_phases."
    else:
        phase_note = "models/_settings.json laisse guidance_phases à 1."
    if custom_sigmas_explicit:
        flag_note = (
            "Le JSON écrit custom_sigmas. bool de WanGP : 1 et la chaîne false "
            "sont vrais, 0 et une chaîne vide sont faux."
        )
    else:
        flag_note = (
            "La clé est absente du JSON. query_model_def de wan_handler pose "
            f"custom_sigmas à vrai pour {first_architecture}. "
            "t2v_1.3B est dans ce handler. "
            "Un finetune qui change seulement l'architecture laisse le drapeau."
        )
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    shown = architecture_where or first_where or model_type
    count, phase_error = _sigma_phase_count(raw_phases)
    if phase_error == "":
        return
    if phase_error is not None:
        _raise_custom_sigmas(
            raw_sigmas,
            raw_phases,
            phase_error,
            value_note,
            phase_note,
            flag_note,
            architecture,
            shown,
        )
        return
    if count is None:
        return
    sigma_error = _parse_custom_sigmas_error(raw_sigmas, count)
    if sigma_error is None:
        _require_sigma_scheduler(
            merged,
            settings,
            raw_sigmas,
            count,
            architecture,
            shown,
            first_architecture,
            queue_sample_solver,
            custom_sigmas_schedulers_explicit,
        )
        return
    _raise_custom_sigmas(
        raw_sigmas,
        raw_phases,
        sigma_error,
        value_note,
        phase_note,
        flag_note,
        architecture,
        shown,
    )


def _require_pace(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    custom_settings_where: str,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_custom_settings: object,
) -> None:
    """Refuse un pace hors de 0,2 et 1.

    ``extra_settings.validate_inputs`` contrôle ``pace`` après
    ``NAG_alpha``. Le réglage est ``custom``. Il n'est visible que si
    ``model.custom_settings`` est une liste et qu'un des cinq premiers
    objets a l'id ``pace``. La valeur vient de ``inputs["custom_settings"]["pace"]``.
    Une clé ``pace`` à la racine est ignorée. Il n'y a pas de défaut
    dans ``models/_settings.json`` : une valeur absente passe. 9, ``9``
    et 1,01 sont refusés. 0,19 est sous 0,2. ``validate_settings`` saute
    la tâche. ``process_tasks_cli`` sort 0, sans MP4. Si ``NAG_alpha``
    2,01 est visible, cette erreur-là sort avant. ``exaggeration`` vient
    ensuite et n'est pas ce saut. Un type ``integer`` et un flottant 1,5
    sont réécrits en chaîne avant ce contrôle. Un type ``int`` et le
    même flottant sont refusés avant les bornes.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if _iter_defs_breaks_before_pace(merged):
        return
    setting = _pace_setting(merged)
    if setting is None:
        return
    kind = _setting_kind(setting.get("type", "number"))
    if kind == "string":
        return
    label = _pace_label(setting)
    minimum = _pace_bound(setting, "min", _PACE_MIN)
    maximum = _pace_bound(setting, "max", _PACE_MAX)
    raw_pace, pace_note = _pace_values(settings, queue_custom_settings)
    if raw_pace is _QUEUE_MODEL_MODE_UNSET:
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    shown = custom_settings_where or architecture_where or first_where or model_type
    flag_note = (
        "L'id pace est lu tel quel, espaces compris, parmi les cinq premiers "
        "objets. Pace ne compte pas. Un élément qui n'est pas un dict est sauté. "
        "Le sixième objet est ignoré. Une liste absente aussi. "
        "La valeur est dans le dict custom_settings, pas dans une clé pace à la racine. "
        "models/_settings.json n'a pas de pace : une valeur absente passe."
    )
    if raw_pace is None or (isinstance(raw_pace, str) and len(raw_pace.strip()) == 0):
        return
    raw_pace, collect_label, parse_error = _rewrite_custom_value(
        raw_pace, setting, merged, kind
    )
    if parse_error is not None:
        _raise_collected_int(
            "pace",
            raw_pace,
            collect_label or "pace",
            parse_error,
            "après NAG_alpha",
            pace_note,
            flag_note,
            architecture,
            shown,
        )
    type_fragment: str | None = None
    if isinstance(raw_pace, bool):
        type_fragment = "must be a integer" if kind == "integer" else "must be a number"
    elif kind == "number" and _color_strength_rejected(raw_pace):
        type_fragment = "must be a number"
    if type_fragment is not None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. custom_settings contient l'id pace et "
            f"pace vaut {raw_pace!r}, qui n'est pas un {kind}. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({label} {type_fragment}) après NAG_alpha. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{pace_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    if kind == "integer":
        parsed = _integer_setting_value(raw_pace)
        if isinstance(parsed, str):
            raise Wan2GPError(
                "La tâche est sautée avant toute vidéo. custom_settings contient l'id pace et "
                f"pace vaut {raw_pace!r}, qui n'est pas un entier. "
                "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
                f"({label} must be {parsed}) après NAG_alpha. "
                "collect_custom_settings_from_inputs transforme un type integer en text : "
                "un flottant 1.5 devient la chaîne 1.5, refusée ici au lieu d'être "
                "tronquée vers 1. Une chaîne 1.5 déjà écrite est refusée de la même façon. "
                "Un flottant entier (1.0, 2.0) reste soumis aux bornes. "
                "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
                "et quitte avec le code 0, sans MP4. "
                f"{pace_note} {flag_note} "
                f"Architecture finale {architecture}, lue dans {shown}."
            )
        number = float(parsed)
    else:
        parsed_number = _color_strength_number(raw_pace)
        if parsed_number is None:
            return
        number = parsed_number
    too_low = minimum is not None and number < minimum
    too_high = maximum is not None and number > maximum
    if not too_low and not too_high:
        return
    shown_pace = _format_color_strength(number)
    if too_low and minimum is not None:
        shown_limit = _format_color_strength(minimum)
        limit = f"en dessous de {shown_limit}"
        wan_message = f"must be at least {shown_limit}"
    elif too_high and maximum is not None:
        shown_limit = _format_color_strength(maximum)
        limit = f"au-dessus de {shown_limit}"
        wan_message = f"must be at most {shown_limit}"
    else:
        return
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. custom_settings contient l'id pace et "
        f"pace vaut {shown_pace}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({label} {wan_message}) après NAG_alpha. "
        "Il n'y a pas de défaut d'usine. 0.2 et 1 passent. 9, 9 en chaîne et 1.01 non. "
        "0.19 est sous 0.2. Un max illisible retire le plafond. "
        "Si min et max sont tous deux dépassés, le minimum sort d'abord. "
        "Si NAG_alpha 2.01 est visible, cette erreur-là sort avant. "
        "exaggeration vient après : une valeur 3 n'est pas ce saut. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{pace_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser pace entre 0.2 et 1, retirer l'id pace, ou relever max dans l'objet."
    )


_EXAGGERATION_ID = "exaggeration"
_EXAGGERATION_LABEL = "Exaggeration"
_EXAGGERATION_MIN = 0.25
_EXAGGERATION_MAX = 2.0


def _exaggeration_setting(merged: dict[str, object]) -> dict[str, object] | None:
    """Premier objet dont l'id est exactement ``exaggeration``, dans les cinq premiers.

    ``Exaggeration`` ne compte pas. Un élément qui n'est pas un dict est sauté.
    Une valeur qui n'est pas une liste, et le sixième objet, sont ignorés.
    """
    raw = merged.get("custom_settings")
    if not isinstance(raw, list):
        return None
    for item in raw[:_CUSTOM_SETTINGS_SCAN]:
        if not isinstance(item, dict):
            continue
        if str(item.get("id", "") or "").strip() == _EXAGGERATION_ID:
            return item
    return None


def _exaggeration_label(setting: dict[str, object]) -> str:
    """Libellé de ``exaggeration`` après ``_trim_label``.

    ``label`` gagne sur ``name``. Une chaîne vide retombe sur ``Exaggeration``.
    Un libellé qui ne laisse que ``(`` retombe sur la clé ``exaggeration``.
    """
    if "label" in setting:
        raw = setting["label"]
    else:
        raw = setting.get("name", _EXAGGERATION_LABEL)
    return _trim_setting_label(str(raw or _EXAGGERATION_LABEL), _EXAGGERATION_ID)


def _exaggeration_values(
    settings: dict[str, object],
    queue_custom_settings: object,
) -> tuple[object, str]:
    """Valeur de ``exaggeration`` et sa provenance.

    La file remplace tout le dict. ``None`` dans la file ne remplace pas.
    Une clé à la racine est ignorée. Une valeur qui n'est pas un dict
    est lue comme un dict vide.
    """
    if queue_custom_settings is not _QUEUE_MODEL_MODE_UNSET and queue_custom_settings is not None:
        bag: object = queue_custom_settings
        note = "La file envoie custom_settings."
    elif "custom_settings" in settings:
        bag = settings["custom_settings"]
        note = "Le JSON de réglages envoie custom_settings."
    else:
        return _QUEUE_MODEL_MODE_UNSET, "Pas de dict custom_settings : la valeur est absente."
    if not isinstance(bag, dict) or "exaggeration" not in bag:
        return _QUEUE_MODEL_MODE_UNSET, (
            f"{note} La clé exaggeration est absente, ou custom_settings n'est pas un dict."
        )
    return bag["exaggeration"], note


def _require_exaggeration(
    merged: dict[str, object],
    settings: dict[str, object],
    model_type: str,
    first_where: str,
    architecture_where: str,
    custom_settings_where: str,
    sliding_window_explicit: bool,
    first_architecture: str | None,
    queue_discard_last_frames: object,
    queue_custom_settings: object,
) -> None:
    """Refuse une exaggeration hors de 0,25 et 2.

    ``extra_settings.validate_inputs`` contrôle ``exaggeration`` après
    ``pace``. Le réglage est ``custom``. Il n'est visible que si
    ``model.custom_settings`` est une liste et qu'un des cinq premiers
    objets a l'id ``exaggeration``. La valeur vient de
    ``inputs["custom_settings"]["exaggeration"]``. Une clé à la racine
    est ignorée. Il n'y a pas de défaut dans ``models/_settings.json`` :
    une valeur absente passe. 3, ``3`` et 2,01 sont refusés. 0,24 est
    sous 0,25. ``validate_settings`` saute la tâche.
    ``process_tasks_cli`` sort 0, sans MP4. Si ``pace`` 9 est visible,
    cette erreur-là sort avant. ``collect_custom_settings_from_inputs``
    réécrit le dict avant ``extra_settings``. Un type ``integer`` et un
    flottant 1,5 deviennent la chaîne ``1.5`` (« must be an integer »).
    Un type ``int`` et le même flottant sont refusés avant les bornes.
    """
    if _frame_scheduler_skips_extra_settings(
        merged,
        settings,
        sliding_window_explicit,
        first_architecture,
        queue_discard_last_frames,
    ):
        return
    if _iter_defs_breaks_before_pace(merged):
        return
    setting = _exaggeration_setting(merged)
    if setting is None:
        return
    kind = _setting_kind(setting.get("type", "number"))
    if kind == "string":
        return
    label = _exaggeration_label(setting)
    minimum = _pace_bound(setting, "min", _EXAGGERATION_MIN)
    maximum = _pace_bound(setting, "max", _EXAGGERATION_MAX)
    raw_value, value_note = _exaggeration_values(settings, queue_custom_settings)
    if raw_value is _QUEUE_MODEL_MODE_UNSET:
        return
    architecture = merged.get("architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = model_type
    shown = custom_settings_where or architecture_where or first_where or model_type
    flag_note = (
        "L'id exaggeration est lu tel quel, espaces compris, parmi les cinq premiers "
        "objets. Exaggeration ne compte pas. Un élément qui n'est pas un dict est sauté. "
        "Le sixième objet est ignoré. Une liste absente aussi. "
        "La valeur est dans le dict custom_settings, pas dans une clé exaggeration à la racine. "
        "models/_settings.json n'a pas d'exaggeration : une valeur absente passe."
    )
    if raw_value is None or (isinstance(raw_value, str) and len(raw_value.strip()) == 0):
        return
    raw_value, collect_label, parse_error = _rewrite_custom_value(
        raw_value, setting, merged, kind
    )
    if parse_error is not None:
        _raise_collected_int(
            "exaggeration",
            raw_value,
            collect_label or "exaggeration",
            parse_error,
            "après pace",
            value_note,
            flag_note,
            architecture,
            shown,
        )
    type_fragment: str | None = None
    if isinstance(raw_value, bool):
        type_fragment = "must be a integer" if kind == "integer" else "must be a number"
    elif kind == "number" and _color_strength_rejected(raw_value):
        type_fragment = "must be a number"
    if type_fragment is not None:
        raise Wan2GPError(
            "La tâche est sautée avant toute vidéo. custom_settings contient l'id exaggeration et "
            f"exaggeration vaut {raw_value!r}, qui n'est pas un {kind}. "
            "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
            f"({label} {type_fragment}) après pace. "
            "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
            "et quitte avec le code 0, sans MP4. "
            f"{value_note} {flag_note} "
            f"Architecture finale {architecture}, lue dans {shown}."
        )
    if kind == "integer":
        parsed = _integer_setting_value(raw_value)
        if isinstance(parsed, str):
            raise Wan2GPError(
                "La tâche est sautée avant toute vidéo. custom_settings contient l'id exaggeration et "
                f"exaggeration vaut {raw_value!r}, qui n'est pas un entier. "
                "extra_settings.validate_inputs (shared/extra_settings.py) renvoie une erreur "
                f"({label} must be {parsed}) après pace. "
                "collect_custom_settings_from_inputs transforme un type integer en text : "
                "un flottant 1.5 ou 2.9 devient une chaîne, refusée ici au lieu d'être "
                "tronquée vers 1 ou 2. Une chaîne 1.5 déjà écrite est refusée de la même façon. "
                "Un flottant entier (2.0) reste soumis aux bornes. "
                "validate_settings la propage. process_tasks_cli compte la tâche comme sautée "
                "et quitte avec le code 0, sans MP4. "
                f"{value_note} {flag_note} "
                f"Architecture finale {architecture}, lue dans {shown}."
            )
        number = float(parsed)
    else:
        parsed_number = _color_strength_number(raw_value)
        if parsed_number is None:
            return
        number = parsed_number
    too_low = minimum is not None and number < minimum
    too_high = maximum is not None and number > maximum
    if not too_low and not too_high:
        return
    shown_value = _format_color_strength(number)
    if too_low and minimum is not None:
        shown_limit = _format_color_strength(minimum)
        limit = f"en dessous de {shown_limit}"
        wan_message = f"must be at least {shown_limit}"
    elif too_high and maximum is not None:
        shown_limit = _format_color_strength(maximum)
        limit = f"au-dessus de {shown_limit}"
        wan_message = f"must be at most {shown_limit}"
    else:
        return
    raise Wan2GPError(
        "La tâche est sautée avant toute vidéo. custom_settings contient l'id exaggeration et "
        f"exaggeration vaut {shown_value}, {limit}. "
        "extra_settings.validate_inputs (shared/extra_settings.py) refuse la valeur "
        f"({label} {wan_message}) après pace. "
        "Il n'y a pas de défaut d'usine. 0.25 et 2 passent. 3, 3 en chaîne et 2.01 non. "
        "0.24 est sous 0.25. Un max illisible retire le plafond. "
        "Si min et max sont tous deux dépassés, le minimum sort d'abord. "
        "Si pace 9 est visible, cette erreur-là sort avant. "
        "Un type integer et un flottant non entier sont déjà refusés comme chaîne. "
        "validate_settings renvoie cette erreur. process_tasks_cli compte la tâche "
        "comme sautée et quitte avec le code 0, sans MP4. "
        f"{value_note} {flag_note} "
        f"Architecture finale {architecture}, lue dans {shown}. "
        "Poser exaggeration entre 0.25 et 2, retirer l'id exaggeration, ou relever max dans l'objet."
    )


def _wan_mp4_target(root: Path | None) -> str:
    """Encodeur ffmpeg exigé par ``wgp_config.json``. Défaut WanGP 18.84 : libx264.

    Le conteneur vient du fichier de config du clone, pas de la file. ``mkv``
    ou ``mov`` produit une vidéo que ce pipeline ne peut pas accepter comme MP4.
    Une clé ``video_container`` nulle ou vide non plus : le fichier n'a pas
    l'extension ``.mp4``. Le JSON lu est celui que ``open`` atteint quand le
    cwd est le clone. Un lien ``/proc/self`` ne doit pas faire lire le leurre
    du pipeline, ni ignorer le fichier du clone.
    """
    codec = "libx264_8"
    payload: dict[str, object] = {}
    logical = None if root is None else root / "wgp_config.json"
    opened = None if root is None else _wan_config_file(root)
    if opened is not None:
        try:
            loaded = json.loads(opened.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise Wan2GPError(f"wgp_config.json illisible ({opened}) : {exc}") from exc
        if not isinstance(loaded, dict):
            raise Wan2GPError(f"wgp_config.json doit être un objet ({opened}).")
        payload = loaded
        codec = str(payload.get("video_output_codec") or "libx264_8").strip().lower() or "libx264_8"
    where = _config_where(logical, opened)
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
        entries = _ld_search_entries(seen, ffmpeg, root)
        needed = (
            _needed_sonames(ffmpeg, seen)
            if any(_subject_is_nonfile_library(path, root) for path in entries)
            else set()
        )
        masked = _preload_masked_sonames(ffmpeg, root, seen) if root is not None else set()
        traps = [
            trap
            for trap in _library_traps(entries, needed, seen, root)
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
    ``sys.path[0]`` initial est le ``realpath`` de ce script. ``wgp.py`` insère
    devant le dossier de ``os.path.abspath(__file__)``, qui ne suit pas le lien.
    Sans module à cet endroit, l'import retombe sur le dossier réel. Le cwd du
    processus reste le clone passé au runner.
    Un lien qui passe par ``/proc/self`` se résout avec ce cwd. ``Path.resolve``
    dans le contrôle suit le cwd du pipeline et peut viser un autre
    ``ffmpeg_bins``, ou ignorer le fichier s'il n'existe que dans le clone.
    ``wgp.py`` lui-même peut être ce lien (``/proc/self/cwd/nested/wgp.py``).
    """
    if os.name == "nt":
        return root
    resolved = _resolved_ffmpeg_setup(root / "shared" / "ffmpeg_setup.py", root)
    if resolved is not None:
        return resolved
    wgp = root / "wgp.py"
    try:
        linked = wgp.is_symlink()
    except OSError:
        linked = False
    if not linked:
        return root
    parent = _symlink_script_parent(wgp, root)
    if parent is None:
        return root
    nested = _resolved_ffmpeg_setup(parent / "shared" / "ffmpeg_setup.py", root)
    if nested is not None:
        return nested
    return parent


def _visible_wan_script(root: Path) -> Path | None:
    """``wgp.py`` ouvert quand le cwd du processus est le clone.

    ``Path.is_file`` suit ``/proc/self`` du contrôle. Un leurre dans le
    dossier du pipeline rend le script présent, et l'absence de leurre le
    rend introuvable, alors que le clone a ``nested/wgp.py``.
    """
    wgp = root / "wgp.py"
    try:
        absolute = str(wgp.absolute())
    except OSError:
        return None
    if os.name != "nt" and _path_depends_on_proc_cwd(absolute):
        opened = _realpath_from_cwd(wgp, root)
        if opened is None:
            return None
        try:
            if not opened.is_file():
                return None
        except OSError:
            return None
        return opened
    try:
        if wgp.is_file():
            return wgp
    except OSError:
        return None
    return None


def _symlink_script_parent(wgp: Path, root: Path) -> Path | None:
    """Dossier que Python place dans ``sys.path[0]`` pour ce lien.

    C'est le ``realpath`` du script, pas ``Path.resolve()`` du contrôle.
    ``/proc/self`` dépend du cwd. Le processus WanGP a celui du clone.
    """
    try:
        absolute = str(wgp.absolute())
    except OSError:
        return None
    if os.name != "nt" and _path_depends_on_proc_cwd(absolute):
        opened = _realpath_from_cwd(wgp, root)
        if opened is None:
            return None
        try:
            if not opened.is_file():
                return None
        except OSError:
            return None
        return opened.parent
    try:
        if not wgp.is_file():
            return None
        return wgp.resolve().parent
    except OSError:
        return None


def _realpath_from_cwd(path: Path, cwd: Path) -> Path | None:
    """``realpath`` de ``path`` dans un processus dont le cwd est ``cwd``.

    ``/proc/self`` désigne ce processus. ``None`` si le chemin n'existe pas
    depuis ce cwd. Un délai est une erreur : le même lien bloquerait
    ``download_ffmpeg``.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _LOADER_DIRECTORY, str(path.absolute())],
            check=False,
            capture_output=True,
            text=True,
            cwd=_probe_cwd(cwd),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(
            f"{path} ne revient pas quand le cwd est {cwd}. "
            "Ce lien traverse /proc/self. download_ffmpeg résout "
            "shared/ffmpeg_setup.py avec le cwd du clone et préfixe un autre "
            "ffmpeg_bins. save_video imprime error saving et le processus "
            "quitte avec le code 0, sans MP4."
        ) from exc
    if completed.returncode != 0:
        return None
    opened = completed.stdout.strip()
    if not opened:
        return None
    return Path(opened)


def _resolved_ffmpeg_setup(path: Path, cwd: Path | None = None) -> Path | None:
    """Parent du clone réel, ou ``None`` si le module n'est pas un fichier.

    ``cwd`` est celui du processus WanGP. Un lien ``shared`` ou
    ``ffmpeg_setup.py`` vers ``/proc/self/cwd/...`` n'a pas la même cible
    ici et dans ce processus. ``Path.resolve()`` du contrôle ne suffit pas :
    il ouvre le dossier du pipeline, y compris quand un leurre rend
    ``is_file`` vrai.
    """
    try:
        absolute = str(path.absolute())
    except OSError:
        return None
    if cwd is not None and os.name != "nt" and _path_depends_on_proc_cwd(absolute):
        opened = _realpath_from_cwd(path, cwd)
        if opened is None:
            return None
        try:
            return opened.parents[1]
        except IndexError:
            return None
    try:
        if not path.is_symlink() and not path.is_file():
            return None
        return path.resolve().parents[1]
    except (OSError, IndexError):
        return None


def _proc_setup_note(root: Path) -> str:
    """Pourquoi ``Path.resolve`` du contrôle ne voit pas le module de WanGP."""
    setup = root / "shared" / "ffmpeg_setup.py"
    try:
        absolute = str(setup.absolute())
    except OSError:
        return ""
    if not _path_depends_on_proc_cwd(absolute):
        return ""
    opened = _realpath_from_cwd(setup, root)
    if opened is None:
        located = "un dossier que ce cwd ne voit pas"
    else:
        try:
            located = str(opened.parents[1])
        except IndexError:
            located = str(opened)
    return (
        f"{setup} suit /proc/self jusqu'au cwd du processus. "
        f"Path.resolve() depuis le pipeline n'ouvre pas {located}. "
        "Le processus WanGP a pour cwd le clone : Path(__file__).resolve() "
        "dans shared/ffmpeg_setup.py retombe sur ce dossier. "
        "download_ffmpeg préfixe son ffmpeg_bins. "
        "Une libx264.so* trop courte, absente du dossier vu d'ici, donne file too short. "
        "ffmpeg -encoders sans ce préfixe annonce encore libx264. "
        "save_video imprime error saving et le processus quitte avec le code 0, sans MP4. "
    )


def _proc_script_note(root: Path, library_root: Path) -> str:
    """Pourquoi ``Path.resolve`` du contrôle ne voit pas le ``wgp.py`` de WanGP.

    Le lien est sur le script, pas sur ``shared``. ``sys.path[0]`` initial
    est le ``realpath`` calculé avec le cwd du clone. Sans module dans le
    dossier du lien, l'import charge ``nested/shared/ffmpeg_setup.py``.
    """
    wgp = root / "wgp.py"
    try:
        absolute = str(wgp.absolute())
    except OSError:
        return ""
    if os.name == "nt" or not _path_depends_on_proc_cwd(absolute):
        return ""
    opened = _realpath_from_cwd(wgp, root)
    if opened is None:
        return ""
    parent = opened.parent
    if not _same_directory(parent, library_root):
        return ""
    return (
        f"{wgp} suit /proc/self jusqu'au cwd du processus. "
        f"Path.resolve() depuis le pipeline n'ouvre pas {parent}. "
        "Python met le realpath du script dans sys.path[0]. "
        "wgp.py insère aussi os.path.dirname(os.path.abspath(__file__)), "
        "qui ne résout pas le lien. Sans shared/ffmpeg_setup.py dans ce dossier, "
        "l'import tombe sur le dossier réel. "
        "Le processus WanGP a pour cwd le clone : ce realpath ouvre le script du clone. "
        "Path(__file__).resolve() dans shared/ffmpeg_setup.py préfixe son ffmpeg_bins. "
        "Une libx264.so* trop courte, absente du dossier vu d'ici, donne file too short. "
        "ffmpeg -encoders sans ce préfixe annonce encore libx264. "
        "save_video imprime error saving et le processus quitte avec le code 0, sans MP4. "
    )


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
    """``ffmpeg_bins`` absolu que ``download_ffmpeg`` ajoute à ``LD_LIBRARY_PATH``.

    ``Path(__file__).resolve()`` est absolu. Un ``WAN2GP_ROOT`` relatif ne
    doit pas le rester : le cwd du mux est le clone, et ``wan/ffmpeg_bins``
    y ouvrirait ``<clone>/wan/ffmpeg_bins``. Le lien ``ffmpeg_bins`` lui-même
    n'est pas suivi : c'est ce nom que WanGP préfixe.
    """
    return (_ffmpeg_library_root(root) / "ffmpeg_bins").absolute()


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


_LOADER_DIRECTORY = """
import os
import sys

try:
    print(os.path.realpath(sys.argv[1]), flush=True)
except OSError:
    raise SystemExit(1)
"""


def _directory_opened_by_loader(bundle: Path, root: Path | None) -> Path:
    """Dossier que le chargeur ouvre quand le cwd du processus est le clone.

    ``download_ffmpeg`` préfixe ``LD_LIBRARY_PATH`` avec le chemin de
    ``ffmpeg_bins``, sans le résoudre. Un lien vers ``/proc/self/cwd`` ou
    ``/proc/net/../cwd`` suit le cwd du processus qui charge la bibliothèque.
    ``Path.resolve`` et ``iterdir``, ici, suivent ``/proc/self`` du contrôle :
    ils listent le dossier du pipeline. Le mux a pour cwd le clone.
    """
    if root is None or os.name == "nt":
        return bundle
    try:
        absolute = str(bundle.absolute())
    except OSError:
        return bundle
    if not _path_depends_on_proc_cwd(absolute):
        return bundle
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _LOADER_DIRECTORY, absolute],
            check=False,
            capture_output=True,
            text=True,
            cwd=_probe_cwd(root),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(
            f"{bundle} ne revient pas quand le cwd est le clone. "
            "Ce lien traverse /proc/self. download_ffmpeg le préfixe sur "
            "LD_LIBRARY_PATH. Le chargeur ouvre alors un soname du clone, "
            "pas celui vu depuis le dossier du pipeline. "
            "save_video imprime error saving et le processus quitte avec le code 0, sans MP4."
        ) from exc
    if completed.returncode != 0:
        return bundle
    opened = completed.stdout.strip()
    if not opened:
        return bundle
    return Path(opened)


def _path_uses_proc_cwd(path: Path) -> bool:
    """``True`` si ouvrir ``path`` suit le cwd magique de ``/proc``."""
    try:
        absolute = str(path.absolute())
    except OSError:
        return False
    return _path_depends_on_proc_cwd(absolute)


def _loader_subject(path: Path, root: Path | None) -> Path | None:
    """Chemin que le chargeur ouvre quand le cwd est le clone.

    ``None`` : le lien traverse ``/proc/self`` et le clone ne l'ouvre pas.
    ``is_file`` dans le contrôle suivrait alors le cwd du pipeline, souvent
    un autre fichier. Un lien indépendant de ce cwd est renvoyé tel quel.
    """
    if root is None or os.name == "nt" or not _path_uses_proc_cwd(path):
        return path
    viewed = _directory_opened_by_loader(path, root)
    if viewed == path:
        return None
    return viewed


def _subject_is_nonfile_library(path: Path, root: Path | None) -> bool:
    """Nom de bibliothèque que le chargeur ouvre sans y trouver un fichier."""
    if not _is_shared_library(path):
        return False
    subject = _loader_subject(path, root)
    if subject is None:
        return False
    try:
        return not subject.is_file()
    except OSError:
        return True


def _search_directory(logical: Path, root: Path | None, notes: list[str] | None) -> Path:
    """Dossier que glibc ouvre, pas celui vu depuis le pipeline.

    ``glibc-hwcaps`` ou ``glibc-hwcaps/<niveau>`` peut viser ``/proc/self/cwd``.
    ``iterdir`` ici suit le cwd du contrôle. Le mux a pour cwd le clone.
    """
    opened = _directory_opened_by_loader(logical, root)
    if notes is not None and opened != logical:
        notes.append(
            f"{logical} suit /proc/self jusqu'au cwd du processus. "
            f"Le chargeur, avec le cwd du clone, ouvre {opened}, "
            "pas le dossier que iterdir voit depuis le pipeline. "
            "glibc cherche d'abord glibc-hwcaps/<niveau>/<soname> dans ce dossier. "
            "download_ffmpeg préfixe quand même LD_LIBRARY_PATH avec ffmpeg_bins. "
            "ffmpeg -encoders sans ce préfixe annonce encore libx264. "
            "Avec le préfixe, le soname trop court du clone donne file too short. "
            "save_video imprime error saving et le processus quitte avec le code 0, sans MP4. "
        )
    return opened


def _proc_shadow_note(bundle: Path, scan: Path) -> str:
    """Pourquoi ``iterdir`` du pipeline ne voit pas le soname du clone."""
    if scan == bundle:
        return ""
    try:
        pipeline_view = bundle.resolve()
    except OSError:
        pipeline_view = bundle
    if scan == pipeline_view:
        return ""
    return (
        f"{bundle} suit /proc/self jusqu'au cwd du processus. "
        f"Le chargeur, avec le cwd du clone, ouvre {scan}, pas {pipeline_view} "
        "(le dossier que iterdir voit depuis le pipeline). "
        "download_ffmpeg écrit ce lien dans LD_LIBRARY_PATH. "
        "ffmpeg -encoders sans ce préfixe annonce encore libx264. "
        "Avec le préfixe, le soname trop court du clone donne file too short. "
        "save_video imprime error saving et le processus quitte avec le code 0, sans MP4. "
    )


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
    Un lien ``ffmpeg_bins`` vers ``/proc/self/cwd`` ou ``/proc/net/../cwd``
    est un dossier pour ``iterdir`` : depuis le pipeline, ce dossier est le
    cwd du pipeline. Le chargeur suit le même lien avec le cwd du clone et
    ouvre la racine du clone. Un ``libx264.so*`` trop court posé là donne
    ``file too short`` au mux, alors que ``-encoders`` sans le préfixe
    annonce encore libx264. Le contrôle relit le dossier depuis un
    processus dont le cwd est le clone.
    Le même écart existe sous ce dossier. ``glibc-hwcaps`` ou un niveau
    ``glibc-hwcaps/<niveau>`` peut viser ``/proc/self/cwd`` pendant que
    ``ffmpeg_bins`` reste un vrai dossier : ``iterdir`` liste le cwd du
    pipeline. Un lien ``libx264.so.<n>`` vers ``/proc/self/cwd/<fichier>``
    est un soname pour le chargeur et un lien cassé, ou un autre fichier,
    pour ``is_file`` ici. Le clone ouvre le fichier trop court.
    """
    if os.name == "nt" or root is None:
        return
    library_root = _ffmpeg_library_root(root)
    bundle = library_root / "ffmpeg_bins"
    scan = _directory_opened_by_loader(bundle, root)
    try:
        if not scan.is_dir():
            return
    except OSError:
        return
    notes: list[str] = []
    entries = _ld_search_entries(scan, ffmpeg, root, notes)
    proc_note = _proc_shadow_note(bundle, scan)
    # Un dossier ou un tube dont le nom ne ressemble pas à une bibliothèque
    # n'est pas ouvert par le chargeur. Inutile de parcourir la fermeture.
    needed = (
        _needed_sonames(ffmpeg, bundle)
        if any(_subject_is_nonfile_library(path, root) for path in entries)
        else set()
    )
    traps = _library_traps(entries, needed, scan, root)
    foreign = "" if _same_directory(library_root, root) else _foreign_library_note(library_root, root)
    proc_setup = _proc_setup_note(root)
    proc_script = _proc_script_note(root, library_root)
    if traps:
        shown = ", ".join(traps)
        raise Wan2GPError(
            proc_setup
            + proc_script
            + proc_note
            + f"{bundle} contient {shown}, soname chargé par {ffmpeg}. "
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
    shadows: list[str] = []
    seen_labels: set[str] = set()
    file_note = ""
    for path in entries:
        subject = _loader_subject(path, root)
        if subject is None or not _is_shared_library(path) or not _is_regular_file(subject):
            continue
        label = _library_label(scan, path)
        if label == path.name and path.parent.name in _glibc_hwcap_names(ffmpeg):
            label = f"glibc-hwcaps/{path.parent.name}/{path.name}"
        if label not in seen_labels:
            seen_labels.add(label)
            shadows.append(label)
        if not file_note and subject != path:
            file_note = (
                f"{path} suit /proc/self jusqu'au cwd du processus. "
                f"Depuis le pipeline, ce lien n'est pas {subject}. "
                "Le chargeur, cwd = clone, ouvre ce soname et obtient file too short. "
                "ffmpeg -encoders sans LD_LIBRARY_PATH annonce encore libx264. "
                "save_video imprime error saving et le processus quitte avec le code 0, sans MP4. "
            )
    if not shadows:
        return
    shown = ", ".join(sorted(shadows))
    raise Wan2GPError(
        proc_setup
        + proc_script
        + proc_note
        + "".join(notes)
        + file_note
        + _quarantine_link_note(root)
        + f"{bundle} contient {shown} alors que l'encodeur retenu est {ffmpeg}. "
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


def _ld_search_entries(
    bundle: Path,
    ffmpeg: Path,
    root: Path | None = None,
    notes: list[str] | None = None,
) -> list[Path]:
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
    ``glibc-hwcaps`` et chaque niveau sont relus avec le cwd du clone quand
    le chemin traverse ``/proc/self``. ``iterdir`` depuis le pipeline y
    verrait un autre dossier, et un ``libx264.so*`` trop court du clone
    passerait.
    """
    try:
        entries = list(bundle.iterdir())
    except OSError:
        return _probe_unlisted_bundle(bundle, ffmpeg)
    allowed = _glibc_hwcap_names(ffmpeg)
    if not allowed:
        return entries
    hwcaps = _search_directory(bundle / "glibc-hwcaps", root, notes)
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
        opened = _search_directory(child, root, notes)
        try:
            is_directory = opened.is_dir()
        except OSError:
            hidden.append(child.name)
            continue
        if not is_directory:
            continue
        try:
            entries.extend(opened.iterdir())
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


def _library_traps(
    entries: list[Path],
    needed: set[str] | None,
    bundle: Path,
    root: Path | None = None,
) -> list[str]:
    """Chemins que le chargeur ouvrirait à la place d'une bibliothèque.

    Un lien cassé ou un socket de ce nom laisse le chargeur passer au
    suivant. Un dossier, un tube, ou un périphérique (lien vers ``/dev/null``)
    est ouvert et fait échouer ffmpeg. La liste ``needed`` couvre les
    sonames directs et ceux demandés par les bibliothèques déjà chargées.
    Le contrôle vaut aussi quand le binaire est dans ``ffmpeg_bins`` : un
    tube du soname y bloque le mux d'essai, et ``libavdevice.so*`` suffit
    au glob de présence. Un soname sous ``glibc-hwcaps/<niveau>/`` est
    ouvert avant celui de la racine. Un lien qui traverse ``/proc/self``
    est statué depuis le cwd du clone : le cwd du pipeline ouvre un autre
    fichier, ou aucun.
    """
    traps: list[str] = []
    for path in sorted(entries, key=lambda item: _library_label(bundle, item)):
        subject = _loader_subject(path, root)
        if subject is None:
            continue
        try:
            if subject.is_file():
                continue
        except OSError:
            continue
        if needed is None:
            if not _is_shared_library(path):
                continue
        elif path.name not in needed:
            continue
        kind = _path_kind(subject)
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


def _wan_directories(root: Path) -> tuple[str, str]:
    """``ffmpeg_bins`` puis la racine, absolus, comme après ``download_ffmpeg``.

    La racine est celle de ``Path(__file__).resolve()``, pas forcément le
    dossier passé au runner. Un lien ``ffmpeg_bins`` n'est pas résolu.
    """
    library = _ffmpeg_library_root(root)
    bundle = os.path.normpath(str((library / "ffmpeg_bins").absolute()))
    repo = os.path.normpath(str(library.absolute()))
    return bundle, repo


# ``os.stat`` suit le lien. Un tube ou un fichier n'est pas un dossier de PATH.
_PROC_CWD_STAT = """
import os
import stat
import sys

try:
    found = os.stat(sys.argv[1])
except OSError:
    raise SystemExit(1)
if not stat.S_ISDIR(found.st_mode):
    raise SystemExit(1)
print(f"{found.st_dev} {found.st_ino}")
"""

_PROC_CWD_MARKERS = ("/proc/self/cwd", "/proc/thread-self/cwd")
_PROC_SELF_COMPONENTS = frozenset({"self", "thread-self"})


def _collapse_slash_and_dot(path: str) -> str:
    """Enlève les ``.`` et les barres doubles, sans annuler ``..``.

    ``/proc/./self/cwd/`` et ``/proc//self/cwd`` sont le même dossier que
    ``/proc/self/cwd``. Annuler ``..`` ici cacherait un lien de dossier.
    """
    pieces = [piece for piece in path.split("/") if piece not in ("", ".")]
    return "/" + "/".join(pieces)


def _is_proc_cwd_path(path: str) -> bool:
    """``True`` si ``path`` passe par le cwd magique de ``/proc``."""
    collapsed = _collapse_slash_and_dot(path)
    return any(
        collapsed == marker or collapsed.startswith(marker + "/")
        for marker in _PROC_CWD_MARKERS
    )


def _mentions_proc_self(path: str) -> bool:
    """``True`` si ``proc`` apparaît avant ``self`` ou ``thread-self``.

    ``..`` et les dossiers réels restent en place. ``normpath`` les efface :
    ``/proc/self/../self/cwd`` et ``/proc/sys/../self/cwd`` deviennent
    ``/proc/self/cwd``, ``/proc/thread-self/../../cwd`` devient ``/cwd``.
    Le noyau entre dans le lien ou dans le dossier (``/proc/sys`` existe)
    avant d'appliquer ``..``. ``self`` n'a pas besoin de toucher ``proc`` :
    ``/proc/sys/../self/cwd`` ouvre encore le cwd du processus. La séquence
    peut être au milieu (``/tmp/../proc/sys/../self/cwd``).
    """
    pieces = [piece for piece in path.split("/") if piece not in ("", ".")]
    seen_proc = False
    for piece in pieces:
        if piece == "proc":
            seen_proc = True
            continue
        if seen_proc and piece in _PROC_SELF_COMPONENTS:
            return True
    return False


def _proc_self_target(raw: str, joined: str) -> bool:
    """``True`` si la cible d'un lien désigne ``/proc/self`` ou ``thread-self``."""
    return any(
        _is_proc_cwd_path(item) or _mentions_proc_self(item) for item in (raw, joined)
    )


def _outer_symlink(path: Path) -> Path | None:
    """Premier lien en partant de la racine, sans le suivre.

    ``lstat`` du chemin complet suit les liens intermédiaires. Sur
    ``lien/cwd`` avec ``lien`` → ``/proc/self``, il lirait la cible de
    ``cwd`` (le dossier réel du contrôle) et perdrait ``/proc/self``.
    """
    parts = path.parts
    if len(parts) < 2:
        return None
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            is_link = current.is_symlink()
        except OSError:
            return None
        if is_link:
            return current
    return None


def _proc_walk_key(path: str) -> str:
    """Clé de boucle qui conserve ``..``.

    ``normpath`` annule ``..`` avant le lien. ``hop1/../cwd`` et
    ``hop2/../cwd`` deviennent le même texte quand les deux liens sont dans
    le même dossier, alors que le noyau entre dans chacun. ``hop2`` peut
    pointer vers ``/proc/net`` : ``..`` retombe alors dans ``/proc/self``.
    Les ``.`` et les barres doubles ne changent pas cette marche.
    """
    pieces = [piece for piece in str(path).split("/") if piece not in ("", ".")]
    if str(path).startswith("/"):
        return "/" + "/".join(pieces)
    return "/".join(pieces)


def _path_depends_on_proc_cwd(path: str) -> bool:
    """``True`` si ouvrir ``path`` suit ``/proc/self`` ou ``thread-self``.

    Le contrôle ne partage pas le cwd du clone. ``Path.resolve()`` ouvre
    alors le dossier du pipeline. Un lien dont la cible est ce chemin proc,
    ou un ``..`` qui ne ramène au cwd qu'après être entré dans le lien ou
    dans un dossier réel de ``/proc`` (``/proc/sys``), a le même effet une
    fois le cwd devenu le clone. Après chaque lien, le chemin recomposé est
    relu : ``lien/../self/cwd`` avec ``lien`` → ``/proc/sys`` ne contient
    pas ``proc`` dans le texte d'origine. Une chaîne ``hop1`` → ``hop2``
    → ``/proc/net`` avec l'entrée ``hop1/../cwd`` non plus. ``normpath``
    confond ``hop1/../cwd`` et ``hop2/../cwd`` : ce n'est pas une boucle,
    le second lien n'a pas encore été lu.
    """
    if _is_proc_cwd_path(path) or _mentions_proc_self(path):
        return True
    node = Path(path)
    seen: set[str] = set()
    for _ in range(32):
        key = _proc_walk_key(str(node))
        if key in seen:
            return False
        seen.add(key)
        link = _outer_symlink(node)
        if link is None:
            return False
        try:
            raw = os.readlink(link)
        except OSError:
            return False
        joined = raw if os.path.isabs(raw) else str(link.parent / raw)
        if _proc_self_target(raw, joined):
            return True
        try:
            suffix = node.relative_to(link)
        except ValueError:
            return False
        node = Path(joined) if str(suffix) == "." else Path(joined) / suffix
        if _is_proc_cwd_path(str(node)) or _mentions_proc_self(str(node)):
            return True
    return False


def _proc_cwd_opens_directory(candidate: str, repo: str, root: Path) -> bool:
    """``True`` si le clone, comme cwd, ouvre ``repo`` via ``candidate``.

    ``/proc/self/cwd`` change avec le processus. L'inode est lu dans un
    processus dont le cwd est celui de WanGP.
    """
    try:
        repo_stat = os.stat(repo)
    except OSError:
        return False
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROC_CWD_STAT, candidate],
            check=False,
            capture_output=True,
            text=True,
            cwd=_probe_cwd(root),
            timeout=_PATH_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise Wan2GPError(
            f"{candidate} ne revient pas quand le cwd est le clone. "
            "download_ffmpeg laisse cette entrée sur le PATH. "
            "exec ffmpeg peut y trouver le fichier que la quarantaine va déplacer. "
            "save_video imprime error saving et le processus quitte avec le code 0, sans MP4."
        ) from exc
    if completed.returncode != 0:
        return False
    fields = completed.stdout.split()
    if len(fields) != 2:
        return False
    try:
        found = (int(fields[0]), int(fields[1]))
    except ValueError:
        return False
    return found == (repo_stat.st_dev, repo_stat.st_ino)


def _same_path_entry(part: str, repo: str, root: Path | None = None) -> bool:
    """``True`` si ouvrir ``part`` arrive sur la racine du clone.

    ``normpath`` annule ``..`` avant un lien de dossier. ``holder/trick/..``
    devient ``holder`` dans le texte, alors que le noyau entre dans ``trick``
    puis remonte jusqu'au clone. Une entrée relative (``.`` ou ``trick/..``)
    est ouverte depuis le cwd du clone, pas depuis le dossier du pipeline.
    ``/proc/self/cwd`` et ``/proc/thread-self/cwd`` dépendent du cwd du
    processus : depuis le clone elles ouvrent cette racine, alors que
    ``Path.resolve()`` dans le contrôle ouvre le dossier du pipeline.
    ``normpath`` annule ``..`` avant ces liens. ``/proc/self/../self/cwd``,
    ``/proc/thread-self/../../cwd`` et ``/proc/self/root/proc/self/cwd``
    ouvrent la même racine une fois le cwd devenu le clone. Un lien de
    dossier vers ``/proc/self`` suivi de ``cwd`` aussi.     ``/proc/sys`` est
    un dossier réel : ``/proc/sys/../self/cwd`` n'a pas ``self`` juste
    après ``proc``, et ``normpath`` en fait quand même ``/proc/self/cwd``.
    Le noyau entre dans ``sys`` avant ``..``. Un lien vers ``/proc/sys``
    suivi de ``../self/cwd`` ouvre la même racine. Deux liens dans le même
    dossier aussi : ``hop1`` → ``hop2`` → ``/proc/net``, entrée
    ``hop1/../cwd``. ``normpath`` confond cette entrée avec ``hop2/../cwd``
    et une boucle s'arrêterait avant ``hop2``. Le noyau entre dans les deux
    liens, puis ``..`` ouvre ``/proc/self/cwd``. Pendant le probe,
    ce dossier doit quitter le PATH : ``exec ffmpeg`` y trouverait encore
    le fichier que la quarantaine va déplacer.
    """
    if os.path.normpath(part) == repo:
        return True
    candidate = part
    if not os.path.isabs(part):
        if root is None:
            return False
        candidate = str((root / part).absolute())
    try:
        opened = Path(candidate).resolve()
        if opened == Path(repo).resolve():
            return True
    except OSError:
        pass
    if root is None or not _path_depends_on_proc_cwd(candidate):
        return False
    return _proc_cwd_opens_directory(candidate, repo, root)


def _wan_path(root: Path, current: str) -> str:
    """PATH après ``_ensure_bin_dir_on_path`` : ``ffmpeg_bins``, puis la racine.

    Les entrées sont absolues et ne suivent pas les liens. ``ffmpeg_bins/ffmpeg``
    peut rester un lien vers le binaire du système : c'est ce nom que WanGP
    met en tête, pas ``/usr/bin``. Pendant un probe, la racine est un dossier
    sans le ``ffmpeg`` qui part en quarantaine : le PATH d'origine qui
    répétait cette racine est retiré, sinon ``exec ffmpeg`` trouverait encore
    le fichier. Une entrée qui ne s'écrit pas comme cette racine est retirée
    aussi quand le noyau l'ouvre quand même (lien de dossier puis ``..``,
    chemin relatif depuis le cwd du clone, ou ``/proc/self`` traversé avant
    ``..``).
    """
    bundle, repo = _wan_directories(root)
    override = _PROBE_REPO_OVERRIDE.get()
    directories = [bundle, override if override is not None else repo]
    prioritized: list[str] = []
    seen: set[str] = set()
    for directory in directories:
        if directory in seen:
            continue
        prioritized.append(directory)
        seen.add(directory)
    hidden = {repo} if override is not None else set()
    kept: list[str] = []
    for part in current.split(os.pathsep):
        if not part:
            continue
        normalized = os.path.normpath(part)
        if normalized in seen or any(
            _same_path_entry(part, item, root) for item in hidden
        ):
            continue
        kept.append(part)
        seen.add(normalized)
    return os.pathsep.join([*prioritized, *kept])


def _repo_ffmpeg_file(root: Path) -> Path | None:
    """Fichier ``ffmpeg`` que ``_quarantine_root_ffmpeg`` va déplacer.

    Un dossier, un tube ou un socket reste en place (``is_file`` est faux).
    Un lien vers un exécutable est un fichier : WanGP déplace le lien.
    """
    if os.name == "nt":
        return None
    candidate = _ffmpeg_library_root(root) / _binary_name("ffmpeg")
    try:
        if candidate.is_file():
            return candidate
    except OSError:
        return None
    return None


def _probe_repo_without_quarantine(root: Path) -> Path | None:
    """Dossier de PATH qui imite la racine après la quarantaine.

    ``ffprobe`` à la racine n'est pas déplacé : le dossier de probe le
    republie. ``ffmpeg`` n'y est pas. Le lien est absolu pour rester valable
    quel que soit le cwd.
    """
    if _repo_ffmpeg_file(root) is None:
        return None
    library = _ffmpeg_library_root(root)
    shadow = Path(tempfile.mkdtemp(prefix="wan2gp-quarantine-"))
    probe = library / _binary_name("ffprobe")
    try:
        probe.lstat()
    except OSError:
        return shadow
    try:
        (shadow / probe.name).symlink_to(probe.absolute())
    except OSError:
        return shadow
    return shadow


def _activate_probe_repo(root: Path | None) -> tuple[Path | None, contextvars.Token[str | None] | None]:
    """Installe le PATH de probe, ou ``(None, None)`` s'il n'y a rien à cacher."""
    if root is None or os.name == "nt":
        return None, None
    shadow = _probe_repo_without_quarantine(root)
    if shadow is None:
        return None, None
    return shadow, _PROBE_REPO_OVERRIDE.set(os.path.normpath(str(shadow)))


def _release_probe_repo(
    shadow: Path | None,
    token: contextvars.Token[str | None] | None,
) -> None:
    if token is not None:
        _PROBE_REPO_OVERRIDE.reset(token)
    if shadow is not None:
        shutil.rmtree(shadow, ignore_errors=True)


def _path_reexec_message(binary: Path, root: Path | None) -> str:
    if root is None:
        where = "ffmpeg_bins puis la racine du clone"
        quarantine = ""
    else:
        bundle, repo = _wan_directories(root)
        where = f"{bundle} puis {repo}"
        hidden = _repo_ffmpeg_file(root)
        if hidden is None:
            quarantine = ""
        else:
            quarantine = (
                f"download_ffmpeg déplace d'abord {hidden.absolute()} vers ffmpeg_quarantine, "
                "puis met cette racine sur le PATH. "
                "Tant que le fichier est là, exec ffmpeg le trouve et -encoders annonce libx264. "
                "Une entrée dont le texte n'est pas cette racine peut quand même l'ouvrir : "
                "le noyau entre dans un lien de dossier avant d'appliquer .., un chemin "
                "relatif (.) est ouvert depuis le cwd du clone, et /proc/self/cwd "
                "comme /proc/thread-self/cwd suivent ce cwd plutôt que celui du contrôle. "
                "normpath annule .. avant ces liens : /proc/self/../self/cwd redevient "
                "/proc/self/cwd dans le texte, /proc/thread-self/../../cwd devient /cwd, "
                "et /proc/self/root/proc/self/cwd ne change pas (/proc/self/root est /). "
                "Le noyau entre dans le lien avant d'appliquer '..'. Un dossier qui est "
                "un lien vers /proc/self, suivi de cwd, ouvre la même racine. "
                "/proc/sys est un dossier réel, pas un lien vers self : "
                "/proc/sys/../self/cwd et /proc/sys/kernel/../../self/cwd n'ont pas "
                "self immédiatement après proc. normpath les réécrit pourtant en "
                "/proc/self/cwd. Le noyau entre dans sys avant '..'. Un lien vers "
                "/proc/sys suivi de ../self/cwd, ou un lien dont la cible est déjà "
                "/proc/sys/../self/cwd, ouvre la même racine. "
                "/proc/sys/../thread-self/cwd aussi. "
                "Deux liens de dossier dans le même répertoire cassent une boucle "
                "basée sur normpath. hop1 pointe vers hop2, hop2 vers /proc/net, "
                "et l'entrée est hop1/../cwd : ni proc ni self dans le texte. "
                "normpath annule .. avant le lien, donc hop1/../cwd et hop2/../cwd "
                "deviennent le même chemin (le dossier parent plus cwd). "
                "S'arrêter là ne lit pas hop2. Le noyau entre dans hop1, puis dans "
                "hop2 (/proc/net), puis .. retombe dans /proc/self et cwd ouvre "
                "la racine du clone. Un lien relatif (rel1 vers rel2) fait pareil. "
                "hop1 vers hop2 vers /proc/sys, avec l'entrée hop1/../self/cwd, aussi : "
                "self est écrit, mais proc n'apparaît qu'au second lien. "
                "Une chaîne de trois liens jusqu'à /proc/net aussi. "
                "hop1/../cwd quand hop2 pointe vers /proc/sys n'ouvre pas la racine : "
                "/proc/cwd n'existe pas. "
                "Le probe retire cette entrée. "
                "Après la quarantaine, ce nom n'existe plus : le script se retrouve lui-même. "
            )
    return (
        f"{binary} ne revient pas quand le PATH commence par {where}. "
        "download_ffmpeg place ces dossiers devant le PATH avant le mux. "
        + quarantine
        + "Un script qui relance ffmpeg ou ffprobe sans chemin (exec ffmpeg) "
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
    masked: set[str] = set()
    for path in _ld_search_entries(bundle, ffmpeg, root):
        subject = _loader_subject(path, root)
        if subject is not None and _is_regular_file(subject):
            masked.add(path.name)
    return masked


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
    entries = _ld_search_entries(seen, ffmpeg, root)
    needed = (
        _needed_sonames(ffmpeg, seen)
        if any(_subject_is_nonfile_library(path, root) for path in entries)
        else set()
    )
    traps = [
        trap
        for trap in _library_traps(entries, needed, seen, root)
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
            "Ce comportement est celui de Fun_InP_1.3B en I2V, d'un image_mode différent de 0, "
            "ou d'une définition defaults/finetunes dont image_outputs est vrai."
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
