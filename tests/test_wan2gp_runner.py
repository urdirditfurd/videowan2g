"""Réglages Wan2GP : grille d'images, VRAM, seed, sans GPU."""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import main
from src.config_loader import load_pipeline_config
from src.errors import Wan2GPError
from src.pipeline import RunOptions, run
from src.prompt_builder import build_prompt, load_catalog
import src.wan2gp_runner as wan2gp_runner
from src.wan2gp_runner import (
    GenerationRequest,
    generate,
    parse_resolution,
    resolve_settings_version,
    snap_frame_count,
)

ROOT = Path(__file__).resolve().parents[1]

# Même tête que wgp.py : abspath ne résout pas le lien, realpath est déjà dans sys.path.
_WGP_IMPORT_PROBE = """\
import os
import sys
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)
import shared.ffmpeg_setup as mod
print(mod.__file__)
"""


class FrameAndResolutionTest(unittest.TestCase):
    def test_frame_count_follows_the_wan_grid(self) -> None:
        self.assertEqual(snap_frame_count(81), 81)
        self.assertEqual(snap_frame_count(80), 81)
        self.assertEqual(snap_frame_count(1), 5)
        self.assertEqual(snap_frame_count(6), 9)

    def test_resolution_stays_on_the_vae_grid(self) -> None:
        self.assertEqual(parse_resolution("480x832"), "480x832")
        with self.assertRaises(Wan2GPError):
            parse_resolution("480x830")
        with self.assertRaises(Wan2GPError):
            parse_resolution("100x100")

    def test_pipeline_yaml_targets_the_light_model(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        self.assertEqual(config.wan.model_type, "t2v_1.3B")
        self.assertEqual(config.wan.profile, 4)
        self.assertEqual(config.wan.attention, "sdpa")
        self.assertEqual(config.wan.frames, 81)
        self.assertEqual(config.wan.fps, 16)
        self.assertEqual(config.wan.resolutions["9:16"], "480x832")
        self.assertEqual(config.wan.resolutions["1:1"], "624x624")
        self.assertIsInstance(config.wan.resolutions["9:16"], str)


class DryRunTest(unittest.TestCase):
    def test_dry_run_writes_settings_and_does_not_spawn(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            request = GenerationRequest(
                prompt=build_prompt(theme, 42),
                settings=config.wan,
                output_dir=output,
                aspect="1:1",
                frames=80,
                dry_run=True,
            )
            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=AssertionError("GPU")):
                result = generate(request)
            payload = json.loads(result.settings_path.read_text(encoding="utf-8"))
            self.assertTrue(result.dry_run)
            self.assertIsNone(result.raw_video)
            self.assertEqual(payload["model_type"], "t2v_1.3B")
            self.assertEqual(payload["base_model_type"], "t2v_1.3B")
            self.assertEqual(payload["image_mode"], 0)
            self.assertEqual(payload["settings_version"], 2.81)
            self.assertEqual(payload["multi_prompts_gen_type"], "FG")
            self.assertEqual(payload["video_length"], 81)
            self.assertEqual(payload["resolution"], "624x624")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["force_fps"], "16")
            self.assertEqual(payload["overrides"], "memory_profile=4, attention_mode='sdpa'")
            self.assertNotIn("override_profile", payload)
            self.assertNotIn("image_start", payload)
            self.assertIn("capybara", str(payload["prompt"]))
            process_at = result.command.index("--process")
            self.assertTrue(str(result.command[process_at + 1]).endswith(".queue.zip"))
            self.assertEqual(result.command[process_at + 2], "--output-dir")
            self.assertEqual(result.command[result.command.index("--verbose") + 1], "2")
            self.assertIn("--profile", result.command)
            self.assertIn("4", result.command)
            self.assertIn("--attention", result.command)
            self.assertIn("sdpa", result.command)
            self.assertNotIn("--i2v-1-3B", result.command)
            self.assertNotIn("--t2v-1-3B", result.command)
            with zipfile.ZipFile(result.command[process_at + 1]) as archive:
                manifest = json.loads(archive.read("queue.json"))
            self.assertEqual(manifest[0]["params"]["model_type"], "t2v_1.3B")
            self.assertEqual(manifest[0]["params"]["image_mode"], 0)
            self.assertTrue(Path(result.command[process_at + 1]).is_absolute())
            self.assertTrue(Path(result.command[result.command.index("--output-dir") + 1]).is_absolute())

    def test_relative_output_is_written_outside_the_wan_clone(self) -> None:
        """Le cwd du processus est le clone : un --output-dir relatif manque le dossier suivi."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg est requis")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            root = Path(directory) / "wan"
            work.mkdir()
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def fake_wan(command: list[str], **kwargs: object) -> _FakeProcess:
                cwd = Path(str(kwargs["cwd"]))
                raw = Path(command[command.index("--output-dir") + 1])
                if not raw.is_absolute():
                    raw = cwd / raw
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            os.chdir(work)
            try:
                with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                    result = generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 23),
                            settings=replace(config.wan, root=root),
                            output_dir=Path("renders"),
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                os.chdir(previous)
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.is_absolute())
            self.assertEqual(result.raw_video.parent, (work / "renders" / "raw" / result.stem).resolve())
            self.assertFalse(str(result.raw_video).startswith(str(root)))

    def test_relative_clone_receives_an_absolute_ffmpeg(self) -> None:
        """imageio lance IMAGEIO_FFMPEG_EXE sans PATH. Depuis le cwd du clone, un chemin relatif manque."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            os.chdir(parent)
            try:
                root = Path("wan")
                bins = root / "ffmpeg_bins"
                bins.mkdir(parents=True)
                (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                ffmpeg_link = bins / "ffmpeg"
                ffprobe_link = bins / "ffprobe"
                ffmpeg_link.symlink_to(system_ffmpeg)
                ffprobe_link.symlink_to(system_ffprobe)
                (bins / "libavdevice.so.61").write_bytes(b"")
                expected_ffmpeg = ffmpeg_link.absolute()
                expected_ffprobe = ffprobe_link.absolute()
                clone = root.absolute()
                seen: dict[str, str] = {}
                previous_env = {
                    key: os.environ.get(key)
                    for key in ("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                }

                def fake_wan(command: list[str], **kwargs: object) -> _FakeProcess:
                    env = kwargs.get("env")
                    assert isinstance(env, dict)
                    seen["ffmpeg"] = str(env["FFMPEG_BINARY"])
                    seen["ffprobe"] = str(env["FFPROBE_BINARY"])
                    seen["imageio"] = str(env["IMAGEIO_FFMPEG_EXE"])
                    seen["cwd"] = str(kwargs["cwd"])
                    seen["path"] = str(env["PATH"])
                    seen["bundle"] = os.path.normpath(str(bins.absolute()))
                    seen["clone"] = os.path.normpath(str(clone))
                    raw = Path(command[command.index("--output-dir") + 1])
                    raw.mkdir(parents=True, exist_ok=True)
                    _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                    return _FakeProcess()

                try:
                    with mock.patch(
                        "src.wan2gp_runner.subprocess.Popen",
                        side_effect=_popen_except_ffmpeg(fake_wan),
                    ):
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 29),
                                settings=replace(config.wan, root=root),
                                output_dir=parent / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
                finally:
                    for key, value in previous_env.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                try:
                    subprocess.run(
                        [str(ffmpeg_link), "-version"],
                        cwd=clone,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError:
                    relative_missing = True
                else:
                    relative_missing = False
                launched = subprocess.run(
                    [seen["imageio"], "-version"],
                    cwd=clone,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertTrue(expected_ffmpeg.is_symlink())
                self.assertNotEqual(expected_ffmpeg.resolve(), expected_ffmpeg)
            finally:
                os.chdir(previous)
        self.assertTrue(relative_missing)
        self.assertEqual(launched.returncode, 0)
        self.assertTrue(Path(seen["cwd"]).is_absolute())
        self.assertEqual(Path(seen["cwd"]), clone)
        for key in ("ffmpeg", "ffprobe", "imageio"):
            self.assertTrue(Path(seen[key]).is_absolute(), seen[key])
        self.assertEqual(Path(seen["ffmpeg"]), expected_ffmpeg)
        self.assertEqual(Path(seen["ffprobe"]), expected_ffprobe)
        self.assertEqual(Path(seen["imageio"]), expected_ffmpeg)
        self.assertNotEqual(Path(seen["ffmpeg"]), Path(system_ffmpeg))
        path_entries = seen["path"].split(os.pathsep)
        self.assertEqual(path_entries[0], seen["bundle"])
        self.assertEqual(path_entries[1], seen["clone"])
        self.assertNotEqual(path_entries[0], str(Path(system_ffmpeg).resolve().parent))

    def test_relative_bundle_short_library_stops_before_the_gpu(self) -> None:
        """Un WAN2GP_ROOT relatif préfixe quand même le ffmpeg_bins absolu."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = Path("wan")
            bins = root / "ffmpeg_bins"
            os.chdir(base)
            try:
                bins.mkdir(parents=True)
                (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                (bins / "ffmpeg").symlink_to(system_ffmpeg)
                (bins / "ffprobe").symlink_to(system_ffprobe)
                (bins / "libavdevice.so.61").write_bytes(b"decoy")
                (bins / soname).write_bytes(b"short")
                absolute_bins = bins.absolute()
                missed = os.environ.copy()
                missed["LD_LIBRARY_PATH"] = "wan/ffmpeg_bins"
                from_clone = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=missed,
                    cwd=root.absolute(),
                )
                self.assertEqual(from_clone.returncode, 0)
                trapped = os.environ.copy()
                trapped["LD_LIBRARY_PATH"] = str(absolute_bins)
                broken = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=trapped,
                    cwd=root.absolute(),
                )
                self.assertNotEqual(broken.returncode, 0)
                self.assertIn(soname, broken.stderr)

                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 71),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                os.chdir(previous)
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn(str(absolute_bins), message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_reexec_wrapper_stops_before_the_gpu(self) -> None:
        """``exec ffmpeg`` passe -encoders, puis boucle avec le PATH de WanGP."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            script = bins / "ffmpeg"
            script.write_text("#!/bin/sh\nexec ffmpeg \"$@\"\n", encoding="utf-8")
            script.chmod(0o755)
            # Mauvais soname : le glob libavdevice.so* est vrai, le chargeur l'ignore.
            (bins / "libavdevice.so.61").write_bytes(b"decoy")
            blind = subprocess.run(
                [str(script), "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(blind.returncode, 0)
            self.assertIn("libx264", blind.stdout)
            trapped = os.environ.copy()
            trapped["PATH"] = os.pathsep.join(
                [
                    os.path.normpath(str(bins.absolute())),
                    os.path.normpath(str(root.absolute())),
                    trapped.get("PATH", ""),
                ]
            )
            with self.assertRaises(subprocess.TimeoutExpired):
                subprocess.run(
                    [str(script), "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=trapped,
                    timeout=1,
                )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 43),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("PATH", message)
            self.assertIn("ffmpeg_bins", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_absolute_ffmpeg_wrapper_still_muxes(self) -> None:
        """Un script qui appelle ``/usr/bin/ffmpeg`` reste un encodeur utilisable."""
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        previous_env = {
            key: os.environ.get(key)
            for key in ("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            script = bins / "ffmpeg"
            script.write_text(
                f"#!/bin/sh\nexec {system_ffmpeg} \"$@\"\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
            (bins / "libavdevice.so.61").write_bytes(b"decoy")
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual(Path(binaries.ffmpeg), script.absolute())
            self.assertNotEqual(Path(binaries.ffmpeg), Path(system_ffmpeg))

    def test_symlinked_ffmpeg_setup_library_stops_before_the_gpu(self) -> None:
        """Le préfixe LD vient du module résolu, pas du ffmpeg_bins du clone."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            local = base / "local"
            real = base / "real"
            bins = local / "ffmpeg_bins"
            bins.mkdir(parents=True)
            _link_ffmpeg_setup(local, real)
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"decoy")
            real_bins = real / "ffmpeg_bins"
            real_bins.mkdir()
            (real_bins / soname).write_bytes(b"short")
            plain = subprocess.run(
                [str(bins / "ffmpeg"), "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(plain.returncode, 0)
            self.assertIn("libx264", plain.stdout)
            trapped = os.environ.copy()
            trapped["LD_LIBRARY_PATH"] = str(real_bins)
            broken = subprocess.run(
                [str(bins / "ffmpeg"), "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                env=trapped,
            )
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn(soname, broken.stderr)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 47),
                            settings=replace(config.wan, root=local),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("ffmpeg_setup", message)
            self.assertIn(str(real), message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_symlinked_wgp_library_stops_before_the_gpu(self) -> None:
        """Un wgp.py lien charge ffmpeg_setup dans le clone réel."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["night_train"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            local = base / "local"
            real = base / "real"
            local.mkdir()
            (real / "shared").mkdir(parents=True)
            (real / "shared" / "__init__.py").write_text("", encoding="utf-8")
            (real / "shared" / "ffmpeg_setup.py").write_text("# module réel\n", encoding="utf-8")
            (real / "wgp.py").write_text("# wan réel\n", encoding="utf-8")
            real_bins = real / "ffmpeg_bins"
            real_bins.mkdir()
            (real_bins / soname).write_bytes(b"short")
            (local / "wgp.py").symlink_to(real / "wgp.py")
            plain = subprocess.run(
                [system_ffmpeg, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(plain.returncode, 0)
            trapped = os.environ.copy()
            trapped["LD_LIBRARY_PATH"] = str(real_bins)
            broken = subprocess.run(
                [system_ffmpeg, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                env=trapped,
            )
            self.assertNotEqual(broken.returncode, 0)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 53),
                            settings=replace(config.wan, root=local),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn("ffmpeg_setup", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_proc_cwd_ffmpeg_setup_library_stops_before_the_gpu(self) -> None:
        """``Path(__file__).resolve()`` de WanGP suit le cwd du clone.

        ``shared`` ou ``ffmpeg_setup.py`` vise ``/proc/self/cwd/nested/shared``,
        ``/proc/net/../cwd/nested/shared``, ou ``hop1/../cwd/nested/shared``
        (``hop1`` → ``hop2`` → ``/proc/net``). Depuis le pipeline,
        ``Path.resolve`` ouvre ce cwd, ou ne voit pas le fichier.
        ``download_ffmpeg`` préfixe ``<clone>/nested/ffmpeg_bins``. Une
        ``libx264`` trop courte seulement là sort 127 (``file too short``).
        ``-encoders`` sans ce préfixe annonce encore libx264.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = ("self-missing", "proc-net-decoy", "hop-decoy", "file-symlink")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    (root / "ffmpeg").symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    nested = root / "nested"
                    module = nested / "shared"
                    module.mkdir(parents=True)
                    (module / "ffmpeg_setup.py").write_text("# module du clone\n", encoding="utf-8")
                    hidden = nested / "ffmpeg_bins"
                    hidden.mkdir()
                    (hidden / soname).write_bytes(b"short")
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    planted = decoy / "ffmpeg"
                    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                    planted.chmod(0o755)
                    if shape == "hop-decoy":
                        (root / "hop2").symlink_to("/proc/net")
                        (root / "hop1").symlink_to("hop2")
                        (root / "shared").symlink_to("hop1/../cwd/nested/shared")
                    elif shape == "file-symlink":
                        (root / "shared").mkdir()
                        (root / "shared" / "ffmpeg_setup.py").symlink_to(
                            "/proc/self/cwd/nested/shared/ffmpeg_setup.py"
                        )
                    elif shape == "self-missing":
                        (root / "shared").symlink_to("/proc/self/cwd/nested/shared")
                    elif shape == "proc-net-decoy":
                        (root / "shared").symlink_to("/proc/net/../cwd/nested/shared")
                    else:
                        self.fail(shape)
                    if shape in {"proc-net-decoy", "hop-decoy"}:
                        mirror = decoy / "nested" / "shared"
                        mirror.mkdir(parents=True)
                        (mirror / "ffmpeg_setup.py").write_text("# leurre\n", encoding="utf-8")
                        (decoy / "nested" / "ffmpeg_bins").mkdir()
                    try:
                        os.chdir(decoy)
                        setup = root / "shared" / "ffmpeg_setup.py"
                        if shape in {"self-missing", "file-symlink"}:
                            self.assertFalse(setup.is_file())
                        else:
                            self.assertEqual(setup.resolve().parents[1], decoy / "nested")
                            self.assertNotEqual(setup.resolve().parents[1], nested)
                        plain = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(plain.returncode, 0, plain.stderr)
                        self.assertIn("libx264", plain.stdout)
                        env = os.environ.copy()
                        prefix = str(hidden.absolute())
                        inherited = env.get("LD_LIBRARY_PATH", "")
                        env["LD_LIBRARY_PATH"] = (
                            prefix if not inherited else prefix + os.pathsep + inherited
                        )
                        shadowed = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            env=env,
                            timeout=5,
                        )
                        self.assertEqual(shadowed.returncode, 127, shadowed.stderr)
                        self.assertIn("file too short", shadowed.stderr)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 31),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn(soname, message)
                        self.assertIn("ffmpeg_setup", message)
                        self.assertIn(str(hidden), message)
                        self.assertIn("/proc/self", message)
                        self.assertIn("LD_LIBRARY_PATH", message)
                        self.assertIn("file too short", message)
                        self.assertIn("error saving", message)
                        self.assertIn("sans MP4", message)
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_cwd_ffmpeg_setup_ignores_a_pipeline_library(self) -> None:
        """Un soname du cwd du pipeline n'est pas le ffmpeg_bins de WanGP.

        Le leurre rend ``Path.resolve`` vrai ici. Le clone n'a pas ce
        soname. Le mux du binaire système reste possible.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "ffmpeg").symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            nested = root / "nested"
            module = nested / "shared"
            module.mkdir(parents=True)
            (module / "ffmpeg_setup.py").write_text("# module du clone\n", encoding="utf-8")
            hidden = nested / "ffmpeg_bins"
            hidden.mkdir()
            (root / "shared").symlink_to("/proc/self/cwd/nested/shared")
            decoy = base / "pipeline"
            decoy.mkdir()
            mirror = decoy / "nested" / "shared"
            mirror.mkdir(parents=True)
            (mirror / "ffmpeg_setup.py").write_text("# leurre\n", encoding="utf-8")
            planted_bins = decoy / "nested" / "ffmpeg_bins"
            planted_bins.mkdir()
            (planted_bins / soname).write_bytes(b"short")
            try:
                os.chdir(decoy)
                setup = root / "shared" / "ffmpeg_setup.py"
                self.assertEqual(setup.resolve().parents[1], decoy / "nested")
                env = os.environ.copy()
                inherited = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    str(planted_bins)
                    if not inherited
                    else str(planted_bins) + os.pathsep + inherited
                )
                decoy_load = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    env=env,
                    timeout=5,
                )
                self.assertEqual(decoy_load.returncode, 127, decoy_load.stderr)
                clean = os.environ.copy()
                clean_prefix = str(hidden.absolute())
                inherited = clean.get("LD_LIBRARY_PATH", "")
                clean["LD_LIBRARY_PATH"] = (
                    clean_prefix if not inherited else clean_prefix + os.pathsep + inherited
                )
                loaded = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    env=clean,
                    timeout=5,
                )
                self.assertEqual(loaded.returncode, 0, loaded.stderr)
                self.assertIn("libx264", loaded.stdout)
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)
                _restore_env(saved)

    def test_proc_cwd_wgp_script_library_stops_before_the_gpu(self) -> None:
        """``sys.path[0]`` de WanGP est le realpath du script, cwd = clone.

        ``wgp.py`` vise ``/proc/self/cwd/nested/wgp.py``,
        ``/proc/net/../cwd/nested/wgp.py`` ou ``hop1/../cwd/nested/wgp.py``
        (``hop1`` → ``hop2`` → ``/proc/net``). ``os.path.abspath`` ne résout
        pas le lien. Sans ``shared`` dans ce dossier, l'import charge
        ``<clone>/nested/shared``. Depuis le pipeline, ``Path.resolve`` ouvre
        ce cwd : un leurre rend ``is_file`` vrai, et sans leurre il est faux.
        ``download_ffmpeg`` préfixe ``<clone>/nested/ffmpeg_bins``. Une
        ``libx264`` trop courte seulement là sort 127 (``file too short``).
        ``-encoders`` sans ce préfixe annonce encore libx264.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = ("self-missing", "proc-net-decoy", "hop-decoy")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    nested = root / "nested"
                    module = nested / "shared"
                    module.mkdir(parents=True)
                    (module / "__init__.py").write_text("", encoding="utf-8")
                    (module / "ffmpeg_setup.py").write_text("# module du clone\n", encoding="utf-8")
                    (nested / "wgp.py").write_text(_WGP_IMPORT_PROBE, encoding="utf-8")
                    hidden = nested / "ffmpeg_bins"
                    hidden.mkdir()
                    (hidden / soname).write_bytes(b"short")
                    if shape == "hop-decoy":
                        (root / "hop2").symlink_to("/proc/net")
                        (root / "hop1").symlink_to("hop2")
                        (root / "wgp.py").symlink_to("hop1/../cwd/nested/wgp.py")
                    elif shape == "proc-net-decoy":
                        (root / "wgp.py").symlink_to("/proc/net/../cwd/nested/wgp.py")
                    elif shape == "self-missing":
                        (root / "wgp.py").symlink_to("/proc/self/cwd/nested/wgp.py")
                    else:
                        self.fail(shape)
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    if shape != "self-missing":
                        mirror = decoy / "nested"
                        (mirror / "shared").mkdir(parents=True)
                        (mirror / "wgp.py").write_text(
                            "settings_version = 9.99\n",
                            encoding="utf-8",
                        )
                        (mirror / "shared" / "ffmpeg_setup.py").write_text(
                            "# leurre\n",
                            encoding="utf-8",
                        )
                        (mirror / "ffmpeg_bins").mkdir()
                    try:
                        os.chdir(decoy)
                        script = root / "wgp.py"
                        self.assertEqual(script.resolve().parent, decoy / "nested")
                        self.assertNotEqual(script.resolve().parent, nested)
                        if shape == "self-missing":
                            self.assertFalse(script.is_file())
                        else:
                            self.assertTrue(script.is_file())
                            self.assertIn(
                                "9.99",
                                script.read_text(encoding="utf-8"),
                            )
                        loaded = subprocess.run(
                            [sys.executable, "wgp.py"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=5,
                        )
                        self.assertEqual(loaded.returncode, 0, loaded.stderr)
                        self.assertEqual(
                            Path(loaded.stdout.strip()),
                            (module / "ffmpeg_setup.py").resolve(),
                        )
                        plain = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(plain.returncode, 0, plain.stderr)
                        self.assertIn("libx264", plain.stdout)
                        env = os.environ.copy()
                        prefix = str(hidden.absolute())
                        inherited = env.get("LD_LIBRARY_PATH", "")
                        env["LD_LIBRARY_PATH"] = (
                            prefix if not inherited else prefix + os.pathsep + inherited
                        )
                        shadowed = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            env=env,
                            timeout=5,
                        )
                        self.assertEqual(shadowed.returncode, 127, shadowed.stderr)
                        self.assertIn("file too short", shadowed.stderr)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 61),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn(soname, message)
                        self.assertIn("wgp.py", message)
                        self.assertIn("/proc/self", message)
                        self.assertIn(str(hidden), message)
                        self.assertIn("LD_LIBRARY_PATH", message)
                        self.assertIn("file too short", message)
                        self.assertIn("error saving", message)
                        self.assertIn("code 0", message)
                        self.assertIn("sans MP4", message)
                        self.assertNotIn("introuvable", message)
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_cwd_wgp_script_ignores_a_pipeline_library(self) -> None:
        """Un soname du cwd du pipeline n'est pas le ffmpeg_bins du script.

        Le leurre rend ``Path.resolve`` vrai ici. Le ``nested/ffmpeg_bins``
        du clone n'a pas ce soname. Le mux du binaire système reste possible.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            nested = root / "nested"
            module = nested / "shared"
            module.mkdir(parents=True)
            (module / "__init__.py").write_text("", encoding="utf-8")
            (module / "ffmpeg_setup.py").write_text("# module du clone\n", encoding="utf-8")
            (nested / "wgp.py").write_text(_WGP_IMPORT_PROBE, encoding="utf-8")
            hidden = nested / "ffmpeg_bins"
            hidden.mkdir()
            (root / "wgp.py").symlink_to("/proc/self/cwd/nested/wgp.py")
            decoy = base / "pipeline"
            mirror = decoy / "nested"
            (mirror / "shared").mkdir(parents=True)
            (mirror / "wgp.py").write_text("# leurre\n", encoding="utf-8")
            (mirror / "shared" / "ffmpeg_setup.py").write_text("# leurre\n", encoding="utf-8")
            planted_bins = mirror / "ffmpeg_bins"
            planted_bins.mkdir()
            (planted_bins / soname).write_bytes(b"short")
            try:
                os.chdir(decoy)
                script = root / "wgp.py"
                self.assertTrue(script.is_file())
                self.assertEqual(script.resolve().parent, mirror)
                env = os.environ.copy()
                inherited = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    str(planted_bins)
                    if not inherited
                    else str(planted_bins) + os.pathsep + inherited
                )
                decoy_load = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    env=env,
                    timeout=5,
                )
                self.assertEqual(decoy_load.returncode, 127, decoy_load.stderr)
                clean = os.environ.copy()
                clean_prefix = str(hidden.absolute())
                inherited = clean.get("LD_LIBRARY_PATH", "")
                clean["LD_LIBRARY_PATH"] = (
                    clean_prefix if not inherited else clean_prefix + os.pathsep + inherited
                )
                loaded = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    env=clean,
                    timeout=5,
                )
                self.assertEqual(loaded.returncode, 0, loaded.stderr)
                self.assertIn("libx264", loaded.stdout)
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)
                _restore_env(saved)

    def test_local_ffmpeg_bins_are_not_the_wan_library_path(self) -> None:
        """Une bibliothèque à côté du binaire du clone n'est pas celle du module résolu."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        previous_env = {
            key: os.environ.get(key)
            for key in ("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        }
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            local = base / "local"
            real = base / "real"
            bins = local / "ffmpeg_bins"
            bins.mkdir(parents=True)
            _link_ffmpeg_setup(local, real)
            ffmpeg_link = bins / "ffmpeg"
            ffmpeg_link.symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"decoy")
            (bins / soname).write_bytes(b"short")
            real_bins = real / "ffmpeg_bins"
            real_bins.mkdir()
            beside = os.environ.copy()
            beside["LD_LIBRARY_PATH"] = str(bins)
            broken = subprocess.run(
                [str(ffmpeg_link), "-hide_banner", "-version"],
                check=False,
                capture_output=True,
                text=True,
                env=beside,
            )
            self.assertNotEqual(broken.returncode, 0)
            prefixed = os.environ.copy()
            prefixed["LD_LIBRARY_PATH"] = str(real_bins)
            healthy = subprocess.run(
                [str(ffmpeg_link), "-hide_banner", "-version"],
                check=False,
                capture_output=True,
                text=True,
                env=prefixed,
            )
            self.assertEqual(healthy.returncode, 0)
            try:
                binaries = wan2gp_runner.require_ffmpeg(local)
            finally:
                for key, value in previous_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual(Path(binaries.ffmpeg), ffmpeg_link.absolute())

    def test_libstdcxx_preload_shadow_stops_before_the_gpu(self) -> None:
        """CONDA_PREFIX/lib masque libx264 après le préchargement de libstdc++."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        saved = _saved_env("CONDA_PREFIX")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prefix = base / "conda"
            lib = prefix / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            (lib / soname).write_bytes(b"short")
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            plain = subprocess.run(
                [system_ffmpeg, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(plain.returncode, 0)
            self.assertIn("libx264", plain.stdout)
            trapped = os.environ.copy()
            trapped["LD_LIBRARY_PATH"] = str(lib)
            broken = subprocess.run(
                [system_ffmpeg, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                env=trapped,
            )
            self.assertNotEqual(broken.returncode, 0)
            self.assertIn(soname, broken.stderr)
            os.environ["CONDA_PREFIX"] = str(prefix)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 61),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn(str(lib), message)
            self.assertIn("preload_preferred_libstdcxx", message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_libstdcxx_preload_fifo_stops_without_opening_it(self) -> None:
        """Un tube du soname dans le dossier de libstdc++ ne doit pas bloquer le mux."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        saved = _saved_env("CONDA_PREFIX")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prefix = base / "conda"
            lib = prefix / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            os.mkfifo(lib / soname)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.environ["CONDA_PREFIX"] = str(prefix)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 62),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("preload_preferred_libstdcxx", message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_unusable_libstdcxx_does_not_prepend_ld_library_path(self) -> None:
        """Un libstdc++.so.6 illisible est ignoré : le soname voisin ne masque pas ffmpeg."""
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        saved = _saved_env(
            "CONDA_PREFIX",
            "FFMPEG_BINARY",
            "FFPROBE_BINARY",
            "IMAGEIO_FFMPEG_EXE",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prefix = base / "conda"
            lib = prefix / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").write_bytes(b"nope")
            (lib / soname).write_bytes(b"short")
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.environ["CONDA_PREFIX"] = str(prefix)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
            finally:
                _restore_env(saved)
            self.assertEqual(Path(binaries.ffmpeg), Path(system_ffmpeg))

    def test_loadable_libstdcxx_directory_still_muxes(self) -> None:
        """Le dossier qui ne contient que le vrai libstdc++ reste un préfixe sain."""
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        libstd = _real_libstdcxx()
        if libstd is None:
            self.skipTest("libstdc++ illisible")
        saved = _saved_env(
            "CONDA_PREFIX",
            "FFMPEG_BINARY",
            "FFPROBE_BINARY",
            "IMAGEIO_FFMPEG_EXE",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prefix = base / "conda"
            lib = prefix / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.environ["CONDA_PREFIX"] = str(prefix)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
            finally:
                _restore_env(saved)
            self.assertEqual(Path(binaries.ffmpeg), Path(system_ffmpeg))

    def test_ffmpeg_bins_masks_a_preload_fifo(self) -> None:
        """Un libx264 réel dans ffmpeg_bins est ouvert avant le tube du préchargement."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        resolved = _resolved_library(system_ffmpeg, soname)
        if resolved is None:
            self.skipTest("bibliothèque transitive introuvable")
        saved = _saved_env(
            "CONDA_PREFIX",
            "FFMPEG_BINARY",
            "FFPROBE_BINARY",
            "IMAGEIO_FFMPEG_EXE",
        )
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            prefix = base / "conda"
            lib = prefix / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            os.mkfifo(lib / soname)
            root = base / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"decoy")
            (bins / soname).symlink_to(resolved)
            os.environ["CONDA_PREFIX"] = str(prefix)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
            finally:
                _restore_env(saved)
            self.assertEqual(Path(binaries.ffmpeg), (bins / "ffmpeg").absolute())

    def test_relative_libstdcxx_prefix_is_opened_from_the_clone_cwd(self) -> None:
        """Un CONDA_PREFIX relatif est ouvert depuis le clone, pas depuis le pipeline."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        saved = _saved_env("CONDA_PREFIX")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lib = base / "relconda" / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            root = base / "wan"
            poisoned = root / "relconda" / "lib"
            poisoned.mkdir(parents=True)
            (poisoned / "libstdc++.so.6").symlink_to(libstd)
            (poisoned / soname).write_bytes(b"short")
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            try:
                os.chdir(base)
                os.environ["CONDA_PREFIX"] = "relconda"
                healthy = os.environ.copy()
                healthy["LD_LIBRARY_PATH"] = "relconda/lib"
                from_pipeline = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=healthy,
                    cwd=base,
                )
                self.assertEqual(from_pipeline.returncode, 0)
                self.assertIn("libx264", from_pipeline.stdout)
                from_clone = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=healthy,
                    cwd=root,
                )
                self.assertNotEqual(from_clone.returncode, 0)
                self.assertIn(soname, from_clone.stderr)
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 63),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                os.chdir(previous)
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn("relconda/lib", message)
            self.assertIn(str(root), message)
            self.assertIn("preload_preferred_libstdcxx", message)
            self.assertIn("cwd", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_relative_libstdcxx_prefix_without_clone_shadow_still_muxes(self) -> None:
        """Le préfixe relatif sain du pipeline ne masque pas le ffmpeg du clone."""
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        saved = _saved_env(
            "CONDA_PREFIX",
            "FFMPEG_BINARY",
            "FFPROBE_BINARY",
            "IMAGEIO_FFMPEG_EXE",
        )
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lib = base / "relconda" / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            (lib / soname).write_bytes(b"short")
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            try:
                os.chdir(base)
                os.environ["CONDA_PREFIX"] = "relconda"
                binaries = wan2gp_runner.require_ffmpeg(root)
            finally:
                os.chdir(previous)
                _restore_env(saved)
            self.assertEqual(Path(binaries.ffmpeg), Path(system_ffmpeg))

    def test_relative_libstdcxx_fifo_is_not_opened(self) -> None:
        """Un tube sous le clone, via un CONDA_PREFIX relatif, ne bloque pas le contrôle."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        saved = _saved_env("CONDA_PREFIX")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            lib = base / "relconda" / "lib"
            lib.mkdir(parents=True)
            (lib / "libstdc++.so.6").symlink_to(libstd)
            root = base / "wan"
            poisoned = root / "relconda" / "lib"
            poisoned.mkdir(parents=True)
            (poisoned / "libstdc++.so.6").symlink_to(libstd)
            os.mkfifo(poisoned / soname)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube d'un CONDA_PREFIX relatif a bloqué le contrôle")

            alarm = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                os.chdir(base)
                os.environ["CONDA_PREFIX"] = "relconda"
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 64),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, alarm)
                os.chdir(previous)
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("relconda/lib", message)
            self.assertIn(str(root), message)
            self.assertIn("error saving", message)
            self.assertNotIn("GPU", message)

    def test_clone_only_relative_libstdcxx_prefix_stops_before_the_gpu(self) -> None:
        """Un libstdc++ visible seulement depuis le clone est quand même préfixé."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        libstd = _real_libstdcxx()
        if soname is None or libstd is None:
            self.skipTest("soname transitif ou libstdc++ illisible")
        saved = _saved_env("CONDA_PREFIX")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            poisoned = root / "relconda" / "lib"
            poisoned.mkdir(parents=True)
            (poisoned / "libstdc++.so.6").symlink_to(libstd)
            (poisoned / soname).write_bytes(b"short")
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            try:
                os.chdir(base)
                os.environ["CONDA_PREFIX"] = "relconda"
                healthy = os.environ.copy()
                healthy["LD_LIBRARY_PATH"] = "relconda/lib"
                from_pipeline = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=healthy,
                    cwd=base,
                )
                self.assertEqual(from_pipeline.returncode, 0)
                self.assertIn("libx264", from_pipeline.stdout)
                from_clone = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=healthy,
                    cwd=root,
                )
                self.assertNotEqual(from_clone.returncode, 0)
                self.assertIn(soname, from_clone.stderr)
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 65),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                os.chdir(previous)
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn("relconda/lib", message)
            self.assertIn(str(root), message)
            self.assertIn("preload_preferred_libstdcxx", message)
            self.assertIn("cwd", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_relative_ld_library_path_is_opened_from_the_clone_cwd(self) -> None:
        """Une entrée relative de LD_LIBRARY_PATH est ouverte depuis le clone."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["night_train"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        saved = _saved_env("CONDA_PREFIX", "LD_LIBRARY_PATH")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            poisoned = root / "badlib"
            poisoned.mkdir(parents=True)
            (poisoned / soname).write_bytes(b"short")
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            try:
                os.chdir(base)
                os.environ.pop("CONDA_PREFIX", None)
                os.environ["LD_LIBRARY_PATH"] = "badlib"
                from_pipeline = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=base,
                )
                self.assertEqual(from_pipeline.returncode, 0)
                self.assertIn("libx264", from_pipeline.stdout)
                from_clone = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                )
                self.assertNotEqual(from_clone.returncode, 0)
                self.assertIn(soname, from_clone.stderr)
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 66),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                os.chdir(previous)
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn(soname, message)
            self.assertIn("badlib", message)
            self.assertIn(str(root), message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("cwd", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_relative_ld_library_path_fifo_is_not_opened(self) -> None:
        """Un tube dans une entrée relative de LD_LIBRARY_PATH ne bloque pas le contrôle."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        saved = _saved_env("CONDA_PREFIX", "LD_LIBRARY_PATH")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            poisoned = root / "badlib"
            poisoned.mkdir(parents=True)
            os.mkfifo(poisoned / soname)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube d'un LD_LIBRARY_PATH relatif a bloqué le contrôle")

            alarm = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                os.chdir(base)
                os.environ.pop("CONDA_PREFIX", None)
                os.environ["LD_LIBRARY_PATH"] = "badlib"
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 67),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, alarm)
                os.chdir(previous)
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("badlib", message)
            self.assertIn(str(root), message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("error saving", message)
            self.assertNotIn("GPU", message)

    def test_image_input_is_refused(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["night_train"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image = output / "still.png"
            image.write_bytes(b"png")
            with self.assertRaises(Wan2GPError) as caught:
                generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 7),
                        settings=config.wan,
                        output_dir=output,
                        aspect="9:16",
                        image_path=image,
                        dry_run=True,
                    )
                )
            self.assertIn("JPEG", str(caught.exception))

    def test_fun_inp_model_is_refused(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Wan2GPError):
                generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 3),
                        settings=replace(config.wan, model_type="fun_inp_1.3B"),
                        output_dir=Path(directory),
                        aspect="9:16",
                        dry_run=True,
                    )
                )

    def test_cli_image_flag_stops_before_generation(self) -> None:
        self.assertEqual(main(["--theme", "cozy_cafe", "--image", "still.png", "--dry-run"]), 2)

    def test_jpeg_instead_of_mp4_is_a_failure(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            output = Path(directory) / "out"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                (raw / "preview.jpg").write_bytes(b"\xff\xd8\xff\xd9")
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 9),
                            settings=replace(config.wan, root=root),
                            output_dir=output,
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            self.assertIn("JPEG", str(caught.exception).upper())

    def test_short_or_image_codec_mp4_is_a_failure(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            output = Path(directory) / "out"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "still.mp4", seconds=0.04, frames=1)
                _encode_mjpeg(raw / "photo.mp4")
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 10),
                            settings=replace(config.wan, root=root),
                            output_dir=output,
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("trop court", message)
            self.assertIn("mjpeg", message)

    def test_playable_mp4_is_accepted_and_ffmpeg_is_passed(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            output = Path(directory) / "out"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            seen: dict[str, str] = {}

            def fake_wan(command: list[str], **kwargs: object) -> _FakeProcess:
                env = kwargs.get("env")
                self.assertIsInstance(env, dict)
                assert isinstance(env, dict)
                seen["ffmpeg"] = str(env["FFMPEG_BINARY"])
                seen["ffprobe"] = str(env["FFPROBE_BINARY"])
                seen["imageio"] = str(env["IMAGEIO_FFMPEG_EXE"])
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 11),
                        settings=replace(config.wan, root=root),
                        output_dir=output,
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertEqual(result.raw_video.suffix, ".mp4")
            self.assertTrue(seen["ffmpeg"].endswith("ffmpeg"))
            self.assertTrue(seen["ffprobe"].endswith("ffprobe"))
            self.assertEqual(seen["imageio"], seen["ffmpeg"])

    def test_missing_ffmpeg_stops_before_the_process(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            with mock.patch("src.wan2gp_runner.shutil.which", return_value=None):
                with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=AssertionError("GPU")):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 12),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            self.assertIn("ffmpeg", str(caught.exception))

    def test_installed_settings_version_is_copied_into_the_queue(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# wan\nsettings_version = 2.81\n", encoding="utf-8")
            result = generate(
                GenerationRequest(
                    prompt=build_prompt(theme, 13),
                    settings=replace(config.wan, root=root),
                    output_dir=Path(directory) / "out",
                    aspect="9:16",
                    dry_run=True,
                )
            )
            payload = json.loads(result.settings_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["settings_version"], 2.81)
            self.assertEqual(payload["overrides"], "memory_profile=4, attention_mode='sdpa'")
            self.assertEqual(resolve_settings_version(None), 2.81)

    def test_settings_version_past_the_first_8kb_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wgp.py").write_text("# " + ("pad " * 3000) + "\nsettings_version = 2.79\n", encoding="utf-8")
            self.assertGreater(len((root / "wgp.py").read_text(encoding="utf-8")), 8000)
            self.assertEqual(resolve_settings_version(root), 2.79)

    def test_legacy_settings_version_keeps_override_profile(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.79\n", encoding="utf-8")
            result = generate(
                GenerationRequest(
                    prompt=build_prompt(theme, 16),
                    settings=replace(config.wan, root=root),
                    output_dir=Path(directory) / "out",
                    aspect="9:16",
                    dry_run=True,
                )
            )
            payload = json.loads(result.settings_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["settings_version"], 2.79)
            self.assertEqual(payload["override_profile"], 4)
            self.assertNotIn("overrides", payload)

    def test_wan_ffmpeg_bins_are_used_when_present(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            output = Path(directory) / "out"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            ffmpeg_link = bins / "ffmpeg"
            ffprobe_link = bins / "ffprobe"
            ffmpeg_link.symlink_to(system_ffmpeg)
            ffprobe_link.symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            seen: dict[str, str] = {}
            previous = {
                key: os.environ.get(key)
                for key in ("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
            }

            def fake_wan(command: list[str], **kwargs: object) -> _FakeProcess:
                env = kwargs.get("env")
                assert isinstance(env, dict)
                seen["ffmpeg"] = str(env["FFMPEG_BINARY"])
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            try:
                with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 14),
                            settings=replace(config.wan, root=root),
                            output_dir=output,
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual(Path(seen["ffmpeg"]).resolve(), Path(system_ffmpeg).resolve())
            self.assertEqual(seen["ffmpeg"], str(ffmpeg_link))

    def test_ffmpeg_bins_without_shared_libs_stop_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 17),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            self.assertIn("libavdevice", str(caught.exception))

    def test_root_ffmpeg_is_ignored_because_wan_quarantines_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "ffmpeg").write_text("pas un encodeur\n", encoding="utf-8")
            root_ffprobe = root / "ffprobe"
            root_ffprobe.symlink_to(system_ffprobe)
            seen: dict[str, str] = {}
            previous = {
                key: os.environ.get(key)
                for key in ("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
            }

            def fake_wan(command: list[str], **kwargs: object) -> _FakeProcess:
                env = kwargs.get("env")
                assert isinstance(env, dict)
                seen["ffmpeg"] = str(env["FFMPEG_BINARY"])
                seen["ffprobe"] = str(env["FFPROBE_BINARY"])
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            try:
                with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 19),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            self.assertEqual(Path(seen["ffmpeg"]).resolve(), Path(system_ffmpeg).resolve())
            self.assertNotEqual(Path(seen["ffmpeg"]), root / "ffmpeg")
            self.assertEqual(seen["ffprobe"], str(root_ffprobe))

    def test_quarantined_root_ffmpeg_no_longer_masks_a_reexec(self) -> None:
        """Le ffmpeg de la racine masque exec ffmpeg jusqu'à la quarantaine."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["night_train"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            bins = root / "ffmpeg_bins"
            shims = base / "shims"
            bins.mkdir(parents=True)
            shims.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            script = shims / "ffmpeg"
            script.write_text('#!/bin/sh\nexec ffmpeg "$@"\n', encoding="utf-8")
            script.chmod(0o755)
            (shims / "ffprobe").symlink_to(system_ffprobe)
            try:
                os.environ["PATH"] = os.pathsep.join([str(shims), os.environ.get("PATH", "")])
                masked = os.environ.copy()
                masked["PATH"] = os.pathsep.join(
                    [
                        os.path.normpath(str(bins.absolute())),
                        os.path.normpath(str(root.absolute())),
                        masked["PATH"],
                    ]
                )
                present = subprocess.run(
                    [str(script), "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=masked,
                    cwd=root,
                    timeout=3,
                )
                self.assertEqual(present.returncode, 0)
                self.assertIn("libx264", present.stdout)
                parked = root / "ffmpeg.parked"
                root_ffmpeg.rename(parked)
                try:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        subprocess.run(
                            [str(script), "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            env=masked,
                            cwd=root,
                            timeout=1,
                        )
                finally:
                    parked.rename(root_ffmpeg)

                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 73),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn("ffmpeg_quarantine", message)
            self.assertIn("PATH", message)
            self.assertIn(str(root_ffmpeg), message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_root_ffmpeg_on_path_is_not_the_imageio_binary(self) -> None:
        """Le ffmpeg de la racine, premier sur le PATH, part en quarantaine."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            try:
                os.environ["PATH"] = os.pathsep.join(
                    [str(root.absolute()), os.environ.get("PATH", "")]
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertNotEqual(Path(binaries.ffmpeg), root_ffmpeg.absolute())
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_bundle_symlink_into_quarantine_is_not_the_imageio_binary(self) -> None:
        """Un lien ffmpeg_bins → ffmpeg de la racine casse après la quarantaine."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (bins / "step").symlink_to(Path("..") / "ffmpeg")
            bundled = bins / "ffmpeg"
            bundled.symlink_to("step")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertNotEqual(Path(binaries.ffmpeg), bundled.absolute())
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
                probe = subprocess.run(
                    [binaries.ffprobe, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(probe.returncode, 0)
            finally:
                _restore_env(saved)

    def test_bundle_symlink_with_libs_stops_before_the_gpu(self) -> None:
        """Le lien vers la racine passe le mux, puis imageio ne trouve plus le binaire."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(Path("..") / "ffmpeg")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 41),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn(str(bundled), message)
            self.assertIn(str(root_ffmpeg), message)
            self.assertIn("ffmpeg_quarantine", message)
            self.assertIn("IMAGEIO_FFMPEG_EXE", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_path_alias_symlink_into_quarantine_is_not_the_imageio_binary(self) -> None:
        """``//<racine>/ffmpeg`` et ``../self/ffmpeg`` cassent après la quarantaine.

        ``normpath`` garde le double slash initial. ``self`` → ``.`` n'apparaît
        pas dans la chaîne du chemin. ``is_file`` suit quand même le lien tant
        que le fichier de la racine est là : ``-encoders`` et le mux réussissent.
        imageio exécute ensuite ce lien, qui n'existe plus.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        aliases = ("double-slash", "self-directory")
        for alias in aliases:
            with self.subTest(alias=alias):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "wan"
                    bins = root / "ffmpeg_bins"
                    bins.mkdir(parents=True)
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    if alias == "self-directory":
                        (root / "self").symlink_to(".", target_is_directory=True)
                        target: Path | str = Path("..") / "self" / "ffmpeg"
                    else:
                        target = "/" + str(root_ffmpeg)
                    bundled = bins / "ffmpeg"
                    bundled.symlink_to(target)
                    (bins / "ffprobe").symlink_to(target)
                    try:
                        binaries = wan2gp_runner.require_ffmpeg(root)
                        self.assertNotEqual(Path(binaries.ffmpeg), bundled.absolute())
                        self.assertEqual(
                            Path(binaries.ffmpeg).resolve(),
                            Path(system_ffmpeg).resolve(),
                        )
                        self.assertNotEqual(Path(binaries.ffprobe), (bins / "ffprobe").absolute())
                        self.assertEqual(
                            Path(binaries.ffprobe).resolve(),
                            Path(system_ffprobe).resolve(),
                        )
                        self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                        quarantine = root / "ffmpeg_quarantine"
                        quarantine.mkdir()
                        root_ffmpeg.rename(quarantine / "ffmpeg")
                        version = subprocess.run(
                            [binaries.ffmpeg, "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        self.assertEqual(version.returncode, 0)
                        probe = subprocess.run(
                            [binaries.ffprobe, "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=3,
                        )
                        self.assertEqual(probe.returncode, 0)
                    finally:
                        _restore_env(saved)

    def test_path_alias_symlink_with_libs_stops_before_the_gpu(self) -> None:
        """Le lien ``../self/ffmpeg`` passe le mux, puis imageio ne le trouve plus."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "self").symlink_to(".", target_is_directory=True)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(Path("..") / "self" / "ffmpeg")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 41),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn(str(bundled), message)
            self.assertIn(str(root_ffmpeg), message)
            self.assertIn("ffmpeg_quarantine", message)
            self.assertIn("IMAGEIO_FFMPEG_EXE", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_dotdot_after_directory_symlink_is_not_the_imageio_binary(self) -> None:
        """``trick/../ffmpeg`` ouvre la racine : normpath croit à une boucle.

        ``trick`` pointe vers un sous-dossier du clone. La chaîne écrite
        revient sur ``ffmpeg_bins/ffmpeg``. Le noyau entre dans ``trick``
        puis remonte, et trouve le fichier que la quarantaine déplace.
        ``is_file``, ``-encoders`` et le mux réussissent tant qu'il est là.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "sub").mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (bins / "trick").symlink_to(root / "sub", target_is_directory=True)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(Path("trick") / ".." / "ffmpeg")
            bundled_probe = bins / "ffprobe"
            bundled_probe.symlink_to(Path("trick") / ".." / "ffmpeg")
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertNotEqual(Path(binaries.ffmpeg), bundled.absolute())
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                self.assertNotEqual(Path(binaries.ffprobe), bundled_probe.absolute())
                self.assertEqual(Path(binaries.ffprobe).resolve(), Path(system_ffprobe).resolve())
                self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
                probe = subprocess.run(
                    [binaries.ffprobe, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(probe.returncode, 0)
            finally:
                _restore_env(saved)

    def test_dotdot_after_directory_symlink_with_libs_stops_before_the_gpu(self) -> None:
        """Le lien ``trick/../ffmpeg`` passe le mux, puis imageio ne le trouve plus."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "sub").mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (bins / "trick").symlink_to(root / "sub", target_is_directory=True)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(Path("trick") / ".." / "ffmpeg")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn(str(bundled), message)
            self.assertIn(str(root_ffmpeg), message)
            self.assertIn("ffmpeg_quarantine", message)
            self.assertIn("IMAGEIO_FFMPEG_EXE", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_dotdot_after_directory_symlink_to_system_binary_survives(self) -> None:
        """``trick/../bin/ffmpeg`` vers le binaire système n'est pas déplacé."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        system_path = Path(system_ffmpeg)
        if system_path.name != "ffmpeg":
            self.skipTest("le binaire système ne s'appelle pas ffmpeg")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            holder = Path(directory) / "holder" / "bin"
            holder.mkdir(parents=True)
            held = holder / "ffmpeg"
            held.symlink_to(system_ffmpeg)
            (bins / "trick").symlink_to(holder, target_is_directory=True)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(Path("trick") / ".." / "bin" / "ffmpeg")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg), bundled.absolute())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_directory_alias_to_system_binary_survives_root_quarantine(self) -> None:
        """Un lien ``../usrbin/ffmpeg`` vers le binaire système n'est pas déplacé."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "usrbin").symlink_to(Path(system_ffmpeg).parent, target_is_directory=True)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(Path("..") / "usrbin" / Path(system_ffmpeg).name)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg), bundled.absolute())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_direct_system_symlink_survives_root_quarantine(self) -> None:
        """Un lien vers /usr/bin/ffmpeg n'est pas le fichier déplacé."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            bundled = bins / "ffmpeg"
            bundled.symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg), bundled.absolute())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_root_ffprobe_through_quarantine_is_not_published(self) -> None:
        """ffprobe → ./ffmpeg casse quand ce fichier part en quarantaine."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            root_ffprobe = root / "ffprobe"
            root_ffprobe.symlink_to("ffmpeg")
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertNotEqual(Path(binaries.ffprobe), root_ffprobe.absolute())
                self.assertEqual(Path(binaries.ffprobe).resolve(), Path(system_ffprobe).resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                probe = subprocess.run(
                    [binaries.ffprobe, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(probe.returncode, 0)
            finally:
                _restore_env(saved)

    def test_hardlink_in_bundle_survives_root_quarantine(self) -> None:
        """Un lien dur n'est pas l'entrée de répertoire que shutil.move retire."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            shutil.copy2(system_ffmpeg, root_ffmpeg)
            bundled = bins / "ffmpeg"
            try:
                os.link(root_ffmpeg, bundled)
            except OSError as exc:
                self.skipTest(f"lien dur indisponible : {exc}")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)
            try:
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg), bundled.absolute())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_ffmpeg_chosen_after_quarantine_reexec_stops_before_the_gpu(self) -> None:
        """Après la quarantaine, which trouve un script exec ffmpeg."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            shims = base / "shims"
            (root / "ffmpeg_bins").mkdir(parents=True)
            shims.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            script = shims / "ffmpeg"
            script.write_text('#!/bin/sh\nexec ffmpeg "$@"\n', encoding="utf-8")
            script.chmod(0o755)
            try:
                os.environ["PATH"] = os.pathsep.join(
                    [str(root.absolute()), str(shims.absolute()), os.environ.get("PATH", "")]
                )
                kept = subprocess.run(
                    [str(root_ffmpeg), "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(kept.returncode, 0)
                self.assertIn("libx264", kept.stdout)

                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 11),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn(str(script.absolute()), message)
            self.assertIn("ffmpeg_quarantine", message)
            self.assertIn(str(root_ffmpeg.absolute()), message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_path_entry_that_opens_the_clone_still_feeds_reexec(self) -> None:
        """``holder/trick/..``, ``.`` et ``trick/..`` laissent le script passer.

        ``normpath`` ne voit pas la racine. Le cwd du probe est le clone :
        le noyau ouvre quand même son ``ffmpeg``. ``-encoders`` annonce
        libx264 tant que le fichier est là. Après la quarantaine, le script
        se relance et aucun MP4 n'est écrit.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        cases = ("absolute-dotdot", "relative-dot", "relative-dotdot")
        for case in cases:
            with self.subTest(case=case):
                saved = _saved_env("PATH")
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    shims = base / "shims"
                    (root / "ffmpeg_bins").mkdir(parents=True)
                    (root / "sub").mkdir()
                    shims.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    script = shims / "ffmpeg"
                    script.write_text('#!/bin/sh\nexec ffmpeg "$@"\n', encoding="utf-8")
                    script.chmod(0o755)
                    if case == "absolute-dotdot":
                        holder = base / "holder"
                        holder.mkdir()
                        (holder / "trick").symlink_to(root / "sub", target_is_directory=True)
                        entry = str(holder / "trick" / "..")
                    elif case == "relative-dot":
                        entry = "."
                    elif case == "relative-dotdot":
                        (root / "trick").symlink_to(root / "sub", target_is_directory=True)
                        entry = "trick/.."
                    else:
                        self.fail(case)
                    original = os.environ.get("PATH", "")
                    try:
                        os.environ["PATH"] = os.pathsep.join(
                            [entry, str(shims.absolute()), original]
                        )
                        kept = subprocess.run(
                            [str(script.absolute()), "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(kept.returncode, 0, kept.stderr)
                        self.assertIn("libx264", kept.stdout)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 13),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                    finally:
                        _restore_env(saved)
                    message = str(caught.exception)
                    self.assertIn(str(script.absolute()), message)
                    self.assertIn("ffmpeg_quarantine", message)
                    self.assertIn(str(root_ffmpeg.absolute()), message)
                    self.assertIn("lien de dossier", message)
                    self.assertIn("error saving", message)
                    self.assertIn("code 0", message)
                    self.assertNotIn("GPU", message)

    def test_path_entry_dotdot_to_another_ffmpeg_survives_quarantine(self) -> None:
        """``trick/..`` vers un autre dossier n'est pas le fichier déplacé."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            outside = base / "outside"
            (outside / "sub").mkdir(parents=True)
            outside_ffmpeg = outside / "ffmpeg"
            outside_ffmpeg.symlink_to(system_ffmpeg)
            holder = base / "holder"
            holder.mkdir()
            (holder / "trick").symlink_to(outside / "sub", target_is_directory=True)
            entry = holder / "trick" / ".."
            try:
                os.environ["PATH"] = os.pathsep.join(
                    [str(entry), os.environ.get("PATH", "")]
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).parent.resolve(), outside.resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_relative_dot_on_path_uses_the_ffmpeg_that_survives(self) -> None:
        """``.`` sur le PATH ne masque pas le ffmpeg du système après quarantaine."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            try:
                os.environ["PATH"] = os.pathsep.join([".", os.environ.get("PATH", "")])
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertNotEqual(Path(binaries.ffmpeg).parent.resolve(), root.resolve())
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_proc_cwd_path_opens_the_clone_for_the_child(self) -> None:
        """``/proc/self/cwd`` vu par le clone n'est pas le dossier du contrôle."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "sub").mkdir(parents=True)
            repo = os.path.normpath(str(root.absolute()))
            outside = base / "outside"
            outside.mkdir()
            alias = base / "alias"
            alias.symlink_to("/proc/self/cwd")
            elsewhere = base / "elsewhere"
            elsewhere.symlink_to(outside)
            via_self = base / "via-self"
            via_self.symlink_to("/proc/self")
            via_thread = base / "via-thread"
            via_thread.symlink_to("/proc/thread-self")
            dotdot_alias = base / "dotdot-alias"
            dotdot_alias.symlink_to("/proc/self/../self/cwd")
            cases = {
                "proc-self": "/proc/self/cwd",
                "proc-thread-self": "/proc/thread-self/cwd",
                "proc-dot-slash": "/proc/./self/cwd/",
                "proc-double-slash": "/proc//self/cwd",
                "proc-sub-dotdot": "/proc/self/cwd/sub/..",
                "proc-self-dotdot": "/proc/self/../self/cwd",
                "proc-fd-dotdot": "/proc/self/fd/../cwd",
                "proc-task-dotdot": "/proc/self/task/../cwd",
                "proc-dot-self-dotdot": "/proc/./self/../self/cwd",
                "proc-double-slash-dotdot": "//proc/self/../self/cwd",
                "proc-thread-dotdot": "/proc/thread-self/../../cwd",
                "proc-self-root": "/proc/self/root/proc/self/cwd",
                "proc-self-root-thread": "/proc/self/root/proc/thread-self/../../cwd",
                "proc-self-root-repo": f"/proc/self/root{repo}",
                "alias": str(alias),
                "dotdot-alias": str(dotdot_alias),
                "via-self": str(via_self / "cwd"),
                "via-thread": str(via_thread / ".." / ".." / "cwd"),
            }
            for name, entry in cases.items():
                with self.subTest(case=name):
                    self.assertTrue(wan2gp_runner._same_path_entry(entry, repo, root))
            self.assertFalse(
                wan2gp_runner._same_path_entry("/proc/self/cwd/sub", repo, root)
            )
            self.assertFalse(
                wan2gp_runner._same_path_entry("/proc/self/../self/cwd/sub", repo, root)
            )
            self.assertFalse(wan2gp_runner._same_path_entry(str(via_self / "cwd" / "sub"), repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry("/proc/self/fd", repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry("/proc/self/root/usr/bin", repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry("/proc/thread-self/../cwd", repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry(str(elsewhere), repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry("/usr/bin", repo, root))

    def test_proc_cwd_on_path_still_feeds_reexec(self) -> None:
        """``/proc/self/cwd`` laisse le script passer tant que le fichier est là.

        Le contrôle tourne dans le dossier du pipeline : ``Path.resolve()``
        n'ouvre pas le clone. Depuis le cwd du processus, le noyau trouve
        le ``ffmpeg`` que la quarantaine va déplacer. ``-encoders`` annonce
        libx264. Après le déplacement, le script se relance et aucun MP4
        n'est écrit.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        cases = {
            "proc-self": "/proc/self/cwd",
            "proc-thread-self": "/proc/thread-self/cwd",
            "proc-sub-dotdot": "/proc/self/cwd/sub/..",
            "proc-self-dotdot": "/proc/self/../self/cwd",
            "proc-thread-dotdot": "/proc/thread-self/../../cwd",
            "proc-self-root": "/proc/self/root/proc/self/cwd",
            "via-thread": "/proc/thread-self",
            "alias": "/proc/self/cwd",
            "alias-dotdot": "/proc/self/../self/cwd",
        }
        for name, entry in cases.items():
            with self.subTest(case=name):
                saved = _saved_env("PATH")
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    shims = base / "shims"
                    (root / "ffmpeg_bins").mkdir(parents=True)
                    (root / "sub").mkdir()
                    shims.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    script = shims / "ffmpeg"
                    script.write_text('#!/bin/sh\nexec ffmpeg "$@"\n', encoding="utf-8")
                    script.chmod(0o755)
                    if name.startswith("alias"):
                        alias = base / "alias"
                        alias.symlink_to(entry)
                        chosen = str(alias)
                    elif name == "via-thread":
                        via = base / "via-thread"
                        via.symlink_to(entry)
                        chosen = str(via / ".." / ".." / "cwd")
                    else:
                        chosen = entry
                    original = os.environ.get("PATH", "")
                    try:
                        os.environ["PATH"] = os.pathsep.join(
                            [chosen, str(shims.absolute()), original]
                        )
                        kept = subprocess.run(
                            [str(script.absolute()), "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(kept.returncode, 0, kept.stderr)
                        self.assertIn("libx264", kept.stdout)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 17),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                    finally:
                        _restore_env(saved)
                    message = str(caught.exception)
                    self.assertIn(str(script.absolute()), message)
                    self.assertIn("ffmpeg_quarantine", message)
                    self.assertIn(str(root_ffmpeg.absolute()), message)
                    self.assertIn("/proc/self/cwd", message)
                    self.assertIn("/proc/thread-self/cwd", message)
                    self.assertIn("normpath", message)
                    self.assertIn("/proc/self/root", message)
                    self.assertIn("error saving", message)
                    self.assertIn("code 0", message)
                    self.assertNotIn("GPU", message)

    def test_proc_cwd_on_path_uses_the_ffmpeg_that_survives(self) -> None:
        """``/proc/self/cwd`` ne masque pas le ffmpeg du système après quarantaine."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            try:
                os.environ["PATH"] = os.pathsep.join(
                    ["/proc/self/cwd", os.environ.get("PATH", "")]
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_proc_self_dotdot_on_path_uses_the_ffmpeg_that_survives(self) -> None:
        """``/proc/self/../self/cwd`` ne masque pas le ffmpeg du système.

        ``normpath`` réécrit cette entrée en ``/proc/self/cwd``. Le noyau,
        avec le cwd du clone, ouvre quand même la racine. Après l'avoir
        retirée du PATH, le binaire du système reste lançable une fois le
        fichier de la racine mis en quarantaine.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            via = base / "via-self"
            via.symlink_to("/proc/self")
            try:
                os.environ["PATH"] = os.pathsep.join(
                    [
                        "/proc/self/../self/cwd",
                        "/proc/thread-self/../../cwd",
                        str(via / "cwd"),
                        os.environ.get("PATH", ""),
                    ]
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_proc_real_dir_dotdot_opens_the_clone_for_the_child(self) -> None:
        """``/proc/sys/../self/cwd`` n'a pas ``self`` juste après ``proc``.

        ``normpath`` réécrit quand même l'entrée en ``/proc/self/cwd``.
        ``download_ffmpeg`` ne la retire pas : ce n'est pas le texte de la
        racine. Depuis le cwd du clone, le noyau entre dans ``sys`` puis
        ouvre cette racine. ``Path.resolve()`` dans le contrôle suit le
        ``/proc/self`` du pipeline.
        """
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "sub").mkdir(parents=True)
            repo = os.path.normpath(str(root.absolute()))
            self.assertNotEqual(os.path.normpath("/proc/sys/../self/cwd"), repo)
            via_sys = base / "via-sys"
            via_sys.symlink_to("/proc/sys")
            direct = base / "direct"
            direct.symlink_to("/proc/sys/../self/cwd")
            elsewhere = base / "elsewhere"
            elsewhere.symlink_to("/proc/sys")
            cases = {
                "proc-sys": "/proc/sys/../self/cwd",
                "proc-sys-kernel": "/proc/sys/kernel/../../self/cwd",
                "proc-sys-thread": "/proc/sys/../thread-self/cwd",
                "proc-sys-dot": "/proc/./sys/../self/cwd",
                "via-sys": str(via_sys / ".." / "self" / "cwd"),
                "direct": str(direct),
            }
            for name, entry in cases.items():
                with self.subTest(case=name):
                    self.assertTrue(wan2gp_runner._same_path_entry(entry, repo, root))
            self.assertFalse(
                wan2gp_runner._same_path_entry("/proc/sys/../self/cwd/sub", repo, root)
            )
            self.assertFalse(wan2gp_runner._same_path_entry("/proc/sys", repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry(str(elsewhere), repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry("/usr/bin", repo, root))

    def test_proc_sys_dotdot_on_path_still_feeds_reexec(self) -> None:
        """``/proc/sys/../self/cwd`` laisse le script passer tant que le fichier est là."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        cases = {
            "proc-sys": "/proc/sys/../self/cwd",
            "proc-sys-thread": "/proc/sys/../thread-self/cwd",
            "via-sys": "/proc/sys",
            "direct": "/proc/sys/../self/cwd",
        }
        for name, entry in cases.items():
            with self.subTest(case=name):
                saved = _saved_env("PATH")
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    shims = base / "shims"
                    (root / "ffmpeg_bins").mkdir(parents=True)
                    (root / "sub").mkdir()
                    shims.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    script = shims / "ffmpeg"
                    script.write_text('#!/bin/sh\nexec ffmpeg "$@"\n', encoding="utf-8")
                    script.chmod(0o755)
                    if name == "via-sys":
                        alias = base / "via-sys"
                        alias.symlink_to(entry)
                        chosen = str(alias / ".." / "self" / "cwd")
                    elif name == "direct":
                        alias = base / "direct"
                        alias.symlink_to(entry)
                        chosen = str(alias)
                    else:
                        chosen = entry
                    original = os.environ.get("PATH", "")
                    try:
                        os.environ["PATH"] = os.pathsep.join(
                            [chosen, str(shims.absolute()), original]
                        )
                        kept = subprocess.run(
                            [str(script.absolute()), "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(kept.returncode, 0, kept.stderr)
                        self.assertIn("libx264", kept.stdout)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 17),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                    finally:
                        _restore_env(saved)
                    message = str(caught.exception)
                    self.assertIn(str(script.absolute()), message)
                    self.assertIn("ffmpeg_quarantine", message)
                    self.assertIn(str(root_ffmpeg.absolute()), message)
                    self.assertIn("/proc/sys/../self/cwd", message)
                    self.assertIn("/proc/sys/../thread-self/cwd", message)
                    self.assertIn("error saving", message)
                    self.assertIn("code 0", message)
                    self.assertNotIn("GPU", message)

    def test_proc_sys_dotdot_on_path_uses_the_ffmpeg_that_survives(self) -> None:
        """``/proc/sys/../self/cwd`` ne masque pas le ffmpeg du système."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            via = base / "via-sys"
            via.symlink_to("/proc/sys")
            direct = base / "direct"
            direct.symlink_to("/proc/sys/../self/cwd")
            try:
                os.environ["PATH"] = os.pathsep.join(
                    [
                        "/proc/sys/../self/cwd",
                        "/proc/sys/../thread-self/cwd",
                        str(via / ".." / "self" / "cwd"),
                        str(direct),
                        os.environ.get("PATH", ""),
                    ]
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_symlink_chain_dotdot_opens_the_clone_for_the_child(self) -> None:
        """``hop1/../cwd`` n'est pas une boucle quand ``hop2`` entre dans ``/proc``.

        ``hop1`` pointe vers ``hop2``, ``hop2`` vers ``/proc/net``. Les deux
        liens sont dans le même dossier : ``normpath`` réécrit ``hop1/../cwd``
        et ``hop2/../cwd`` en ``<parent>/cwd``. Ce n'est pas le texte de la
        racine, donc ``download_ffmpeg`` garde l'entrée. Le noyau entre dans
        ``hop1``, puis dans ``hop2``, et ``..`` ouvre ``/proc/self/cwd``.
        ``Path.resolve()`` dans le contrôle suit le cwd du pipeline.
        """
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "sub").mkdir(parents=True)
            outside = base / "outside"
            outside.mkdir()
            repo = os.path.normpath(str(root.absolute()))
            self.assertNotEqual(os.path.normpath(str(base / "hop1" / ".." / "cwd")), repo)

            hop2 = base / "hop2"
            hop2.symlink_to("/proc/net")
            hop1 = base / "hop1"
            hop1.symlink_to(hop2)
            rel2 = base / "rel2"
            rel2.symlink_to("/proc/net")
            rel1 = base / "rel1"
            rel1.symlink_to("rel2")
            sys2 = base / "sys2"
            sys2.symlink_to("/proc/sys")
            sys1 = base / "sys1"
            sys1.symlink_to(sys2)
            mid = base / "mid"
            mid.symlink_to(hop2)
            hop3 = base / "hop3"
            hop3.symlink_to(mid)
            elsewhere2 = base / "elsewhere2"
            elsewhere2.symlink_to(outside)
            elsewhere1 = base / "elsewhere1"
            elsewhere1.symlink_to(elsewhere2)

            cases = {
                "net-chain": str(hop1 / ".." / "cwd"),
                "rel-chain": str(rel1 / ".." / "cwd"),
                "sys-chain": str(sys1 / ".." / "self" / "cwd"),
                "three-hop": str(hop3 / ".." / "cwd"),
                "direct-net": str(hop2 / ".." / "cwd"),
            }
            for name, entry in cases.items():
                with self.subTest(case=name):
                    self.assertTrue(wan2gp_runner._same_path_entry(entry, repo, root))
            self.assertFalse(
                wan2gp_runner._same_path_entry(str(sys1 / ".." / "cwd"), repo, root)
            )
            self.assertFalse(
                wan2gp_runner._same_path_entry(str(hop1 / ".." / "cwd" / "sub"), repo, root)
            )
            self.assertFalse(
                wan2gp_runner._same_path_entry(str(elsewhere1 / ".." / "cwd"), repo, root)
            )
            self.assertFalse(wan2gp_runner._same_path_entry(str(hop2), repo, root))
            self.assertFalse(wan2gp_runner._same_path_entry("/usr/bin", repo, root))

    def test_symlink_chain_dotdot_on_path_still_feeds_reexec(self) -> None:
        """``hop1/../cwd`` laisse le script passer tant que le fichier est là."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        cases = ("net-chain", "sys-chain", "rel-chain")
        for case in cases:
            with self.subTest(case=case):
                saved = _saved_env("PATH")
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    shims = base / "shims"
                    (root / "ffmpeg_bins").mkdir(parents=True)
                    (root / "sub").mkdir()
                    shims.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    script = shims / "ffmpeg"
                    script.write_text('#!/bin/sh\nexec ffmpeg "$@"\n', encoding="utf-8")
                    script.chmod(0o755)
                    if case == "net-chain":
                        second = base / "hop2"
                        second.symlink_to("/proc/net")
                        first = base / "hop1"
                        first.symlink_to(second)
                        chosen = str(first / ".." / "cwd")
                    elif case == "sys-chain":
                        second = base / "sys2"
                        second.symlink_to("/proc/sys")
                        first = base / "sys1"
                        first.symlink_to(second)
                        chosen = str(first / ".." / "self" / "cwd")
                    elif case == "rel-chain":
                        second = base / "rel2"
                        second.symlink_to("/proc/net")
                        first = base / "rel1"
                        first.symlink_to("rel2")
                        chosen = str(first / ".." / "cwd")
                    else:
                        self.fail(case)
                    original = os.environ.get("PATH", "")
                    try:
                        os.environ["PATH"] = os.pathsep.join(
                            [chosen, str(shims.absolute()), original]
                        )
                        kept = subprocess.run(
                            [str(script.absolute()), "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(kept.returncode, 0, kept.stderr)
                        self.assertIn("libx264", kept.stdout)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 17),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                    finally:
                        _restore_env(saved)
                    message = str(caught.exception)
                    self.assertIn(str(script.absolute()), message)
                    self.assertIn("ffmpeg_quarantine", message)
                    self.assertIn(str(root_ffmpeg.absolute()), message)
                    self.assertIn("hop1/../cwd", message)
                    self.assertIn("/proc/net", message)
                    self.assertIn("hop1/../self/cwd", message)
                    self.assertIn("normpath", message)
                    self.assertIn("error saving", message)
                    self.assertIn("code 0", message)
                    self.assertNotIn("GPU", message)

    def test_symlink_chain_dotdot_on_path_uses_the_ffmpeg_that_survives(self) -> None:
        """``hop1/../cwd`` ne masque pas le ffmpeg du système après quarantaine."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        saved = _saved_env("PATH", "FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            (root / "ffmpeg_bins").mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            hop2 = base / "hop2"
            hop2.symlink_to("/proc/net")
            hop1 = base / "hop1"
            hop1.symlink_to(hop2)
            rel2 = base / "rel2"
            rel2.symlink_to("/proc/net")
            rel1 = base / "rel1"
            rel1.symlink_to("rel2")
            sys2 = base / "sys2"
            sys2.symlink_to("/proc/sys")
            sys1 = base / "sys1"
            sys1.symlink_to(sys2)
            try:
                os.environ["PATH"] = os.pathsep.join(
                    [
                        str(hop1 / ".." / "cwd"),
                        str(rel1 / ".." / "cwd"),
                        str(sys1 / ".." / "self" / "cwd"),
                        os.environ.get("PATH", ""),
                    ]
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0)
            finally:
                _restore_env(saved)

    def test_proc_net_bundle_symlink_is_not_the_imageio_binary(self) -> None:
        """``tonet/../cwd/ffmpeg`` ouvre la racine seulement depuis le clone.

        ``tonet`` pointe vers ``/proc/net``. ``normpath`` annule ``..`` avant
        ce lien. ``is_file`` depuis un cwd qui a son propre ``ffmpeg`` réussit,
        et ce n'est pas le fichier déplacé. Avec le cwd du clone, le noyau
        entre dans ``/proc/net``, ``..`` ouvre ``/proc/self`` et ``cwd`` tombe
        sur le fichier de la racine. ``-encoders`` réussit tant qu'il est là.
        Après la quarantaine, imageio ne trouve plus le lien.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        shapes = ("tonet", "hop-chain")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    bins = root / "ffmpeg_bins"
                    bins.mkdir(parents=True)
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    (decoy / "ffmpeg").symlink_to(system_ffmpeg)
                    if shape == "tonet":
                        (bins / "tonet").symlink_to("/proc/net")
                        target = "tonet/../cwd/ffmpeg"
                    else:
                        (bins / "hop2").symlink_to("/proc/net")
                        (bins / "hop1").symlink_to("hop2")
                        target = "hop1/../cwd/ffmpeg"
                    bundled = bins / "ffmpeg"
                    bundled.symlink_to(target)
                    (bins / "ffprobe").symlink_to(system_ffprobe)
                    try:
                        os.chdir(decoy)
                        self.assertTrue(bundled.is_file())
                        self.assertTrue(
                            wan2gp_runner._depends_on_quarantined_file(bundled, root_ffmpeg)
                        )
                        kept = subprocess.run(
                            [str(bundled.absolute()), "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(kept.returncode, 0, kept.stderr)
                        binaries = wan2gp_runner.require_ffmpeg(root)
                        self.assertNotEqual(Path(binaries.ffmpeg), bundled.absolute())
                        self.assertEqual(
                            Path(binaries.ffmpeg).resolve(),
                            Path(system_ffmpeg).resolve(),
                        )
                        self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                        quarantine = root / "ffmpeg_quarantine"
                        quarantine.mkdir()
                        root_ffmpeg.rename(quarantine / "ffmpeg")
                        version = subprocess.run(
                            [binaries.ffmpeg, "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(version.returncode, 0, version.stderr)
                        with self.assertRaises(FileNotFoundError):
                            subprocess.run(
                                [str(bundled.absolute()), "-hide_banner", "-version"],
                                check=False,
                                capture_output=True,
                                text=True,
                                cwd=root,
                                timeout=3,
                            )
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_net_bundle_symlink_with_libs_stops_before_the_gpu(self) -> None:
        """Le lien via ``/proc/net`` passe le mux, puis imageio ne le trouve plus."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["rainy_window"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            decoy = base / "pipeline"
            decoy.mkdir()
            (decoy / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "tonet").symlink_to("/proc/net")
            bundled = bins / "ffmpeg"
            bundled.symlink_to("tonet/../cwd/ffmpeg")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / soname).symlink_to(library)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            try:
                os.chdir(decoy)
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 61),
                                settings=replace(config.wan, root=root),
                                output_dir=base / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                os.chdir(previous)
                _restore_env(saved)
            message = str(caught.exception)
            self.assertIn(str(bundled), message)
            self.assertIn(str(root_ffmpeg), message)
            self.assertIn("ffmpeg_quarantine", message)
            self.assertIn("IMAGEIO_FFMPEG_EXE", message)
            self.assertIn("/proc/net", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("GPU", message)

    def test_proc_net_symlink_to_another_ffmpeg_survives_quarantine(self) -> None:
        """``/proc/net`` puis ``root`` ou ``cwd/sub`` n'est pas le fichier déplacé."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        if Path(system_ffmpeg).absolute() != Path("/usr/bin/ffmpeg"):
            self.skipTest("le ffmpeg du système n'est pas /usr/bin/ffmpeg")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        shapes = ("proc-root", "clone-sub")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    bins = root / "ffmpeg_bins"
                    bins.mkdir(parents=True)
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    (bins / "tonet").symlink_to("/proc/net")
                    bundled = bins / "ffmpeg"
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    if shape == "proc-root":
                        bundled.symlink_to("tonet/../root/usr/bin/ffmpeg")
                    else:
                        (root / "sub").mkdir()
                        (root / "sub" / "ffmpeg").symlink_to(system_ffmpeg)
                        (decoy / "sub").mkdir()
                        (decoy / "sub" / "ffmpeg").symlink_to(system_ffmpeg)
                        bundled.symlink_to("tonet/../cwd/sub/ffmpeg")
                    (bins / "ffprobe").symlink_to(system_ffprobe)
                    (bins / soname).symlink_to(library)
                    try:
                        if shape == "clone-sub":
                            os.chdir(decoy)
                        self.assertTrue(bundled.is_file())
                        binaries = wan2gp_runner.require_ffmpeg(root)
                        self.assertEqual(Path(binaries.ffmpeg), bundled.absolute())
                        self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                        quarantine = root / "ffmpeg_quarantine"
                        quarantine.mkdir()
                        root_ffmpeg.rename(quarantine / "ffmpeg")
                        version = subprocess.run(
                            [binaries.ffmpeg, "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(version.returncode, 0, version.stderr)
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_directory_symlink_is_not_the_imageio_binary(self) -> None:
        """``ffmpeg_bins`` vers ``/proc/self/cwd`` n'est pas le binaire publié.

        ``is_symlink`` suit le dossier. Un fichier régulier ``ffmpeg`` dans
        le cwd du pipeline rend ce prédicat faux et ``is_file`` vrai. Ce
        n'est pas l'entrée de la racine. Avec le cwd du clone, le même
        chemin ouvre le fichier que ``download_ffmpeg`` déplace. ``-encoders``
        réussit tant qu'il est là. ``IMAGEIO_FFMPEG_EXE`` garderait ce
        chemin, puis imageio ne le trouve plus.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        shapes = ("self-cwd", "proc-net")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    root_ffmpeg = root / "ffmpeg"
                    root_ffmpeg.symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    planted = decoy / "ffmpeg"
                    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                    planted.chmod(0o755)
                    if shape == "self-cwd":
                        target = "/proc/self/cwd"
                    elif shape == "proc-net":
                        target = "/proc/net/../cwd"
                    else:
                        self.fail(shape)
                    bins = root / "ffmpeg_bins"
                    bins.symlink_to(target)
                    bundled = bins / "ffmpeg"
                    try:
                        os.chdir(decoy)
                        self.assertTrue(bundled.is_file())
                        self.assertFalse(bundled.is_symlink())
                        self.assertTrue(
                            wan2gp_runner._depends_on_quarantined_file(bundled, root_ffmpeg)
                        )
                        kept = subprocess.run(
                            [str(bundled.absolute()), "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(kept.returncode, 0, kept.stderr)
                        binaries = wan2gp_runner.require_ffmpeg(root)
                        self.assertNotEqual(Path(binaries.ffmpeg), bundled.absolute())
                        self.assertEqual(
                            Path(binaries.ffmpeg).resolve(),
                            Path(system_ffmpeg).resolve(),
                        )
                        self.assertEqual(os.environ["IMAGEIO_FFMPEG_EXE"], binaries.ffmpeg)
                        quarantine = root / "ffmpeg_quarantine"
                        quarantine.mkdir()
                        root_ffmpeg.rename(quarantine / "ffmpeg")
                        version = subprocess.run(
                            [binaries.ffmpeg, "-hide_banner", "-version"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            timeout=3,
                        )
                        self.assertEqual(version.returncode, 0, version.stderr)
                        with self.assertRaises(FileNotFoundError):
                            subprocess.run(
                                [str(bundled.absolute()), "-hide_banner", "-version"],
                                check=False,
                                capture_output=True,
                                text=True,
                                cwd=root,
                                timeout=3,
                            )
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_cwd_bundle_short_library_stops_before_the_gpu(self) -> None:
        """Un ``libx264`` trop court du clone est chargé via ``/proc/self/cwd``.

        ``ffmpeg_bins`` pointe vers ``/proc/self/cwd``, ``/proc/net/../cwd``
        ou ``/proc/self/cwd/only`` (dossier présent seulement dans le clone).
        ``iterdir`` depuis le pipeline ne voit pas le soname : il liste le
        cwd du pipeline, ou lève ``FileNotFoundError`` si ``only`` n'y est
        pas. Le binaire du PATH est retenu. ``download_ffmpeg`` préfixe
        quand même ``LD_LIBRARY_PATH`` avec le lien. Le chargeur, cwd =
        clone, ouvre ce soname et sort 127 (``file too short``).
        ``-encoders`` sans le préfixe annonce encore libx264.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libx264 illisible")
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = ("self-cwd", "proc-net", "clone-subdir")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    (root / "ffmpeg").symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    library_dir = root
                    if shape == "clone-subdir":
                        library_dir = root / "only"
                        library_dir.mkdir()
                    (library_dir / soname).write_bytes(b"short")
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    planted = decoy / "ffmpeg"
                    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                    planted.chmod(0o755)
                    if shape == "self-cwd":
                        target = "/proc/self/cwd"
                    elif shape == "proc-net":
                        target = "/proc/net/../cwd"
                    elif shape == "clone-subdir":
                        target = "/proc/self/cwd/only"
                    else:
                        self.fail(shape)
                    bins = root / "ffmpeg_bins"
                    bins.symlink_to(target)
                    try:
                        os.chdir(decoy)
                        if shape == "clone-subdir":
                            with self.assertRaises(FileNotFoundError):
                                list(bins.iterdir())
                        else:
                            self.assertEqual([path.name for path in bins.iterdir()], ["ffmpeg"])
                        plain = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(plain.returncode, 0, plain.stderr)
                        self.assertIn("libx264", plain.stdout)
                        env = os.environ.copy()
                        prefix = str(bins.absolute())
                        inherited = env.get("LD_LIBRARY_PATH", "")
                        env["LD_LIBRARY_PATH"] = (
                            prefix if not inherited else prefix + os.pathsep + inherited
                        )
                        shadowed = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            env=env,
                            timeout=5,
                        )
                        self.assertEqual(shadowed.returncode, 127, shadowed.stderr)
                        self.assertIn("file too short", shadowed.stderr)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 17),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn(soname, message)
                        self.assertIn("LD_LIBRARY_PATH", message)
                        self.assertIn("file too short", message)
                        self.assertIn("error saving", message)
                        self.assertIn(str(bins), message)
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_cwd_bundle_ignores_a_library_only_in_the_pipeline_cwd(self) -> None:
        """Un soname du cwd du pipeline n'est pas celui du chargeur.

        ``iterdir`` le verrait. Le processus ffmpeg a pour cwd le clone :
        ``/proc/self/cwd/<soname>`` n'est pas ce fichier. Le mux du binaire
        système reste possible.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libx264 illisible")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "ffmpeg").symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            decoy = base / "pipeline"
            decoy.mkdir()
            planted = decoy / "ffmpeg"
            planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            planted.chmod(0o755)
            (decoy / soname).write_bytes(b"short")
            bins = root / "ffmpeg_bins"
            bins.symlink_to("/proc/self/cwd")
            try:
                os.chdir(decoy)
                self.assertIn(soname, [path.name for path in bins.iterdir()])
                env = os.environ.copy()
                prefix = str(bins.absolute())
                inherited = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    prefix if not inherited else prefix + os.pathsep + inherited
                )
                loaded = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    env=env,
                    timeout=5,
                )
                self.assertEqual(loaded.returncode, 0, loaded.stderr)
                self.assertIn("libx264", loaded.stdout)
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)
                _restore_env(saved)

    def test_hwcap_proc_cwd_short_library_stops_before_the_gpu(self) -> None:
        """Un ``libx264`` trop court du clone est chargé via un sous-dossier proc.

        ``ffmpeg_bins`` est un vrai dossier. ``glibc-hwcaps`` ou le niveau
        ``glibc-hwcaps/<niveau>`` pointe vers ``/proc/self/cwd``,
        ``/proc/net/../cwd`` ou ``/proc/self/cwd/only``. ``iterdir`` depuis
        le pipeline ne voit pas le soname. Le binaire du PATH est retenu.
        ``download_ffmpeg`` préfixe quand même ``LD_LIBRARY_PATH``. Le
        chargeur, cwd = clone, ouvre ce soname et sort 127 (``file too short``).
        ``-encoders`` sans le préfixe annonce encore libx264.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = ("level-self", "level-proc-net", "level-only", "hwcaps-dir")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    (root / "ffmpeg").symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    bins = root / "ffmpeg_bins"
                    bins.mkdir()
                    if shape == "hwcaps-dir":
                        hidden = root / level
                        hidden.mkdir()
                        (hidden / soname).write_bytes(b"short")
                        (bins / "glibc-hwcaps").symlink_to("/proc/self/cwd")
                    else:
                        slot = bins / "glibc-hwcaps" / level
                        slot.parent.mkdir(parents=True)
                        if shape == "level-self":
                            (root / soname).write_bytes(b"short")
                            slot.symlink_to("/proc/self/cwd")
                        elif shape == "level-proc-net":
                            (root / soname).write_bytes(b"short")
                            slot.symlink_to("/proc/net/../cwd")
                        elif shape == "level-only":
                            hidden = root / "only"
                            hidden.mkdir()
                            (hidden / soname).write_bytes(b"short")
                            slot.symlink_to("/proc/self/cwd/only")
                        else:
                            self.fail(shape)
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    planted = decoy / "ffmpeg"
                    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                    planted.chmod(0o755)
                    try:
                        os.chdir(decoy)
                        plain = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(plain.returncode, 0, plain.stderr)
                        self.assertIn("libx264", plain.stdout)
                        env = os.environ.copy()
                        prefix = str(bins.absolute())
                        inherited = env.get("LD_LIBRARY_PATH", "")
                        env["LD_LIBRARY_PATH"] = (
                            prefix if not inherited else prefix + os.pathsep + inherited
                        )
                        shadowed = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            env=env,
                            timeout=5,
                        )
                        self.assertEqual(shadowed.returncode, 127, shadowed.stderr)
                        self.assertIn("file too short", shadowed.stderr)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 19),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn(soname, message)
                        self.assertIn(level, message)
                        self.assertIn("glibc-hwcaps", message)
                        self.assertIn("iterdir", message)
                        self.assertIn("LD_LIBRARY_PATH", message)
                        self.assertIn("file too short", message)
                        self.assertIn("error saving", message)
                        self.assertIn(str(bins), message)
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_soname_proc_cwd_symlink_stops_before_the_gpu(self) -> None:
        """Un lien de soname vers ``/proc/self/cwd`` charge le fichier du clone.

        ``ffmpeg_bins/<soname>`` vise ``/proc/self/cwd/payload.so`` ou
        ``/proc/net/../cwd/payload.so``. Depuis le pipeline, ``is_file``
        ne voit pas ce fichier. Le chargeur, cwd = clone, le lit et sort
        127 (``file too short``). ``-encoders`` sans ``LD_LIBRARY_PATH``
        annonce encore libx264.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libx264 illisible")
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = ("payload-self", "payload-proc-net")
        for shape in shapes:
            with self.subTest(shape=shape):
                saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
                    (root / "ffmpeg").symlink_to(system_ffmpeg)
                    (root / "ffprobe").symlink_to(system_ffprobe)
                    (root / "payload.so").write_bytes(b"short")
                    bins = root / "ffmpeg_bins"
                    bins.mkdir()
                    target = (
                        "/proc/self/cwd/payload.so"
                        if shape == "payload-self"
                        else "/proc/net/../cwd/payload.so"
                    )
                    if shape not in {"payload-self", "payload-proc-net"}:
                        self.fail(shape)
                    (bins / soname).symlink_to(target)
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    planted = decoy / "ffmpeg"
                    planted.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                    planted.chmod(0o755)
                    try:
                        os.chdir(decoy)
                        self.assertFalse((bins / soname).is_file())
                        plain = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        self.assertEqual(plain.returncode, 0, plain.stderr)
                        self.assertIn("libx264", plain.stdout)
                        env = os.environ.copy()
                        prefix = str(bins.absolute())
                        inherited = env.get("LD_LIBRARY_PATH", "")
                        env["LD_LIBRARY_PATH"] = (
                            prefix if not inherited else prefix + os.pathsep + inherited
                        )
                        shadowed = subprocess.run(
                            [system_ffmpeg, "-hide_banner", "-encoders"],
                            check=False,
                            capture_output=True,
                            text=True,
                            cwd=root,
                            env=env,
                            timeout=5,
                        )
                        self.assertEqual(shadowed.returncode, 127, shadowed.stderr)
                        self.assertIn("file too short", shadowed.stderr)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 23),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn(soname, message)
                        self.assertIn("payload.so", message)
                        self.assertIn("iterdir", message)
                        self.assertIn("LD_LIBRARY_PATH", message)
                        self.assertIn("file too short", message)
                        self.assertIn("error saving", message)
                        self.assertIn(str(bins), message)
                    finally:
                        os.chdir(previous)
                        _restore_env(saved)

    def test_proc_cwd_library_only_in_the_pipeline_still_muxes(self) -> None:
        """Un soname visible seulement depuis le pipeline n'est pas chargé.

        Le lien passe par ``/proc/self``. ``is_file`` ici le voit. Le
        processus ffmpeg a pour cwd le clone : ce fichier n'y est pas.
        Le mux du binaire système reste possible.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "ffmpeg").symlink_to(system_ffmpeg)
            (root / "ffprobe").symlink_to(system_ffprobe)
            bins = root / "ffmpeg_bins"
            bins.mkdir()
            slot = bins / "glibc-hwcaps" / level
            slot.parent.mkdir(parents=True)
            slot.symlink_to("/proc/self/cwd")
            (bins / soname).symlink_to("/proc/self/cwd/payload.so")
            decoy = base / "pipeline"
            decoy.mkdir()
            (decoy / soname).write_bytes(b"short")
            (decoy / "payload.so").write_bytes(b"short")
            try:
                os.chdir(decoy)
                self.assertTrue((bins / soname).is_file())
                self.assertIn(soname, [path.name for path in slot.iterdir()])
                env = os.environ.copy()
                prefix = str(bins.absolute())
                inherited = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = (
                    prefix if not inherited else prefix + os.pathsep + inherited
                )
                loaded = subprocess.run(
                    [system_ffmpeg, "-hide_banner", "-encoders"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    env=env,
                    timeout=5,
                )
                self.assertEqual(loaded.returncode, 0, loaded.stderr)
                self.assertIn("libx264", loaded.stdout)
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)
                _restore_env(saved)

    def test_directory_symlink_to_another_ffmpeg_survives_quarantine(self) -> None:
        """Un ``ffmpeg_bins`` vers un autre dossier n'est pas la racine du clone."""
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        library = None if soname is None else _resolved_library(system_ffmpeg, soname)
        if soname is None or library is None:
            self.skipTest("libavdevice du système est requis")
        saved = _saved_env("FFMPEG_BINARY", "FFPROBE_BINARY", "IMAGEIO_FFMPEG_EXE")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            other = base / "other"
            other.mkdir()
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            root_ffmpeg = root / "ffmpeg"
            root_ffmpeg.symlink_to(system_ffmpeg)
            (other / "ffmpeg").symlink_to(system_ffmpeg)
            (other / "ffprobe").symlink_to(system_ffprobe)
            (other / soname).symlink_to(library)
            decoy = base / "pipeline"
            decoy.mkdir()
            planted = decoy / "ffmpeg"
            planted.write_bytes(b"regular")
            bins = root / "ffmpeg_bins"
            bins.symlink_to(other)
            bundled = bins / "ffmpeg"
            try:
                os.chdir(decoy)
                self.assertTrue(bundled.is_file())
                self.assertFalse(
                    wan2gp_runner._depends_on_quarantined_file(bundled, root_ffmpeg)
                )
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg), bundled.absolute())
                quarantine = root / "ffmpeg_quarantine"
                quarantine.mkdir()
                root_ffmpeg.rename(quarantine / "ffmpeg")
                version = subprocess.run(
                    [binaries.ffmpeg, "-hide_banner", "-version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=root,
                    timeout=3,
                )
                self.assertEqual(version.returncode, 0, version.stderr)
            finally:
                os.chdir(previous)
                _restore_env(saved)

    def test_ffmpeg_directory_in_bins_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").mkdir()
            (bins / "ffprobe").mkdir()

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 31),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("FFMPEG_BINARY", message)
            self.assertIn("dossier", message)
            self.assertIn(str(bins / "ffmpeg"), message)
            self.assertIn(str(bins / "ffprobe"), message)

    def test_root_ffmpeg_fifo_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.mkfifo(root / "ffmpeg")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 32),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(str(root / "ffmpeg"), message)
            self.assertIn("FFMPEG_BINARY", message)

    def test_bundled_ffmpeg_wins_over_a_root_directory(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "ffmpeg").mkdir()
            ffmpeg_link = bins / "ffmpeg"
            ffprobe_link = bins / "ffprobe"
            ffmpeg_link.symlink_to(system_ffmpeg)
            ffprobe_link.symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            seen: dict[str, str] = {}

            def fake_wan(command: list[str], **kwargs: object) -> _FakeProcess:
                env = kwargs.get("env")
                assert isinstance(env, dict)
                seen["ffmpeg"] = str(env["FFMPEG_BINARY"])
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 33),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertEqual(seen["ffmpeg"], str(ffmpeg_link))

    def test_short_needed_library_beside_bundled_ffmpeg_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        ldd = subprocess.run(
            ["ldd", system_ffmpeg],
            check=False,
            capture_output=True,
            text=True,
        )
        soname = ""
        for line in ldd.stdout.splitlines():
            name = line.strip().split(" ", 1)[0]
            if name.startswith("libavcodec.so"):
                soname = name
                break
        if not soname:
            self.skipTest("libavcodec absent du chargement de ffmpeg")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            # Le glob libavdevice.so* est satisfait par un soname que ce binaire ne charge pas.
            (bins / "libavdevice.so.61").write_bytes(b"")
            (bins / soname).write_bytes(b"")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 34),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_ffmpeg_bins_shared_libs_without_a_binary_stop_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "libavcodec.so.60").write_bytes(b"")
            (bins / "libavdevice.so.61").write_bytes(b"")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 21),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("libavcodec.so.60", message)

    def test_ffmpeg_bins_encoder_library_without_a_binary_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "libx264.so.164").write_bytes(b"")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 29),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("libx264.so.164", message)

    def test_needed_soname_directory_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavcodec(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavcodec illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / soname).mkdir()

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 35),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("dossier", message)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_needed_soname_directory_symlink_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavcodec(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavcodec illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            real = bins / "pas-une-bibliotheque"
            real.mkdir()
            (bins / soname).symlink_to(real)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 36),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("dossier", message)
            self.assertIn(soname, message)

    def test_needed_soname_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavcodec(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavcodec illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.mkfifo(bins / soname)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube du soname a bloqué le contrôle ffmpeg")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 37),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)

    def test_bundled_ffmpeg_needed_device_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavdevice illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            # Le glob libavdevice.so* est vrai pour ce tube, et c'est le NEEDED.
            os.mkfifo(bins / soname)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube à côté du binaire a bloqué le mux d'essai")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 41),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("libavdevice.so", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_needed_soname_null_device_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavcodec(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavcodec illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / soname).symlink_to("/dev/null")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 39),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("périphérique", message)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("/dev/null", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_unrelated_null_device_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavcodec(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavcodec illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            # Autre numéro que le NEEDED : le chargeur ne s'arrête pas dessus.
            (bins / "libavdevice.so.61").symlink_to("/dev/null")
            self.assertNotEqual(soname, "libavdevice.so.61")

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 40),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_unrelated_library_directory_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            # Autre soname que celui du binaire : le chargeur ne l'ouvre pas.
            (bins / "libavdevice.so.61").mkdir()

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 38),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_bundled_ffmpeg_unrelated_device_fifo_still_muxes(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _needed_libavdevice(system_ffmpeg)
        if soname is None:
            self.skipTest("soname libavdevice illisible")
        decoy = "libavdevice.so.61" if soname != "libavdevice.so.61" else "libavdevice.so.62"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            os.mkfifo(bins / decoy)

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("un tube d'un autre soname a bloqué le mux")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(fake_wan),
                ):
                    result = generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 42),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_transitive_soname_directory_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        direct = _needed_names(system_ffmpeg)
        if soname is None or direct is None:
            self.skipTest("soname transitif illisible")
        self.assertNotIn(soname, direct)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / soname).mkdir()

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 43),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("dossier", message)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("libavcodec", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_transitive_soname_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.mkfifo(bins / soname)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube du soname transitif a bloqué le contrôle ffmpeg")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 44),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)

    def test_bundled_ffmpeg_transitive_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        device = _needed_libavdevice(system_ffmpeg)
        device_path = None if device is None else _resolved_library(system_ffmpeg, device)
        if soname is None or device is None or device_path is None:
            self.skipTest("soname transitif ou libavdevice illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / device).symlink_to(device_path)
            os.mkfifo(bins / soname)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube transitif à côté du binaire a bloqué le mux d'essai")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 45),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_interpreter_directory_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        interpreter = _interpreter_basename(system_ffmpeg)
        if interpreter is None:
            self.skipTest("interpréteur illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / interpreter).mkdir()

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 46),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_hwcap_transitive_file_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            (slot / soname).write_bytes(b"pas un ELF")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 47),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(level, message)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("code 0", message)

    def test_hwcap_transitive_directory_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "glibc-hwcaps" / level / soname).mkdir(parents=True)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 48),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("dossier", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_hwcap_directory_without_readelf_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "glibc-hwcaps" / level / soname).mkdir(parents=True)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            real_which = shutil.which

            def which(name: str, *args: object, **kwargs: object) -> str | None:
                if name == "readelf":
                    return None
                return real_which(name, *args, **kwargs)

            wan2gp_runner._GLIBC_HWCAPS = None
            try:
                with (
                    mock.patch("src.wan2gp_runner.shutil.which", side_effect=which),
                    mock.patch(
                        "src.wan2gp_runner.subprocess.Popen",
                        side_effect=_popen_except_ffmpeg(refuse_gpu),
                    ),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 54),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                wan2gp_runner._GLIBC_HWCAPS = None
            message = str(caught.exception)
            self.assertIn("dossier", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(level, message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_unreadable_hwcap_file_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            hwcaps = bins / "glibc-hwcaps"
            slot = hwcaps / level
            slot.mkdir(parents=True)
            (slot / soname).write_bytes(b"pas un ELF")
            hwcaps.chmod(0o111)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 55),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                hwcaps.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(level, message)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("code 0", message)
            self.assertIn("listable", message)

    def test_unreadable_hwcap_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            hwcaps = bins / "glibc-hwcaps"
            slot = hwcaps / level
            slot.mkdir(parents=True)
            os.mkfifo(slot / soname)
            hwcaps.chmod(0o111)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube glibc-hwcaps non listable a bloqué le contrôle ffmpeg")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 56),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
                hwcaps.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("listable", message)

    def test_unreadable_hwcap_unrelated_file_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        level = _hwcap_level(system_ffmpeg)
        if level is None:
            self.skipTest("niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            hwcaps = bins / "glibc-hwcaps"
            slot = hwcaps / level
            slot.mkdir(parents=True)
            (slot / "notes.txt").write_text("rien", encoding="utf-8")
            hwcaps.chmod(0o111)
            try:
                def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                    raw = Path(command[command.index("--output-dir") + 1])
                    raw.mkdir(parents=True, exist_ok=True)
                    _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                    return _FakeProcess()

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(fake_wan),
                ):
                    result = generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 57),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                hwcaps.chmod(0o755)
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_unlisted_bundle_hwcap_file_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            (slot / soname).write_bytes(b"pas un ELF")
            bins.chmod(0o111)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 58),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                bins.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(level, message)
            self.assertIn(soname, message)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn("listable", message)
            self.assertIn("iterdir", message)
            self.assertIn("code 0", message)

    def test_unlisted_bundle_hwcap_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            os.mkfifo(slot / soname)
            bins.chmod(0o111)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube dans ffmpeg_bins non listable a bloqué le contrôle")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 59),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
                bins.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(soname, message)
            self.assertIn("listable", message)
            self.assertIn("iterdir", message)
            self.assertIn("error saving", message)

    def test_unlisted_bundle_root_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            os.mkfifo(bins / soname)
            bins.chmod(0o111)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube à la racine de ffmpeg_bins non listable a bloqué le contrôle")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 60),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
                bins.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn(soname, message)
            self.assertIn("iterdir", message)
            self.assertIn("error saving", message)
            self.assertNotIn("GPU", message)

    def test_unlisted_bundle_unrelated_file_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        level = _hwcap_level(system_ffmpeg)
        if level is None:
            self.skipTest("niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            (slot / "notes.txt").write_text("rien", encoding="utf-8")
            (bins / "notes.txt").write_text("rien", encoding="utf-8")
            bins.chmod(0o111)
            try:
                def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                    raw = Path(command[command.index("--output-dir") + 1])
                    raw.mkdir(parents=True, exist_ok=True)
                    _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                    return _FakeProcess()

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(fake_wan),
                ):
                    result = generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 61),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                bins.chmod(0o755)
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_unlisted_bundle_with_visible_libavdevice_still_muxes(self) -> None:
        """Mode 111 : glob est vide, lstat voit libavdevice, un nouveau nom échoue.

        Le lien vers le ffmpeg du système n'est pas inscriptible : copy2 ne le
        tronque pas. Le mux système réussit.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            bins.chmod(0o111)
            try:
                def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                    raw = Path(command[command.index("--output-dir") + 1])
                    raw.mkdir(parents=True, exist_ok=True)
                    _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                    return _FakeProcess()

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(fake_wan),
                ):
                    result = generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 62),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                bins.chmod(0o755)
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_unlisted_bundle_short_needed_library_stops_before_the_gpu(self) -> None:
        """Mode 111 et libavdevice visible : le soname trop court reste un échec."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        ldd = subprocess.run(
            ["ldd", system_ffmpeg],
            check=False,
            capture_output=True,
            text=True,
        )
        soname = ""
        for line in ldd.stdout.splitlines():
            name = line.strip().split(" ", 1)[0]
            if name.startswith("libavcodec.so"):
                soname = name
                break
        if not soname:
            self.skipTest("libavcodec absent du chargement de ffmpeg")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            (bins / soname).write_bytes(b"")
            bins.chmod(0o111)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 63),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                bins.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("LD_LIBRARY_PATH", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("sans libavdevice", message)

    def test_writable_unlisted_bundle_stops_before_the_gpu(self) -> None:
        """Mode 311 : non listable, mais open() peut remplacer ffmpeg."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            bins.chmod(0o311)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 64),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                bins.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("listable", message)
            self.assertIn("glob", message)
            self.assertIn("libavdevice.so.61", message)
            self.assertIn("lstat", message)
            self.assertIn("téléchargement", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("sans libavdevice", message)

    def test_unlisted_bundle_writable_binary_stops_before_the_gpu(self) -> None:
        """Mode 111 : copy2 tronque un ffmpeg déjà là, sans créer de nom."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            shutil.copy2(system_ffmpeg, bins / "ffmpeg")
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            original = (bins / "ffmpeg").read_bytes()
            bins.chmod(0o111)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 67),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                bins.chmod(0o755)
            self.assertEqual((bins / "ffmpeg").read_bytes(), original)
            message = str(caught.exception)
            self.assertIn("listable", message)
            self.assertIn("glob", message)
            self.assertIn("libavdevice.so.61", message)
            self.assertIn("lstat", message)
            self.assertIn("copy2", message)
            self.assertIn(str(bins / "ffmpeg"), message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("sans libavdevice", message)

    def test_unlisted_bundle_symlink_to_writable_binary_stops_before_the_gpu(self) -> None:
        """Mode 111 : copy2 suit le lien et écrase la cible inscriptible."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            owned = Path(directory) / "owned-ffmpeg"
            shutil.copy2(system_ffmpeg, owned)
            (bins / "ffmpeg").symlink_to(owned)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / "libavdevice.so.61").write_bytes(b"")
            original = owned.read_bytes()
            bins.chmod(0o111)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 68),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                bins.chmod(0o755)
            self.assertEqual(owned.read_bytes(), original)
            message = str(caught.exception)
            self.assertIn("copy2", message)
            self.assertIn(str(bins / "ffmpeg"), message)
            self.assertIn("libavdevice.so.61", message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)
            self.assertNotIn("sans libavdevice", message)

    def test_unlisted_bundle_without_libavdevice_stays_incomplete(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            bins.chmod(0o111)
            try:
                def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                    raise AssertionError("GPU")

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 66),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                bins.chmod(0o755)
            message = str(caught.exception)
            self.assertIn("sans libavdevice", message)
            self.assertIn("retélécharger", message)

    def test_unreadable_soname_symlink_does_not_block(self) -> None:
        """Lien de soname vers un dossier en mode 000 : le chargeur le saute."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            hidden = bins / "hidden"
            hidden.mkdir()
            (hidden / "lib.so").write_bytes(b"pas un ELF")
            (bins / soname).symlink_to(hidden / "lib.so")
            hidden.chmod(0o000)
            try:
                def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                    raw = Path(command[command.index("--output-dir") + 1])
                    raw.mkdir(parents=True, exist_ok=True)
                    _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                    return _FakeProcess()

                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(fake_wan),
                ):
                    result = generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 65),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            finally:
                hidden.chmod(0o755)
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_hwcap_transitive_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            os.mkfifo(slot / soname)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube glibc-hwcaps a bloqué le contrôle ffmpeg")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 49),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)

    def test_hwcap_null_device_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or level is None:
            self.skipTest("soname transitif ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            (slot / soname).symlink_to("/dev/null")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 50),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("périphérique", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_bundled_ffmpeg_hwcap_fifo_stops_without_opening_it(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        device = _needed_libavdevice(system_ffmpeg)
        device_path = None if device is None else _resolved_library(system_ffmpeg, device)
        level = _hwcap_level(system_ffmpeg)
        if soname is None or device is None or device_path is None or level is None:
            self.skipTest("soname transitif, libavdevice ou niveau glibc-hwcaps illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "ffmpeg").symlink_to(system_ffmpeg)
            (bins / "ffprobe").symlink_to(system_ffprobe)
            (bins / device).symlink_to(device_path)
            slot = bins / "glibc-hwcaps" / level
            slot.mkdir(parents=True)
            os.mkfifo(slot / soname)

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            def expired(_signum: int, _frame: object) -> None:
                raise TimeoutError("le tube glibc-hwcaps à côté du binaire a bloqué le mux")

            previous = signal.signal(signal.SIGALRM, expired)
            signal.alarm(15)
            try:
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 51),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            message = str(caught.exception)
            self.assertIn("tube", message)
            self.assertIn("glibc-hwcaps", message)
            self.assertIn(soname, message)
            self.assertIn("error saving", message)
            self.assertIn("code 0", message)

    def test_non_hwcap_subdirectory_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "glibc-hwcaps" / "pas-un-hwcap" / soname).mkdir(parents=True)

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 52),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_hwcap_unrelated_directory_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        level = _hwcap_level(system_ffmpeg)
        direct = _needed_names(system_ffmpeg)
        if level is None or direct is None:
            self.skipTest("niveau glibc-hwcaps ou NEEDED illisible")
        unrelated = "libnotloaded.so.9"
        self.assertNotIn(unrelated, direct)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "glibc-hwcaps" / level / unrelated).mkdir(parents=True)

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 53),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_file_directly_under_glibc_hwcaps_does_not_block(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg est requis")
        soname = _transitive_soname(system_ffmpeg)
        if soname is None:
            self.skipTest("soname transitif illisible")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (bins / "glibc-hwcaps").mkdir()
            (bins / "glibc-hwcaps" / soname).write_bytes(b"pas un ELF")

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                _encode(raw / "clip.mp4", seconds=0.2, frames=4)
                return _FakeProcess()

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 54),
                        settings=replace(config.wan, root=root),
                        output_dir=Path(directory) / "out",
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertTrue(result.raw_video.name.endswith(".mp4"))

    def test_encode_error_buried_in_the_log_is_reported(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            progress = "\n".join(f"  [{step}/30] Denoising" for step in range(1, 80))

            def fake_wan(_command: list[str], **_kwargs: object) -> _FakeProcess:
                return _FakeProcess(
                    log=(
                        "error saving /tmp/clip.mp4: libx264 encoder open failed\n"
                        + progress
                        + "\nQueue completed: 1/1 tasks\n"
                    )
                )

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 30),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("error saving", message)
            self.assertIn("libx264 encoder open failed", message)

    def test_swallowed_encode_error_is_reported(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def fake_wan(_command: list[str], **_kwargs: object) -> _FakeProcess:
                return _FakeProcess(log="error saving /tmp/clip.mp4: Error while opening encoder\n")

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 22),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("error saving", message)
            self.assertIn("Aucun fichier", message)

    def test_exit_zero_without_a_file_is_a_failure(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def fake_wan(_command: list[str], **_kwargs: object) -> _FakeProcess:
                return _FakeProcess(log="  [SKIP] Task 1 failed validation: no model\nQueue completed: 0/1 tasks (1 skipped)\n")

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 18),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Aucun fichier", message)
            self.assertIn("[SKIP]", message)

    def test_proc_cwd_config_container_stops_before_the_gpu(self) -> None:
        """``open("wgp_config.json")`` suit le cwd du clone, pas celui du pipeline.

        Le lien vise ``/proc/self/cwd/secret.json``,
        ``/proc/net/../cwd/secret.json`` ou ``hop1/../cwd/secret.json``
        (``hop1`` → ``hop2`` → ``/proc/net``). Depuis le pipeline, ``is_file``
        est faux, ou le leurre est un ``mp4``. Le clone a un ``mkv`` (ou un
        codec ProRes). WanGP écrit ce conteneur et sort 0, sans MP4.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = ("self-missing", "proc-net-decoy", "hop-decoy", "self-prores")
        for shape in shapes:
            with self.subTest(shape=shape):
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
                    if shape == "self-prores":
                        hidden = {
                            "video_container": "mp4",
                            "video_output_codec": "prores_422",
                        }
                    else:
                        hidden = {
                            "video_container": "mkv",
                            "video_output_codec": "libx264_8",
                        }
                    (root / "secret.json").write_text(
                        json.dumps(hidden) + "\n",
                        encoding="utf-8",
                    )
                    if shape == "hop-decoy":
                        (root / "hop2").symlink_to("/proc/net")
                        (root / "hop1").symlink_to("hop2")
                        (root / "wgp_config.json").symlink_to("hop1/../cwd/secret.json")
                    elif shape == "proc-net-decoy":
                        (root / "wgp_config.json").symlink_to("/proc/net/../cwd/secret.json")
                    else:
                        (root / "wgp_config.json").symlink_to("/proc/self/cwd/secret.json")
                    decoy = base / "pipeline"
                    decoy.mkdir()
                    if shape in {"proc-net-decoy", "hop-decoy"}:
                        (decoy / "secret.json").write_text(
                            '{"video_container": "mp4", "video_output_codec": "libx264_8"}\n',
                            encoding="utf-8",
                        )
                    try:
                        os.chdir(decoy)
                        link = root / "wgp_config.json"
                        if shape in {"self-missing", "self-prores"}:
                            self.assertFalse(link.is_file())
                        else:
                            self.assertTrue(link.is_file())
                            self.assertIn("mp4", link.read_text(encoding="utf-8"))
                            self.assertNotIn("mkv", link.read_text(encoding="utf-8"))

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 71),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn("/proc/self", message)
                        self.assertIn("secret.json", message)
                        self.assertIn("code 0", message)
                        if shape == "self-prores":
                            self.assertIn("ProRes", message)
                        else:
                            self.assertIn("mkv", message)
                            self.assertIn("video_container", message)
                    finally:
                        os.chdir(previous)

    def test_proc_cwd_config_ignores_a_pipeline_container(self) -> None:
        """Un ``mkv`` du cwd du pipeline n'est pas le ``wgp_config.json`` du clone.

        Le leurre rend ``is_file`` vrai ici. Le clone n'a pas ce fichier.
        WanGP écrit la config par défaut, conteneur ``mp4``.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            (root / "wgp_config.json").symlink_to("/proc/self/cwd/secret.json")
            decoy = base / "pipeline"
            decoy.mkdir()
            (decoy / "secret.json").write_text(
                '{"video_container": "mkv", "video_output_codec": "libx264_8"}\n',
                encoding="utf-8",
            )
            try:
                os.chdir(decoy)
                link = root / "wgp_config.json"
                self.assertTrue(link.is_file())
                self.assertIn("mkv", link.read_text(encoding="utf-8"))
                self.assertFalse((root / "secret.json").exists())
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)

    def test_proc_cwd_model_def_stops_before_the_gpu(self) -> None:
        """``defaults`` et ``finetunes`` sont lus depuis le cwd du clone.

        ``image_outputs`` force ``image_mode`` à 1 même si la file demande 0.
        WanGP écrit un JPEG et sort 0. Un lien vers ``/proc/self`` n'est pas
        le fichier que ``is_file`` ou ``iterdir`` voient depuis le pipeline.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = (
            "plain",
            "file-self",
            "file-proc-net",
            "file-hop",
            "dir-self",
            "dir-proc-net",
            "dir-only",
            "finetune-self",
            "audio-plain",
            "mesh-plain",
            "world-plain",
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
                    flag = {
                        "audio-plain": "audio_only",
                        "mesh-plain": "mesh_outputs",
                        "world-plain": "world_outputs",
                    }.get(shape, "image_outputs")
                    hidden = json.dumps({"model": {flag: True}}) + "\n"
                    decoy_text = json.dumps({"model": {flag: False}}) + "\n"
                    defaults = root / "defaults"
                    finetunes = root / "finetunes"
                    if shape in {"plain", "audio-plain", "mesh-plain", "world-plain"}:
                        defaults.mkdir()
                        (defaults / "t2v_1.3B.json").write_text(hidden, encoding="utf-8")
                    elif shape == "file-self":
                        defaults.mkdir()
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (defaults / "t2v_1.3B.json").symlink_to("/proc/self/cwd/secret.json")
                    elif shape == "file-proc-net":
                        defaults.mkdir()
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (defaults / "t2v_1.3B.json").symlink_to("/proc/net/../cwd/secret.json")
                    elif shape == "file-hop":
                        defaults.mkdir()
                        (defaults / "hop2").symlink_to("/proc/net")
                        (defaults / "hop1").symlink_to("hop2")
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (defaults / "t2v_1.3B.json").symlink_to("hop1/../cwd/secret.json")
                    elif shape == "dir-self":
                        (root / "t2v_1.3B.json").write_text(hidden, encoding="utf-8")
                        defaults.symlink_to("/proc/self/cwd")
                    elif shape == "dir-proc-net":
                        (root / "t2v_1.3B.json").write_text(hidden, encoding="utf-8")
                        defaults.symlink_to("/proc/net/../cwd")
                    elif shape == "dir-only":
                        only = root / "only"
                        only.mkdir()
                        (only / "t2v_1.3B.json").write_text(hidden, encoding="utf-8")
                        defaults.symlink_to("/proc/self/cwd/only")
                    elif shape == "finetune-self":
                        defaults.mkdir()
                        (defaults / "t2v_1.3B.json").write_text(
                            json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                            encoding="utf-8",
                        )
                        finetunes.mkdir()
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (finetunes / "t2v_1.3B.json").symlink_to("/proc/self/cwd/secret.json")
                    else:
                        raise AssertionError(shape)
                    pipeline = base / "pipeline"
                    pipeline.mkdir()
                    if shape in {"file-proc-net", "file-hop", "dir-self", "dir-proc-net"}:
                        (pipeline / ("t2v_1.3B.json" if shape.startswith("dir-") else "secret.json")).write_text(
                            decoy_text,
                            encoding="utf-8",
                        )
                    try:
                        os.chdir(pipeline)
                        if shape == "file-self":
                            self.assertFalse((defaults / "t2v_1.3B.json").is_file())
                        elif shape in {"file-proc-net", "file-hop"}:
                            seen = (defaults / "t2v_1.3B.json").read_text(encoding="utf-8")
                            self.assertIn("false", seen)
                            self.assertNotIn("true", seen)
                        elif shape in {"dir-self", "dir-proc-net"}:
                            seen = next(defaults.glob("t2v_1.3B.json")).read_text(encoding="utf-8")
                            self.assertIn("false", seen)
                        elif shape == "dir-only":
                            self.assertFalse((pipeline / "only").exists())

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 72),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn(flag, message)
                        self.assertIn("code 0", message)
                        if shape == "finetune-self":
                            self.assertIn("finetunes/t2v_1.3B.json", message)
                        else:
                            self.assertIn("defaults/t2v_1.3B.json", message)
                        if flag == "image_outputs":
                            self.assertIn("JPEG", message)
                        if shape not in {"plain", "audio-plain", "mesh-plain", "world-plain"}:
                            self.assertIn("/proc/self", message)
                        if shape in {"file-self", "file-proc-net", "file-hop", "finetune-self"}:
                            self.assertIn("secret.json", message)
                    finally:
                        os.chdir(previous)

    def test_proc_cwd_model_def_ignores_a_pipeline_image_outputs(self) -> None:
        """Un ``image_outputs`` du cwd du pipeline n'est pas la définition du clone.

        Le leurre rend ``is_file`` vrai ici. Le clone n'a pas ce fichier.
        WanGP garde la définition sans le drapeau et peut écrire un MP4.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            defaults.mkdir()
            (defaults / "t2v_1.3B.json").symlink_to("/proc/self/cwd/secret.json")
            pipeline = base / "pipeline"
            pipeline.mkdir()
            (pipeline / "secret.json").write_text(
                json.dumps({"model": {"image_outputs": True}}) + "\n",
                encoding="utf-8",
            )
            try:
                os.chdir(pipeline)
                link = defaults / "t2v_1.3B.json"
                self.assertTrue(link.is_file())
                self.assertIn("true", link.read_text(encoding="utf-8"))
                self.assertFalse((root / "secret.json").exists())
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)

    def test_finetune_can_clear_image_outputs(self) -> None:
        """Le JSON ``finetunes`` arrive après ``defaults`` et peut retirer le drapeau."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": True, "architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": False}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_plugin_model_defaults_stop_before_the_gpu(self) -> None:
        """Le ``defaults`` d'un plugin activé est fusionné avant ``defaults/``.

        ``os.path.abspath`` le rend absolu. Le ``t2v_1.3B`` officiel ne
        réécrit pas ``image_outputs``, donc le drapeau du plugin reste.
        WanGP force ``image_mode`` à 1, écrit un JPEG et sort 0.
        ``/proc/self`` n'est pas le fichier que ``is_file`` voit ici.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        shapes = (
            "plain",
            "file-self",
            "file-proc-net",
            "file-hop",
            "dir-only",
            "dir-self",
            "meta-self",
            "meta-proc-net",
            "meta-hop",
            "defaults-proc",
            "system",
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                previous = Path.cwd()
                with tempfile.TemporaryDirectory() as directory:
                    base = Path(directory)
                    root = base / "wan"
                    root.mkdir()
                    (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
                    defaults = root / "defaults"
                    defaults.mkdir()
                    (defaults / "t2v_1.3B.json").write_text(
                        json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                        encoding="utf-8",
                    )
                    hidden = json.dumps({"model": {"image_outputs": True}}) + "\n"
                    decoy_text = json.dumps({"model": {"image_outputs": False}}) + "\n"
                    plugin_id = "guides" if shape == "system" else "capy"
                    info = {
                        "model_handlers": [f"{plugin_id}.models"],
                        "defaults": "/proc/self/cwd/bag" if shape == "defaults-proc" else "./defs",
                        "profiles": "./profiles",
                    }
                    if shape == "dir-self":
                        plugin = root / plugin_id
                    elif shape == "dir-only":
                        plugin = root / "only" / plugin_id
                    else:
                        plugin = root / "plugins" / plugin_id
                    defs = plugin / "defs"
                    defs.mkdir(parents=True)
                    (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
                    (plugin / "plugin_info.json").write_text(
                        json.dumps(info) + "\n",
                        encoding="utf-8",
                    )
                    if shape == "defaults-proc":
                        bag = root / "bag"
                        bag.mkdir()
                        (bag / "t2v_1.3B.json").write_text(hidden, encoding="utf-8")
                    elif shape not in {"file-self", "file-proc-net", "file-hop"}:
                        (defs / "t2v_1.3B.json").write_text(hidden, encoding="utf-8")
                    if shape == "system":
                        pass
                    else:
                        (root / "wgp_config.json").write_text(
                            json.dumps(
                                {
                                    "enabled_plugins": [plugin_id],
                                    "video_container": "mp4",
                                    "video_output_codec": "libx264_8",
                                }
                            )
                            + "\n",
                            encoding="utf-8",
                        )
                    if shape == "file-self":
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (defs / "t2v_1.3B.json").symlink_to("/proc/self/cwd/secret.json")
                    elif shape == "file-proc-net":
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (defs / "t2v_1.3B.json").symlink_to("/proc/net/../cwd/secret.json")
                    elif shape == "file-hop":
                        (defs / "hop2").symlink_to("/proc/net")
                        (defs / "hop1").symlink_to("hop2")
                        (root / "secret.json").write_text(hidden, encoding="utf-8")
                        (defs / "t2v_1.3B.json").symlink_to("hop1/../cwd/secret.json")
                    elif shape == "dir-only":
                        (root / "plugins").symlink_to("/proc/self/cwd/only")
                    elif shape == "dir-self":
                        (root / "plugins").symlink_to("/proc/self/cwd")
                    elif shape == "meta-self":
                        (root / "meta.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
                        (plugin / "plugin_info.json").unlink()
                        (plugin / "plugin_info.json").symlink_to("/proc/self/cwd/meta.json")
                    elif shape == "meta-proc-net":
                        (root / "meta.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
                        (plugin / "plugin_info.json").unlink()
                        (plugin / "plugin_info.json").symlink_to("/proc/net/../cwd/meta.json")
                    elif shape == "meta-hop":
                        (root / "meta.json").write_text(json.dumps(info) + "\n", encoding="utf-8")
                        (plugin / "hop2").symlink_to("/proc/net")
                        (plugin / "hop1").symlink_to("hop2")
                        (plugin / "plugin_info.json").unlink()
                        (plugin / "plugin_info.json").symlink_to("hop1/../cwd/meta.json")
                    pipeline = base / "pipeline"
                    pipeline.mkdir()
                    if shape in {"file-proc-net", "file-hop"}:
                        (pipeline / "secret.json").write_text(decoy_text, encoding="utf-8")
                    elif shape == "defaults-proc":
                        decoy_bag = pipeline / "bag"
                        decoy_bag.mkdir()
                        (decoy_bag / "t2v_1.3B.json").write_text(decoy_text, encoding="utf-8")
                    elif shape in {"meta-self", "meta-proc-net", "meta-hop"}:
                        (pipeline / "meta.json").write_text(
                            json.dumps({"name": "leurre"}) + "\n",
                            encoding="utf-8",
                        )
                    try:
                        os.chdir(pipeline)
                        if shape == "file-self":
                            self.assertFalse((defs / "t2v_1.3B.json").is_file())
                        elif shape in {"file-proc-net", "file-hop"}:
                            seen = (defs / "t2v_1.3B.json").read_text(encoding="utf-8")
                            self.assertIn("false", seen)
                            self.assertNotIn("true", seen)
                        elif shape == "dir-self":
                            self.assertFalse((root / "plugins" / plugin_id).exists())
                        elif shape == "dir-only":
                            self.assertFalse((pipeline / "only").exists())
                            self.assertFalse((root / "plugins").is_dir())
                        elif shape == "defaults-proc":
                            seen = Path("/proc/self/cwd/bag/t2v_1.3B.json").read_text(encoding="utf-8")
                            self.assertIn("false", seen)
                            self.assertNotIn("true", seen)
                        elif shape in {"meta-self", "meta-proc-net", "meta-hop"}:
                            seen = (plugin / "plugin_info.json").read_text(encoding="utf-8")
                            self.assertNotIn("model_handlers", seen)

                        def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                            raise AssertionError("GPU")

                        with mock.patch(
                            "src.wan2gp_runner.subprocess.Popen",
                            side_effect=_popen_except_ffmpeg(refuse_gpu),
                        ):
                            with self.assertRaises(Wan2GPError) as caught:
                                generate(
                                    GenerationRequest(
                                        prompt=build_prompt(theme, 73),
                                        settings=replace(config.wan, root=root),
                                        output_dir=base / "out",
                                        aspect="9:16",
                                        dry_run=False,
                                    )
                                )
                        message = str(caught.exception)
                        self.assertIn("image_outputs", message)
                        self.assertIn("JPEG", message)
                        self.assertIn("code 0", message)
                        if shape != "defaults-proc":
                            self.assertIn(plugin_id, message)
                        if shape in {
                            "file-self",
                            "file-proc-net",
                            "file-hop",
                            "dir-only",
                            "dir-self",
                            "meta-self",
                            "meta-proc-net",
                            "meta-hop",
                            "defaults-proc",
                        }:
                            self.assertIn("/proc/self", message)
                        if shape in {"file-self", "file-proc-net", "file-hop"}:
                            self.assertIn("secret.json", message)
                        if shape == "dir-only":
                            self.assertIn("only", message)
                        if shape == "defaults-proc":
                            self.assertIn("bag", message)
                        if shape in {"meta-self", "meta-proc-net", "meta-hop"}:
                            self.assertIn("meta.json", message)
                    finally:
                        os.chdir(previous)

    def test_plugin_proc_defaults_ignore_a_pipeline_file(self) -> None:
        """Un ``image_outputs`` seulement dans le cwd du pipeline n'est pas chargé.

        ``defaults`` vaut ``/proc/self/cwd/bag``. Le leurre rend le JSON
        lisible ici. Le clone n'a pas ce dossier. WanGP ne fusionne rien
        et peut écrire un MP4.
        """
        system_ffmpeg = shutil.which("ffmpeg")
        system_ffprobe = shutil.which("ffprobe")
        if system_ffmpeg is None or system_ffprobe is None:
            self.skipTest("ffmpeg est requis")
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            plugin = root / "plugins" / "capy"
            plugin.mkdir(parents=True)
            (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
            (plugin / "plugin_info.json").write_text(
                json.dumps(
                    {
                        "model_handlers": ["capy.models"],
                        "defaults": "/proc/self/cwd/bag",
                        "profiles": "./profiles",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "wgp_config.json").write_text(
                json.dumps(
                    {
                        "enabled_plugins": ["capy"],
                        "video_container": "mp4",
                        "video_output_codec": "libx264_8",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pipeline = base / "pipeline"
            bag = pipeline / "bag"
            bag.mkdir(parents=True)
            (bag / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": True}}) + "\n",
                encoding="utf-8",
            )
            try:
                os.chdir(pipeline)
                self.assertIn("true", Path("/proc/self/cwd/bag/t2v_1.3B.json").read_text(encoding="utf-8"))
                self.assertFalse((root / "bag").exists())
                binaries = wan2gp_runner.require_ffmpeg(root)
                self.assertEqual(Path(binaries.ffmpeg).resolve(), Path(system_ffmpeg).resolve())
            finally:
                os.chdir(previous)

    def test_disabled_plugin_model_defaults_are_not_read(self) -> None:
        """Un plugin absent de ``enabled_plugins`` et non système est ignoré."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            plugin = root / "plugins" / "capy"
            defs = plugin / "defs"
            defs.mkdir(parents=True)
            (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
            (plugin / "plugin_info.json").write_text(
                json.dumps(
                    {
                        "model_handlers": ["capy.models"],
                        "defaults": "./defs",
                        "profiles": "./profiles",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (defs / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": True}}) + "\n",
                encoding="utf-8",
            )
            (root / "wgp_config.json").write_text(
                json.dumps({"enabled_plugins": []}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_finetune_clears_a_plugin_image_outputs(self) -> None:
        """``finetunes/*.json`` est trié après le chemin absolu du plugin."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            plugin = root / "plugins" / "capy"
            defs = plugin / "defs"
            defs.mkdir(parents=True)
            finetunes = root / "finetunes"
            finetunes.mkdir(parents=True)
            (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
            (plugin / "plugin_info.json").write_text(
                json.dumps(
                    {
                        "model_handlers": ["capy.models"],
                        "defaults": "./defs",
                        "profiles": "./profiles",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (defs / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": True}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": False}}) + "\n",
                encoding="utf-8",
            )
            (root / "wgp_config.json").write_text(
                json.dumps({"enabled_plugins": ["capy"]}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_defaults_keep_image_outputs_set_after_a_plugin(self) -> None:
        """Le dossier ``defaults`` officiel passe après le plugin et réécrit la clé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            plugin = root / "plugins" / "capy"
            defs = plugin / "defs"
            defs.mkdir(parents=True)
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
            (plugin / "plugin_info.json").write_text(
                json.dumps(
                    {
                        "model_handlers": ["capy.models"],
                        "defaults": "./defs",
                        "profiles": "./profiles",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (defs / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": False}}) + "\n",
                encoding="utf-8",
            )
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": True}}) + "\n",
                encoding="utf-8",
            )
            (root / "wgp_config.json").write_text(
                json.dumps({"enabled_plugins": ["capy"]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("defaults/t2v_1.3B.json", message)
            self.assertIn("image_outputs", message)

    def test_first_json_architecture_forces_image_outputs(self) -> None:
        """``init_model_def`` ne voit que le premier JSON.

        Un plugin activé est trié avant ``defaults/``. Son architecture
        ``chrono_edit`` fait poser ``image_outputs`` par ``query_model_def``
        sans que la clé soit écrite. Le defaults officiel de ``t2v_1.3B``
        remet l'architecture mais omet le drapeau : il reste vrai. WanGP
        force ``image_mode`` à 1, écrit un JPEG et sort 0.
        """
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            plugin = root / "plugins" / "capy"
            defs = plugin / "defs"
            defs.mkdir(parents=True)
            defaults = root / "defaults"
            defaults.mkdir()
            (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
            (plugin / "plugin_info.json").write_text(
                json.dumps(
                    {
                        "model_handlers": ["capy.models"],
                        "defaults": "./defs",
                        "profiles": "./profiles",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (defs / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "chrono_edit", "name": "leurre"}}) + "\n",
                encoding="utf-8",
            )
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "t2v_1.3B",
                            "description": "officiel",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "wgp_config.json").write_text(
                json.dumps({"enabled_plugins": ["capy"], "video_container": "mp4"}) + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("image_outputs", message)
            self.assertIn("JPEG", message)
            self.assertIn("code 0", message)
            self.assertIn("chrono_edit", message)
            self.assertIn("query_model_def", message)
            self.assertIn("capy", message)

    def test_later_chrono_architecture_forces_image_mode(self) -> None:
        """L'architecture finale passe à ``validate_generative_settings``.

        ``init_model_def`` a déjà tourné sur ``t2v_1.3B`` : ``image_outputs``
        reste faux. ``get_base_model_type`` renvoie pourtant ``chrono_edit``.
        ``model_mode`` vaut ``null`` (``models/_settings.json``). Le handler
        fait ``image_mode = 1``, écrit un JPEG et ``process_tasks_cli`` sort 0.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {"model": {"architecture": "chrono_edit", "image_outputs": False}}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("validate_generative_settings", message)
            self.assertIn("chrono_edit", message)
            self.assertIn("JPEG", message)
            self.assertIn("model_mode=None", message)
            self.assertIn("finetunes/t2v_1.3B.json", message)
            self.assertIn("image_outputs peut rester faux", message)
            self.assertNotIn("La clé est absente", message)

    def test_chrono_model_mode_two_keeps_a_video(self) -> None:
        """``model_mode`` 2 laisse ``image_mode`` à 0 : WanGP écrit une vidéo.

        La clé est hors de ``model``. ``refresh_model_defs`` la range dans
        ``settings``. ``clean_settings`` la garde si la file ne l'envoie pas.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "chrono_edit"},
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_model_mode=0,
                )
            self.assertIn("model_mode=0", str(caught.exception))

    def test_queue_model_mode_two_overrides_a_jpeg_setting(self) -> None:
        """Une file qui envoie ``model_mode`` 2 gagne sur le JSON."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "image_prompt_types_allowed": "",
                        },
                        "model_mode": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_model_mode=2,
            )

    def test_image_modes_without_zero_clears_model_mode_two(self) -> None:
        """``model_mode`` 2 ne tient pas si ``image_modes`` cache le mode 0.

        ``validate_settings`` remplace ``[0, 1, 2]`` par ``image_modes``.
        La file demande ``image_mode`` 0. S'il n'est pas dans la liste,
        ``model_mode`` 2 devient None avant ``validate_generative_settings``.
        ``chrono_edit`` écrit alors un JPEG et le CLI sort 0. ``image_outputs``
        peut rester faux. ``model_modes`` posé sur le premier JSON survit
        au finetune qui ne réécrit pas la clé.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "model_modes": {
                                "label": "Chrono Edit Process",
                                "default": 2,
                                "image_modes": [1, 2],
                            },
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "chrono_edit", "image_outputs": False},
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("validate_settings", message)
            self.assertIn("image_modes", message)
            self.assertIn("model_mode=2", message)
            self.assertIn("None", message)
            self.assertIn("JPEG", message)
            self.assertIn("code 0", message)
            self.assertIn("defaults/t2v_1.3B.json", message)
            self.assertIn("chrono_edit", message)

    def test_image_modes_including_zero_keeps_model_mode_two(self) -> None:
        """``image_modes`` qui contient 0 laisse ``model_mode`` 2 écrire une vidéo."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "image_prompt_types_allowed": "",
                            "model_modes": {"image_modes": [0]},
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_model_mode_two_cleared_by_image_modes(self) -> None:
        """La file envoie 2, mais ``image_modes`` vide l'annule avant le handler."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            finetunes = root / "finetunes"
            finetunes.mkdir(parents=True)
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "model_modes": {"image_modes": []},
                        },
                        "model_mode": 0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_model_mode=2,
                )
            message = str(caught.exception)
            self.assertIn("image_modes", message)
            self.assertIn("finetunes/t2v_1.3B.json", message)
            self.assertIn("model_mode=2", message)

    def test_non_iterable_image_modes_is_not_a_silent_jpeg(self) -> None:
        """Un ``image_modes`` entier lève TypeError dans WanGP : pas un succès silencieux.

        ``0 not in 5`` n'est pas le chemin qui écrit un JPEG et sort 0.
        ``model_mode`` 2 reste donc accepté ici.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "model_modes": {"image_modes": 5},
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_chrono_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le finetune force le JPEG."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir()
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "t2v_1.3B",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "chrono_edit"}}) + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("validate_generative_settings", message)
            self.assertIn("chrono_edit", message)
            self.assertIn("code 0", message)

    def test_later_qwen21_architecture_forces_image_mode(self) -> None:
        """L'architecture finale passe à ``fix_settings``, pas seulement au premier JSON.

        ``init_model_def`` a déjà tourné sur ``t2v_1.3B`` : ``image_outputs``
        reste absent. ``get_base_model_type`` renvoie pourtant
        ``qwen_image_21_7B``. ``clean_settings`` applique ``image_mode`` 0
        de la file, puis ``models/qwen21/qwen21_handler.py`` le remet à 1.
        Un JPEG est écrit et ``process_tasks_cli`` sort 0.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "qwen_image_21_7B"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("fix_settings", message)
            self.assertIn("qwen21_handler.py", message)
            self.assertIn("qwen_image_21_7B", message)
            self.assertIn("JPEG", message)
            self.assertIn("code 0", message)
            self.assertIn("finetunes/t2v_1.3B.json", message)
            self.assertIn("image_outputs peut rester faux", message)
            self.assertNotIn("La clé est absente", message)

    def test_explicit_false_still_forced_by_qwen21_fix_settings(self) -> None:
        """``image_outputs`` faux n'empêche pas ``fix_settings`` de Qwen 2.1."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "qwen_image_21_7B",
                            "image_outputs": False,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("fix_settings", message)
            self.assertIn("qwen_image_21_7B", message)
            self.assertIn("JPEG", message)
            self.assertNotIn("La clé est absente", message)

    def test_qwen21_cleared_back_to_t2v_still_writes_a_video(self) -> None:
        """Revenir à ``t2v_1.3B`` et effacer ``image_outputs`` laisse la vidéo.

        ``fix_settings`` suit l'architecture finale. Celle de Wan ne remet
        pas ``image_mode`` à 1.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "qwen_image_21_7B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "image_outputs": False,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_qwen21_image_outputs_keeps_the_flag_error(self) -> None:
        """Un ``image_outputs`` vrai reste l'erreur du drapeau, pas ``fix_settings``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "qwen_image_21_7B",
                            "image_outputs": True,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("image_outputs", message)
            self.assertNotIn("fix_settings", message)

    def test_later_qwen21_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le finetune force le JPEG de Qwen 2.1."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir()
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "t2v_1.3B",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "qwen_image_21_7B"}}) + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("fix_settings", message)
            self.assertIn("qwen_image_21_7B", message)
            self.assertIn("code 0", message)

    def test_explicit_false_clears_a_handler_image_outputs(self) -> None:
        """Un JSON suivant qui écrit ``image_outputs`` à faux gagne sur le handler."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            plugin = root / "plugins" / "capy"
            defs = plugin / "defs"
            defs.mkdir(parents=True)
            finetunes = root / "finetunes"
            finetunes.mkdir()
            (plugin / "plugin.py").write_text("# plugin\n", encoding="utf-8")
            (plugin / "plugin_info.json").write_text(
                json.dumps(
                    {
                        "model_handlers": ["capy.models"],
                        "defaults": "./defs",
                        "profiles": "./profiles",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (defs / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "chrono_edit"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"image_outputs": False, "architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (root / "wgp_config.json").write_text(
                json.dumps({"enabled_plugins": ["capy"]}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_lone_finetune_architecture_stops_before_the_gpu(self) -> None:
        """Sans defaults, le finetune est le premier JSON et le handler s'applique."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            finetunes = root / "finetunes"
            finetunes.mkdir(parents=True)
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "flux"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("flux", message)
            self.assertIn("image_outputs", message)
            self.assertIn("JPEG", message)

    def test_handler_audio_architecture_stops_before_the_gpu(self) -> None:
        """``chatterbox`` écrit un fichier audio, pas un MP4, même sans la clé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "chatterbox", "name": "voix"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("audio_only", message)
            self.assertIn("chatterbox", message)
            self.assertIn("code 0", message)

    def test_mkv_container_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "wgp_config.json").write_text(
                '{"video_container": "mkv", "video_output_codec": "libx264_8"}\n',
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 24),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("video_container", message)
            self.assertIn("mkv", message)

    def test_null_video_container_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            for raw in (None, "", " mp4"):
                (root / "wgp_config.json").write_text(
                    json.dumps({"video_container": raw, "video_output_codec": "libx264_8"}) + "\n",
                    encoding="utf-8",
                )
                with mock.patch(
                    "src.wan2gp_runner.subprocess.Popen",
                    side_effect=_popen_except_ffmpeg(refuse_gpu),
                ):
                    with self.assertRaises(Wan2GPError) as caught:
                        generate(
                            GenerationRequest(
                                prompt=build_prompt(theme, 27),
                                settings=replace(config.wan, root=root),
                                output_dir=Path(directory) / "out",
                                aspect="9:16",
                                dry_run=False,
                            )
                        )
                message = str(caught.exception)
                self.assertIn("video_container", message)
                self.assertIn(".mp4", message)

    def test_prores_codec_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            (root / "wgp_config.json").write_text(
                '{"video_container": "mp4", "video_output_codec": "prores_422"}\n',
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 25),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            self.assertIn("ProRes", str(caught.exception))

    def test_non_mp4_output_is_a_failure(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")

            def fake_wan(command: list[str], **_kwargs: object) -> _FakeProcess:
                raw = Path(command[command.index("--output-dir") + 1])
                raw.mkdir(parents=True, exist_ok=True)
                (raw / "clip.mkv").write_bytes(b"not a video")
                return _FakeProcess(log="Queue completed: 1/1 tasks\n")

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=_popen_except_ffmpeg(fake_wan)):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 26),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("clip.mkv", message)
            self.assertIn("pas un MP4", message)
            self.assertNotIn("Aucun fichier", message)

    def test_broken_ffmpeg_bins_stop_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            bins = root / "ffmpeg_bins"
            bins.mkdir(parents=True)
            (root / "wgp.py").write_text("# faux wan\n", encoding="utf-8")
            for name in ("ffmpeg", "ffprobe"):
                binary = bins / name
                binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                binary.chmod(0o755)
            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 15),
                            settings=replace(config.wan, root=root),
                            output_dir=Path(directory) / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            self.assertIn("libx264", str(caught.exception))
            self.assertIn("ffmpeg_bins", str(caught.exception))

    def test_real_run_without_install_stops_before_the_gpu(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Wan2GPError):
                generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 1),
                        settings=config.wan,
                        output_dir=Path(directory),
                        aspect="9:16",
                        dry_run=False,
                    )
                )

    def test_pipeline_dry_run_writes_a_manifest(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        with tempfile.TemporaryDirectory() as directory:
            scoped = replace(config, output_dir=Path(directory))
            result = run(RunOptions(theme_id="sunset_train", seed=11, dry_run=True), config=scoped)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(result.dry_run)
            self.assertIsNone(result.video_path)
            self.assertEqual(manifest["seed"], 11)
            self.assertIn("capybara", manifest["prompt"])
            self.assertIn("golden hour", manifest["prompt"].lower())
            self.assertIn("wgp.py", manifest["command"])
            self.assertIn("--verbose", manifest["command"])
            self.assertIn("2", manifest["command"])
            self.assertTrue(any(str(part).endswith(".queue.zip") for part in manifest["command"]))

    def test_injected_image_modes_clear_chrono_model_mode_two(self) -> None:
        """Le premier ``query_model_def`` laisse ``image_modes`` ``[2]``.

        ``init_model_def`` n'est pas rappelé. Le finetune passe à
        ``chrono_edit``, met ``image_outputs`` à faux et ``model_mode`` à 2,
        sans écrire ``model_modes``. WanGP garde ``[2]``. ``0`` n'y est pas,
        ``model_mode`` devient None, un JPEG est écrit et le CLI sort 0.
        """
        injecting = (
            "qwen_image_21_7B",
            "qwen_image_20B",
            "qwen_image_edit_20B",
            "qwen_image_edit_plus_20B",
            "qwen_image_edit_plus2_20B",
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
        )
        for architecture in injecting:
            with self.subTest(architecture=architecture):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "wan"
                    defaults = root / "defaults"
                    finetunes = root / "finetunes"
                    defaults.mkdir(parents=True)
                    finetunes.mkdir()
                    (defaults / "t2v_1.3B.json").write_text(
                        json.dumps({"model": {"architecture": architecture}}) + "\n",
                        encoding="utf-8",
                    )
                    (finetunes / "t2v_1.3B.json").write_text(
                        json.dumps(
                            {
                                "model": {
                                    "architecture": "chrono_edit",
                                    "image_outputs": False,
                                },
                                "model_mode": 2,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(Wan2GPError) as caught:
                        wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                    message = str(caught.exception)
                    self.assertIn("injecte", message)
                    self.assertIn("image_modes [2]", message)
                    self.assertIn(architecture, message)
                    self.assertIn("defaults/t2v_1.3B.json", message)
                    self.assertIn("model_mode=2", message)
                    self.assertIn("None", message)
                    self.assertIn("JPEG", message)
                    self.assertIn("code 0", message)
                    self.assertIn("chrono_edit", message)
                    self.assertNotIn("La clé est absente", message)
                    self.assertNotIn("fix_settings", message)

    def test_architectures_without_injected_image_modes_keep_video(self) -> None:
        """Sans injection, ``model_mode`` 2 et ``chrono_edit`` écrivent une vidéo.

        ``pi_flux2`` a ``supports_inpaint`` mais pas ``model_modes``.
        ``qwen_image_layered_20B`` et les Flux hors inpaint non plus.
        """
        plain = (
            "t2v_1.3B",
            "pi_flux2",
            "qwen_image_layered_20B",
            "flux_chroma",
            "flux_chroma_radiance",
            "flux_dev_umo",
            "flux_dev_uso",
            "flux_dev_kontext_dreamomni2",
            "flux3_klein",
        )
        for architecture in plain:
            with self.subTest(architecture=architecture):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory) / "wan"
                    defaults = root / "defaults"
                    finetunes = root / "finetunes"
                    defaults.mkdir(parents=True)
                    finetunes.mkdir()
                    (defaults / "t2v_1.3B.json").write_text(
                        json.dumps({"model": {"architecture": architecture}}) + "\n",
                        encoding="utf-8",
                    )
                    (finetunes / "t2v_1.3B.json").write_text(
                        json.dumps(
                            {
                                "model": {
                                    "architecture": "chrono_edit",
                                    "image_outputs": False,
                                },
                                "model_mode": 2,
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_rewritten_model_modes_drop_the_injected_list(self) -> None:
        """Un JSON qui contient ``model_modes`` remplace l'injection.

        Sans la clé ``image_modes``, ``validate_settings`` garde ``[0, 1, 2]``.
        ``model_mode`` 2 écrit une vidéo.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "qwen_image_21_7B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "model_modes": {"label": "Inpainting Method"},
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_first_json_model_modes_replace_the_query_injection(self) -> None:
        """Le ``update`` de ``init_model_def`` remplace ``model_modes`` tout de suite.

        Le finetune ne réécrit pas la clé. ``image_modes`` est déjà parti
        avec le premier JSON : ``model_mode`` 2 tient.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "flux",
                            "model_modes": {"label": "Inpainting Method"},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_image_modes_win_over_the_injection(self) -> None:
        """La liste écrite par le finetune est celle de WanGP, pas l'injection."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "krea2_raw"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "model_modes": {"image_modes": [1]},
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("finetunes/t2v_1.3B.json", message)
            self.assertIn("image_modes", message)
            self.assertNotIn("injecte", message)

    def test_queue_model_mode_two_cleared_by_injected_image_modes(self) -> None:
        """La file envoie 2. L'injection du premier JSON l'annule quand même."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "qwen_image_edit_20B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_model_mode=2,
                )
            message = str(caught.exception)
            self.assertIn("injecte", message)
            self.assertIn("qwen_image_edit_20B", message)
            self.assertIn("model_mode=2", message)

    def test_injected_image_modes_stop_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand l'injection efface ``model_mode`` 2."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir()
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "qwen_image_21_7B",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("injecte", message)
            self.assertIn("qwen_image_21_7B", message)
            self.assertIn("code 0", message)

    def test_fun_inp_start_image_is_skipped(self) -> None:
        """``fun_inp_1.3B`` injecte ``SEVL`` et exige une image de départ.

        ``update_default_settings`` pose ``image_prompt_type`` à ``S``.
        La file n'a pas d'image. ``validate_settings`` échoue, la tâche est
        sautée et ``process_tasks_cli`` sort 0, sans MP4. ``image_outputs``
        n'est pas posé : le drapeau ne suffit pas à voir l'échec.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "fun_inp_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("fun_inp_1.3B", message)
            self.assertIn("image_prompt_type", message)
            self.assertIn("SEVL", message)
            self.assertIn("sautée", message)
            self.assertIn("code 0", message)
            self.assertIn("query_model_def", message)
            self.assertNotIn("image_outputs", message)

    def test_chrono_injected_start_image_skips_model_mode_two(self) -> None:
        """``model_mode`` 2 n'écrit pas de vidéo si l'image de départ manque.

        Chrono est le premier JSON : ``query_model_def`` injecte ``SEVL``
        sans ``black_frame``. La validation saute avant
        ``validate_generative_settings``.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "chrono_edit", "image_outputs": False},
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("chrono_edit", message)
            self.assertIn("SEVL", message)
            self.assertIn("sautée", message)
            self.assertIn("code 0", message)

    def test_empty_prompt_types_keep_chrono_model_mode_two(self) -> None:
        """Sans ``S`` dans les types autorisés, ``model_mode`` 2 écrit une vidéo."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "chrono_edit",
                            "image_outputs": False,
                            "image_prompt_types_allowed": "",
                        },
                        "model_mode": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_settings_clear_the_injected_start_image(self) -> None:
        """Un ``image_prompt_type`` vide dans les réglages gagne sur le ``S``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "fun_inp_1.3B"},
                        "image_prompt_type": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_i2v_black_frame_is_not_a_skipped_task(self) -> None:
        """``black_frame`` fabrique l'image manquante : la tâche n'est pas sautée."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "i2v"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_fun_inp_without_s_is_not_skipped(self) -> None:
        """Le premier ``t2v_1.3B`` injecte une chaîne sans ``S``.

        Le finetune passe à ``fun_inp_1.3B`` sans réécrire la clé.
        ``update_default_settings`` ne pose pas ``S``.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "fun_inp_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_fun_inp_start_image_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand Fun InP saute la tâche."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            defaults.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "fun_inp_1.3B",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("fun_inp_1.3B", message)
            self.assertIn("SEVL", message)
            self.assertIn("code 0", message)

    def test_phantom_reference_image_is_skipped(self) -> None:
        """Phantom injecte un premier choix ``I`` et pose ``video_prompt_type``.

        ``one_image_ref_needed`` est commenté dans le handler. La lettre
        ``I`` suffit : ``validate_settings`` saute la tâche, le CLI sort 0.
        ``image_outputs`` n'est pas posé.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "phantom_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("phantom_1.3B", message)
            self.assertIn("video_prompt_type", message)
            self.assertIn("image de référence", message)
            self.assertIn("code 0", message)
            self.assertNotIn("one_image_ref_needed", message)
            self.assertNotIn("image_outputs", message)

    def test_empty_video_prompt_still_adds_phantom_i(self) -> None:
        """Un ``video_prompt_type`` vide reprend ``I`` via le premier choix."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "phantom_1.3B"},
                        "video_prompt_type": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("ajoute I", str(caught.exception))

    def test_phantom_without_i_choice_keeps_a_video(self) -> None:
        """Sans ``I`` dans le premier choix ni dans le type, Phantom ne saute pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "phantom_1.3B",
                            "image_ref_choices": {
                                "choices": [("None", ""), ("Reference Image", "I")],
                            },
                        },
                        "video_prompt_type": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_standin_one_image_ref_is_skipped(self) -> None:
        """StandIn pose ``one_image_ref_needed`` même si ``I`` est retiré."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "standin"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("standin", message)
            self.assertIn("one_image_ref_needed", message)
            self.assertIn("code 0", message)

    def test_lynx_one_image_ref_is_skipped(self) -> None:
        """Lynx a un guide personnalisé et ``one_image_ref_needed``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "lynx"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("one_image_ref_needed", message)
            self.assertIn("lynx", message)
            self.assertIn("code 0", message)

    def test_later_standin_strips_i_and_keeps_a_video(self) -> None:
        """Après ``t2v_1.3B``, ``I`` est retiré et le drapeau n'est pas injecté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "standin"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_recam_control_video_is_skipped(self) -> None:
        """``V`` survit au filtre IK : Recam exige une vidéo de contrôle."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "recam_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("recam_1.3B", message)
            self.assertIn("vidéo de contrôle", message)
            self.assertIn("UV", message)
            self.assertIn("code 0", message)

    def test_cleared_recam_video_prompt_keeps_a_video(self) -> None:
        """Un ``video_prompt_type`` vide efface le ``UV`` d'usine de Recam."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "recam_1.3B"},
                        "video_prompt_type": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_fantasy_audio_is_skipped(self) -> None:
        """Fantasy après ``t2v_1.3B`` n'a pas ``S``, mais exige un audio."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "fantasy"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("fantasy", message)
            self.assertIn("audio_prompt_type", message)
            self.assertIn("code 0", message)
            self.assertNotIn("SEVL", message)

    def test_cleared_fantasy_audio_keeps_a_video(self) -> None:
        """Un ``audio_prompt_type`` vide retire l'exigence d'audio de Fantasy."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "fantasy"},
                        "audio_prompt_type": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_phantom_reference_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand Phantom saute faute d'image."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            defaults.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "phantom_1.3B",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("phantom_1.3B", message)
            self.assertIn("code 0", message)
            self.assertIn("image de référence", message)

    def test_wanmove_trajectory_is_skipped_before_the_start_image(self) -> None:
        """WanMove exige le ``.npy`` avant l'image de départ ``S``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "wanmove"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("wanmove", message)
            self.assertIn("Trajectory File (.npy)", message)
            self.assertIn("custom_guide", message)
            self.assertIn("avant l'image de départ", message)
            self.assertIn("code 0", message)
            self.assertNotIn("SEVL", message)
            self.assertNotIn("image_prompt_type", message)

    def test_later_t2v_keeps_the_wanmove_trajectory(self) -> None:
        """Un finetune ``t2v_1.3B`` ne retire pas le guide injecté au premier JSON."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "wanmove"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trajectory File (.npy)", message)
            self.assertIn("t2v_1.3B", message)
            self.assertIn("avant l'image de départ", message)
            self.assertIn("code 0", message)
            self.assertNotIn("SEVL", message)

    def test_wanmove_after_t2v_does_not_inject_the_trajectory(self) -> None:
        """``query_model_def`` ne tourne pas sur le finetune : pas de guide."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "wanmove"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_custom_guide_is_skipped(self) -> None:
        """Un JSON qui écrit ``required`` vrai est exigé même sans WanMove."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "custom_guide": {
                                "label": "Trajectory File (.npy)",
                                "required": True,
                            },
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trajectory File (.npy)", message)
            self.assertIn("JSON", message)
            self.assertIn("code 0", message)

    def test_cleared_custom_guide_after_wanmove_keeps_a_video(self) -> None:
        """``required`` faux dans un JSON suivant remplace l'injection."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "wanmove"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "custom_guide": {"label": "Trajectory File (.npy)", "required": False},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_hidden_custom_guide_keeps_a_video(self) -> None:
        """Sans la lettre demandée, le guide n'est pas visible et n'est pas exigé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "custom_guide": {
                                "label": "Trajectory File (.npy)",
                                "required": True,
                                "video_prompt_type": "V",
                            },
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_provided_custom_guide_keeps_a_video(self) -> None:
        """Un chemin dans les réglages satisfait ``custom_guide is None``."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "custom_guide": {
                                "label": "Trajectory File (.npy)",
                                "required": True,
                            },
                        },
                        "custom_guide": "traj.npy",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_wanmove_trajectory_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le guide de WanMove manque."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir()
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "wanmove", "name": "WanMove"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "name": "Wan2.1 Text2video 1.3B",
                            "architecture": "t2v_1.3B",
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 11),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Trajectory File (.npy)", message)
            self.assertIn("code 0", message)
            self.assertIn("t2v_1.3B", message)

    def test_ovi_overlap_is_skipped(self) -> None:
        """Ovi garde le recouvrement 5 alors que overlap_max vaut 1."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ovi"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("ovi", message)
            self.assertIn("sliding_window_overlap", message)
            self.assertIn("5", message)
            self.assertIn("at most 1", message)
            self.assertIn("code 0", message)

    def test_later_t2v_keeps_the_ovi_overlap_cap(self) -> None:
        """Un finetune t2v ne retire pas le plafond injecté par le premier Ovi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ovi"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("t2v_1.3B", message)
            self.assertIn("at most 1", message)
            self.assertIn("premier", message)

    def test_ovi_after_t2v_does_not_inject_the_overlap_cap(self) -> None:
        """query_model_def ne tourne pas pour le finetune Ovi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ovi"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_overlap_cap_on_t2v_is_skipped(self) -> None:
        """Un JSON t2v qui borne overlap_max à 1 saute avec le défaut 5."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sliding_window_defaults": {"overlap_max": 1, "overlap_min": 1},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("JSON", message)
            self.assertIn("at most 1", message)

    def test_ovi_overlap_of_one_keeps_a_video(self) -> None:
        """Un recouvrement de 1 est dans la borne d'Ovi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "ovi"},
                        "sliding_window_overlap": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_ovi_without_sliding_window_keeps_a_video(self) -> None:
        """sliding_window faux retire le contrôle du recouvrement."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "ovi",
                            "sliding_window": False,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_overlap_of_one_keeps_ovi(self) -> None:
        """La file à 1 remplace le défaut 5."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ovi"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_sliding_window_overlap=1,
            )

    def test_ovi_overlap_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le recouvrement d'Ovi est hors borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            defaults.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ovi", "name": "Ovi"}}) + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("ovi", message)
            self.assertIn("at most 1", message)
            self.assertIn("code 0", message)

    def test_h3_window_min_rejects_ti2v_size(self) -> None:
        """MiniMax pose window_min à 124. ti2v_2_2 laisse la taille à 121."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ti2v_2_2"}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("ti2v_2_2", message)
            self.assertIn("sliding_window_size", message)
            self.assertIn("121", message)
            self.assertIn("at least 124", message)
            self.assertIn("code 0", message)

    def test_h3_finetune_does_not_inject_window_min(self) -> None:
        """query_model_def ne tourne pas pour le finetune MiniMax."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_h3_then_t2v_size_stays_inside_the_window(self) -> None:
        """129 est entre 124 et 481. Le finetune t2v_1.3B ne change pas la taille."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_h3_factory_size_stays_inside_its_window(self) -> None:
        """MiniMax pose la taille à 362, sous son window_max 481."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_window_min_is_skipped(self) -> None:
        """Un JSON qui pose window_min à 200 saute avec la taille 129."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sliding_window_defaults": {"window_min": 200},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("JSON", message)
            self.assertIn("129", message)
            self.assertIn("at least 200", message)

    def test_explicit_window_max_is_skipped(self) -> None:
        """Un JSON qui pose window_max à 80 saute avec la taille 129."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sliding_window_defaults": {"window_max": 80},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 80", message)
            self.assertIn("129", message)

    def test_sliding_window_off_ignores_window_min(self) -> None:
        """sliding_window faux retire le contrôle de la taille."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": False,
                            "sliding_window_defaults": {"window_min": 200},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_settings_size_inside_window_min_keeps_a_video(self) -> None:
        """Une taille écrite dans le JSON, dans la borne, reste acceptée."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sliding_window_defaults": {"window_min": 200},
                        },
                        "sliding_window_size": 200,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_size_inside_h3_window_keeps_ti2v(self) -> None:
        """La file à 200 remplace les 121 de ti2v_2_2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ti2v_2_2"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_sliding_window_size=200,
            )

    def test_frames_multiplier_widens_window_max(self) -> None:
        """get_max_frames élargit window_max. 129 passe sous un plafond de 80."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (root / "wgp_config.json").write_text(
                json.dumps({"max_frames_multiplier": 2, "video_container": "mp4"}) + "\n",
                encoding="utf-8",
            )
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sliding_window_defaults": {"window_max": 80},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_h3_size_without_injected_bounds_is_skipped(self) -> None:
        """sliding_window vrai sans bornes injectées : 362 dépasse 257."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "minimax_h3_ref2va",
                            "sliding_window": True,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("362", message)
            self.assertIn("at most 257", message)

    def test_non_integer_window_size_is_skipped(self) -> None:
        """Une taille qui n'est pas un entier saute avant le GPU."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                        },
                        "sliding_window_size": "large",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("must be an integer", message)
            self.assertIn("code 0", message)

    def test_h3_window_min_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand la taille MiniMax est hors borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir()
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va", "name": "H3"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "ti2v_2_2", "name": "TI2V"}}) + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("121", message)
            self.assertIn("at least 124", message)
            self.assertIn("code 0", message)

    def test_sub_parallel_overlap_max_8_is_skipped(self) -> None:
        """overlap_max 8 laisse le recouvrement 5 et refuse le défaut 17."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("sub_parallel_window_overlap", message)
            self.assertIn("17", message)
            self.assertIn("at most 8", message)
            self.assertIn("code 0", message)

    def test_overlap_max_8_without_sub_parallel_keeps_a_video(self) -> None:
        """Sans le drapeau, le recouvrement 5 reste sous overlap_max 8."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sub_parallel_default_cap_keeps_a_video(self) -> None:
        """Sans overlap_max, le plafond reste 97 et le défaut 17 passe."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_scail2_then_t2v_keeps_factory_sub_parallel(self) -> None:
        """scail2 injecte le drapeau. overlap_max reste 97 : 17 passe."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "scail2_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_scail2_finetune_does_not_inject_sub_parallel(self) -> None:
        """query_model_def ne tourne pas pour le finetune scail2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "scail2_1.3B",
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "video_prompt_type": "",
                        "image_prompt_type": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_scail2_then_overlap_max_8_is_skipped(self) -> None:
        """Le drapeau injecté survit. overlap_max 8 refuse le défaut 17."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "scail2_14B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("sub_parallel_window_overlap", message)
            self.assertIn("17", message)
            self.assertIn("at most 8", message)
            self.assertIn("scail2_14B", message)
            self.assertIn("premier", message)
            self.assertIn("code 0", message)

    def test_explicit_sub_parallel_false_clears_scail2(self) -> None:
        """Un JSON qui écrit le drapeau à faux retire l'injection de scail2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "scail2_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": False,
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sliding_window_off_still_rejects_sub_parallel(self) -> None:
        """Le drapeau parallèle ne dépend pas de sliding_window."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": False,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("sub_parallel_window_overlap", message)
            self.assertIn("at most 8", message)

    def test_settings_sub_parallel_overlap_inside_cap_keeps_a_video(self) -> None:
        """Un recouvrement parallèle écrit dans la borne reste accepté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_sub_parallel_overlap_inside_cap_keeps_a_video(self) -> None:
        """La file à 8 remplace le défaut 17."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_sub_parallel_window_overlap=8,
            )

    def test_sub_parallel_below_overlap_min_is_skipped(self) -> None:
        """Le recouvrement de fenêtre tient. Le parallèle 17 est sous overlap_min."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_min": 20, "overlap_max": 40},
                        },
                        "sliding_window_overlap": 20,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("sub_parallel_window_overlap", message)
            self.assertIn("17", message)
            self.assertIn("at least 20", message)

    def test_empty_sub_parallel_overlap_keeps_a_video(self) -> None:
        """Une chaîne vide n'est pas une erreur pour validate_setting_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_overlap": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_non_integer_sub_parallel_overlap_is_skipped(self) -> None:
        """Un recouvrement parallèle qui n'est pas un entier saute avant le GPU."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_overlap": "large",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("must be an integer", message)
            self.assertIn("code 0", message)

    def test_bool_sub_parallel_overlap_is_skipped(self) -> None:
        """True n'est pas un entier pour extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                        },
                        "sub_parallel_window_overlap": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("must be a integer", message)

    def test_sub_parallel_overlap_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le recouvrement parallèle sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            defaults.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "name": "T2V",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("17", message)
            self.assertIn("at most 8", message)
            self.assertIn("code 0", message)

    def test_sub_parallel_size_300_is_skipped(self) -> None:
        """La taille 300 dépasse window_max 257. Le recouvrement 8 tient."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 300,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("sub_parallel_window_size", message)
            self.assertIn("300", message)
            self.assertIn("at most 257", message)
            self.assertIn("Sub Parallel Window Size", message)
            self.assertNotIn("Sub Parallel Window Overlap", message)
            self.assertIn("code 0", message)

    def test_size_300_is_reported_before_a_bad_overlap(self) -> None:
        """Les deux sortent de la borne. WanGP refuse la taille d'abord."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 300,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Sub Parallel Window Size must be at most 257", message)
            self.assertNotIn("Sub Parallel Window Overlap", message)

    def test_sub_parallel_size_at_window_max_keeps_a_video(self) -> None:
        """257 est le plafond d'usine. Le recouvrement 8 reste accepté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 257,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sub_parallel_size_just_over_max_is_skipped(self) -> None:
        """258 dépasse le plafond 257 même si le recouvrement tient."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sliding_window_size": 129,
                        "sub_parallel_window_size": 258,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 257", str(caught.exception))

    def test_factory_sub_parallel_size_ignores_a_tight_window_max(self) -> None:
        """La taille d'usine 0 passe sous un window_max de 80."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": False,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"window_max": 80},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_json_window_max_rejects_parallel_size(self) -> None:
        """window_max 200 laisse sliding_window_size 129 et refuse 300."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"window_max": 200},
                        },
                        "sub_parallel_window_size": 300,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 200", message)
            self.assertIn("sub_parallel_window_size", message)

    def test_window_min_does_not_apply_to_parallel_size(self) -> None:
        """window_min 124 borne la fenêtre. La taille parallèle 1 passe."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"window_min": 124, "window_max": 481},
                        },
                        "sliding_window_size": 200,
                        "sub_parallel_window_size": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_negative_sub_parallel_size_is_skipped(self) -> None:
        """Le minimum fixe est 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": -1,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at least 0", str(caught.exception))

    def test_queue_sub_parallel_size_300_is_skipped(self) -> None:
        """La file à 300 remplace la taille d'usine 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_sub_parallel_window_size=300,
                )
            message = str(caught.exception)
            self.assertIn("300", message)
            self.assertIn("at most 257", message)
            self.assertIn("file", message.lower())

    def test_queue_sub_parallel_size_overrides_a_bad_json(self) -> None:
        """La file à 0 remplace un JSON à 300."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 300,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_sub_parallel_window_size=0,
            )

    def test_scail2_then_parallel_size_300_is_skipped(self) -> None:
        """Le drapeau injecté survit. window_max reste 257 et refuse 300."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "scail2_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "t2v_1.3B"},
                        "sub_parallel_window_size": 300,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 257", message)
            self.assertIn("scail2_1.3B", message)
            self.assertIn("premier", message)

    def test_scail2_finetune_size_300_does_not_inject(self) -> None:
        """query_model_def ne tourne pas : la taille 300 n'est pas lue."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "t2v_1.3B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "scail2_1.3B"},
                        "video_prompt_type": "",
                        "image_prompt_type": "",
                        "sub_parallel_window_size": 300,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_false_ignores_parallel_size_300(self) -> None:
        """Un drapeau faux retire l'injection, même avec une taille 300."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "scail2_14B"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": False,
                        },
                        "sub_parallel_window_size": 300,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sliding_off_still_rejects_parallel_size(self) -> None:
        """Le drapeau parallèle ne dépend pas de sliding_window."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": False,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 300,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Sub Parallel Window Size must be at most 257", message)
            self.assertNotIn("Sub Parallel Window Overlap", message)

    def test_h3_window_max_allows_parallel_size_300(self) -> None:
        """Les bornes MiniMax laissent 300 sous 481. La taille de fenêtre 129 tient."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                        },
                        "sub_parallel_window_size": 300,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_h3_window_max_rejects_parallel_size_500(self) -> None:
        """500 dépasse le window_max 481 injecté par MiniMax."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            finetunes = root / "finetunes"
            defaults.mkdir(parents=True)
            finetunes.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps({"model": {"architecture": "minimax_h3_ref2va"}}) + "\n",
                encoding="utf-8",
            )
            (finetunes / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                        },
                        "sub_parallel_window_size": 500,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("500", message)
            self.assertIn("at most 481", message)
            self.assertIn("sub_parallel_window_size", message)

    def test_frames_multiplier_widens_parallel_size_cap(self) -> None:
        """get_max_frames porte le plafond à 513. 300 passe, le recouvrement 8 aussi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (root / "wgp_config.json").write_text(
                json.dumps({"max_frames_multiplier": 2, "video_container": "mp4"}) + "\n",
                encoding="utf-8",
            )
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 300,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_frames_multiplier_still_rejects_parallel_size_600(self) -> None:
        """600 reste au-dessus du plafond élargi 513."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (root / "wgp_config.json").write_text(
                json.dumps({"max_frames_multiplier": 2, "video_container": "mp4"}) + "\n",
                encoding="utf-8",
            )
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 600,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 513", str(caught.exception))

    def test_empty_sub_parallel_size_keeps_a_video(self) -> None:
        """Une chaîne vide n'est pas une erreur pour validate_setting_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": "",
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_string_sub_parallel_size_300_is_skipped(self) -> None:
        """« 300 » est un entier au-dessus de 257."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": "300",
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 257", message)
            self.assertNotIn("must be an integer", message)

    def test_non_integer_sub_parallel_size_is_skipped(self) -> None:
        """Une taille qui n'est pas un entier saute avant le recouvrement."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": "large",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("must be an integer", message)
            self.assertIn("Sub Parallel Window Size", message)
            self.assertNotIn("Sub Parallel Window Overlap", message)
            self.assertIn("code 0", message)

    def test_bool_sub_parallel_size_is_skipped(self) -> None:
        """True n'est pas un entier pour extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sub_parallel_windows": True,
                        },
                        "sub_parallel_window_size": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a integer", str(caught.exception))

    def test_sub_parallel_size_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand la taille parallèle sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            defaults = root / "defaults"
            defaults.mkdir()
            (defaults / "t2v_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "name": "T2V",
                            "sliding_window": True,
                            "sub_parallel_windows": True,
                            "sliding_window_defaults": {"overlap_max": 8},
                        },
                        "sub_parallel_window_size": 300,
                        "sub_parallel_window_overlap": 8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("300", message)
            self.assertIn("at most 257", message)
            self.assertIn("code 0", message)

    def test_color_strength_2_is_skipped(self) -> None:
        """La force 2 dépasse 1. La taille 0 et le recouvrement 8 tiennent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 2,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Color Correction Strength must be at most 1", message)
            self.assertIn("code 0", message)
            self.assertNotIn("Sub Parallel Window", message)

    def test_color_strength_1_01_is_skipped(self) -> None:
        """1.01 est un nombre au-dessus de 1, pas une erreur de type."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": 1.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("1.01", message)
            self.assertIn("at most 1", message)
            self.assertNotIn("must be a number", message)

    def test_color_strength_bounds_keep_a_video(self) -> None:
        """0 et 1 sont dans la borne. Le défaut d'usine 0 aussi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 1,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": 0},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "t2v_1.3B"})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_color_correction_false_ignores_strength_2(self) -> None:
        """Un drapeau faux laisse passer une force 2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": False},
                {"sliding_window_color_correction_strength": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_injected_color_correction_rejects_strength_2(self) -> None:
        """query_model_def de Wan pose le drapeau. La clé peut manquer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"sliding_window_color_correction_strength": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 1", message)
            self.assertIn("query_model_def", message)
            self.assertIn("code 0", message)

    def test_animate2_clears_injected_color_correction(self) -> None:
        """animate2 remet le drapeau à faux dans le même query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "animate2"},
                {
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                    "sliding_window_color_correction_strength": 2,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_animate2_keeps_injected_color_correction(self) -> None:
        """Un finetune animate2 ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "animate2"},
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                    "sliding_window_color_correction_strength": 2,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Color Correction Strength must be at most 1", str(caught.exception))

    def test_later_false_clears_color_correction(self) -> None:
        """Un JSON suivant qui écrit le drapeau à faux l'efface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "color_correction": False},
                    "sliding_window_color_correction_strength": 2,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_parallel_overlap_is_reported_before_color(self) -> None:
        """Le recouvrement 17 dépasse overlap_max 8 avant la force 2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 17,
                    "sliding_window_color_correction_strength": 2,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Sub Parallel Window Overlap", message)
            self.assertNotIn("Color Correction Strength", message)

    def test_non_wan_without_flag_ignores_strength_2(self) -> None:
        """ovi n'injecte pas color_correction. La force 2 est ignorée."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "ovi"},
                {
                    "sliding_window_overlap": 1,
                    "sliding_window_color_correction_strength": 2,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_color_correction_on_non_wan_is_skipped(self) -> None:
        """La clé écrite à vrai suffit, même hors du handler Wan."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "ovi", "color_correction": True},
                {
                    "sliding_window_overlap": 1,
                    "sliding_window_color_correction_strength": 2,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 1", str(caught.exception))

    def test_bool_color_strength_is_skipped(self) -> None:
        """True n'est pas un nombre pour extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": True},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a number", str(caught.exception))

    def test_empty_color_strength_keeps_a_video(self) -> None:
        """Une chaîne vide n'est pas une erreur pour validate_setting_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_non_number_color_strength_is_skipped(self) -> None:
        """Une force illisible saute avant toute vidéo."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": "large"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("must be a number", message)
            self.assertIn("code 0", message)

    def test_string_color_strength_2_is_skipped(self) -> None:
        """« 2 » est le nombre 2, au-dessus de 1."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": "2"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 1", message)
            self.assertNotIn("must be a number", message)

    def test_negative_color_strength_is_skipped(self) -> None:
        """Une force négative est sous le minimum 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": -0.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at least 0", str(caught.exception))

    def test_queue_color_strength_2_is_skipped(self) -> None:
        """La file à 2 remplace la force d'usine 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v_1.3B"})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_color_correction_strength=2,
                )
            message = str(caught.exception)
            self.assertIn("at most 1", message)
            self.assertIn("file", message.lower())

    def test_queue_color_strength_overrides_a_bad_json(self) -> None:
        """La file à 0 remplace un JSON à 2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "color_correction": True},
                {"sliding_window_color_correction_strength": 2},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_color_correction_strength=0,
            )

    def test_color_strength_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand la force sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 2,
                },
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("2", message)
            self.assertIn("at most 1", message)
            self.assertIn("code 0", message)

    def test_overlap_noise_151_is_skipped(self) -> None:
        """Le bruit 151 dépasse 150. La force 0 et le recouvrement 8 tiennent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "vace_class": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 151,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn(
                "Noise to be added to overlapped frames to reduce blur effect must be at most 150",
                message,
            )
            self.assertIn("code 0", message)
            self.assertNotIn("Color Correction Strength", message)
            self.assertNotIn("Sub Parallel Window", message)

    def test_overlap_noise_bounds_keep_a_video(self) -> None:
        """0 et 150 sont dans la borne. Le défaut d'usine 0 aussi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": 150},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": 0},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "vace_1.3B"})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_overlap_noise_float_is_truncated(self) -> None:
        """int tronque : 150.5 passe, 151.9 devient 151."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": 150.5},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": 151.9},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertNotIn("must be an integer", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": -0.1},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_string_overlap_noise_151_is_skipped(self) -> None:
        """« 151 » et « 151.0 » sont l'entier 151. « 151.9 » ne l'est pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": "151"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 150", str(caught.exception))
            self.assertNotIn("must be an integer", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": "151.0"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 150", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": "151.9"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be an integer", str(caught.exception))

    def test_negative_overlap_noise_is_skipped(self) -> None:
        """Un bruit négatif est sous le minimum 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": -1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at least 0", str(caught.exception))

    def test_bool_overlap_noise_is_skipped(self) -> None:
        """True n'est pas un entier pour extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": True},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a integer", str(caught.exception))

    def test_empty_overlap_noise_keeps_a_video(self) -> None:
        """Une chaîne vide n'est pas une erreur pour validate_setting_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_non_integer_overlap_noise_is_skipped(self) -> None:
        """Un bruit illisible saute avant toute vidéo."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": "large"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("must be an integer", message)
            self.assertIn("code 0", message)

    def test_injected_vace_rejects_noise_151(self) -> None:
        """query_model_def de VACE pose vace_class. La clé peut manquer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "vace_1.3B"},
                {"sliding_window_overlap_noise": 151},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertIn("query_model_def", message)
            self.assertIn("vace_1.3B", message)
            self.assertIn("code 0", message)

    def test_vace_finetune_after_t2v_does_not_inject(self) -> None:
        """Un finetune VACE ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "vace_1.3B"},
                    "sliding_window_overlap_noise": 151,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_false_clears_vace_class(self) -> None:
        """Un JSON suivant qui écrit vace_class à faux l'efface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "vace_1.3B", "vace_class": True},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "vace_class": False},
                    "sliding_window_overlap_noise": 151,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_t2v_without_vace_ignores_noise_151(self) -> None:
        """Sans vace_class, model_type t2v_1.3B ignore le bruit 151."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"sliding_window_overlap_noise": 151},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_architecture_sky_df_is_not_the_model_type(self) -> None:
        """L'architecture du JSON ne rend pas le bruit visible."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "sky_df_1.3B"},
                {"sliding_window_overlap_noise": 151},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sky_df_model_type_rejects_noise_151(self) -> None:
        """sky_df_1.3B rend le bruit visible. df_handler ignore une force 2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "sky_df_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "sky_df_1.3B"},
                        "sliding_window_color_correction_strength": 2,
                        "sliding_window_overlap_noise": 151,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "sky_df_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertIn("sky_df_1.3B", message)
            self.assertNotIn("Color Correction Strength", message)
            self.assertIn("code 0", message)

    def test_diffusion_forcing_model_type_rejects_noise_151(self) -> None:
        """Le sous-chaîne diffusion_forcing suffit, même sans vace_class."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "clip_diffusion_forcing.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "t2v_1.3B", "color_correction": False},
                        "sliding_window_overlap_noise": 151,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "clip_diffusion_forcing")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertIn("diffusion_forcing", message)

    def test_color_strength_is_reported_before_noise(self) -> None:
        """La force 2 sort avant le bruit 151."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "color_correction": True,
                    "vace_class": True,
                },
                {
                    "sliding_window_color_correction_strength": 2,
                    "sliding_window_overlap_noise": 151,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Color Correction Strength", message)
            self.assertNotIn("overlap_noise", message)
            self.assertNotIn("blur effect", message)

    def test_color_false_lets_noise_through(self) -> None:
        """Un drapeau de couleur faux laisse passer la force 2. Le bruit 151 saute."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "color_correction": False,
                    "vace_class": True,
                },
                {
                    "sliding_window_color_correction_strength": 2,
                    "sliding_window_overlap_noise": 151,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertNotIn("Color Correction Strength", message)

    def test_parallel_overlap_is_reported_before_noise(self) -> None:
        """Le recouvrement 17 dépasse overlap_max 8 avant le bruit 151."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "vace_class": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 17,
                    "sliding_window_overlap_noise": 151,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Sub Parallel Window Overlap", message)
            self.assertNotIn("blur effect", message)

    def test_queue_overlap_noise_151_is_skipped(self) -> None:
        """La file à 151 remplace le bruit d'usine 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "vace_1.3B"})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_overlap_noise=151,
                )
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertIn("file", message.lower())

    def test_queue_overlap_noise_overrides_a_bad_json(self) -> None:
        """La file à 0 remplace un JSON à 151."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True},
                {"sliding_window_overlap_noise": 151},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_overlap_noise=0,
            )

    def test_overlap_noise_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le bruit sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "vace_class": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 151,
                },
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("151", message)
            self.assertIn("at most 150", message)
            self.assertIn("code 0", message)

    def test_discard_last_frames_21_is_skipped(self) -> None:
        """Le rejet 21 dépasse 20. Le bruit 0 et la force 0 tiennent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 0,
                    "sliding_window_discard_last_frames": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Discard Last Frames of a Window must be at most 20", message)
            self.assertIn("code 0", message)
            self.assertNotIn("blur effect", message)
            self.assertNotIn("Color Correction Strength", message)
            self.assertNotIn("Sub Parallel Window", message)

    def test_discard_last_frames_bounds_keep_a_video(self) -> None:
        """0 et 20 sont dans la borne. Le défaut d'usine 0 aussi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": 20},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": 0},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "t2v_1.3B", "sliding_window": True})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_discard_last_frames_float_is_truncated(self) -> None:
        """int tronque : 20.9 et -0.1 passent, 21.5 devient 21."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": 20.9},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": -0.1},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": 21.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 20", message)
            self.assertNotIn("must be an integer", message)

    def test_string_discard_21_is_skipped(self) -> None:
        """« 21 » est l'entier 21. int() l'accepte, donc extra_settings refuse."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": "21"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 20", str(caught.exception))
            self.assertNotIn("must be an integer", str(caught.exception))

    def test_discard_scheduler_string_is_not_a_silent_skip(self) -> None:
        """« large », « 21.5 » et « 21.0 » lèvent ValueError avant extra_settings.

        Le frame scheduler fait int() quand image_mode est 0 et
        sliding_window est vrai. Le processus ne sort pas 0.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in ("large", "21.5", "21.0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "sliding_window": True},
                    {"sliding_window_discard_last_frames": raw},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_negative_discard_last_frames_is_skipped(self) -> None:
        """Un rejet négatif est sous le minimum 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": -1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at least 0", str(caught.exception))

    def test_bool_discard_last_frames_is_skipped(self) -> None:
        """True n'est pas un entier pour extra_settings. int(True) ne plante pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": True},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a integer", str(caught.exception))

    def test_empty_discard_last_frames_keeps_a_video(self) -> None:
        """Une chaîne vide n'est pas une erreur pour validate_setting_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sliding_window_off_ignores_discard_21(self) -> None:
        """sliding_window faux retire le contrôle, y compris pour t2v_1.3B."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False},
                {"sliding_window_discard_last_frames": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"sliding_window_discard_last_frames": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_injected_sliding_window_rejects_discard_21(self) -> None:
        """Le premier JSON qui injecte sliding_window refuse 21 sans la clé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for architecture in ("t2v", "vace_1.3B", "ti2v_2_2"):
                _write_model(
                    root,
                    {"architecture": architecture},
                    {"sliding_window_discard_last_frames": 21},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("at most 20", message)
                self.assertIn("query_model_def", message)
                self.assertIn(architecture, message)
                self.assertIn("code 0", message)

    def test_animate_injection_rejects_discard_21(self) -> None:
        """animate injecte sliding_window. Les lettres d'usine sont vidées."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "animate"},
                {
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                    "sliding_window_discard_last_frames": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 20", message)
            self.assertIn("animate", message)

    def test_finetune_keeps_injected_sliding_window(self) -> None:
        """Un finetune t2v_1.3B ne retire pas le sliding_window injecté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "sliding_window_discard_last_frames": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 20", str(caught.exception))

    def test_finetune_t2v_after_t2v_1_3b_does_not_inject(self) -> None:
        """Un t2v seulement en finetune ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v"},
                    "sliding_window_discard_last_frames": 21,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_false_clears_sliding_window(self) -> None:
        """Un JSON suivant qui écrit sliding_window à faux l'efface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v", "sliding_window": True},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "sliding_window": False},
                    "sliding_window_discard_last_frames": 21,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_multitalk_factory_discard_keeps_a_video(self) -> None:
        """multitalk pose 4, dans la borne. Les médias d'usine sont vidés."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "multitalk"},
                {"image_prompt_type": "", "audio_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "multitalk"},
                {
                    "image_prompt_type": "",
                    "audio_prompt_type": "",
                    "sliding_window_discard_last_frames": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 20", str(caught.exception))

    def test_sky_df_hides_discard_21(self) -> None:
        """sky_df dans model_type cache le rejet. La casse ne compte pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "SKY_DF_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "t2v_1.3B", "sliding_window": True},
                        "sliding_window_discard_last_frames": 21,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "SKY_DF_1.3B")

    def test_diffusion_forcing_hides_discard_21(self) -> None:
        """diffusion_forcing cache le rejet même si sliding_window est vrai."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "clip_diffusion_forcing.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "color_correction": False,
                        },
                        "sliding_window_discard_last_frames": 21,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "clip_diffusion_forcing")

    def test_noise_151_is_reported_before_discard(self) -> None:
        """Le bruit 151 sort avant le rejet 21."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "vace_class": True,
                    "color_correction": True,
                },
                {
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 151,
                    "sliding_window_discard_last_frames": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertNotIn("Discard Last Frames", message)

    def test_color_strength_is_reported_before_discard(self) -> None:
        """La force 2 sort avant le rejet 21."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "color_correction": True,
                },
                {
                    "sliding_window_color_correction_strength": 2,
                    "sliding_window_discard_last_frames": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Color Correction Strength", message)
            self.assertNotIn("Discard Last Frames", message)

    def test_queue_discard_21_is_skipped(self) -> None:
        """La file à 21 remplace le rejet d'usine 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v", "sliding_window": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_discard_last_frames=21,
                )
            message = str(caught.exception)
            self.assertIn("at most 20", message)
            self.assertIn("file", message.lower())

    def test_queue_discard_overrides_a_bad_json(self) -> None:
        """La file à 0 remplace un JSON à 21."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_discard_last_frames": 21},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_discard_last_frames=0,
            )

    def test_discard_last_frames_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le rejet sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 0,
                    "sliding_window_discard_last_frames": 21,
                },
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("21", message)
            self.assertIn("at most 20", message)
            self.assertIn("code 0", message)

    def test_trim_first_frames_11_is_skipped(self) -> None:
        """Le trim 11 dépasse 10. Le rejet 0, le bruit 0 et la force 0 tiennent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 0,
                    "sliding_window_discard_last_frames": 0,
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn(
                "Trim First Frames in First Window or if there is no Overlap Frames must be at most 10",
                message,
            )
            self.assertIn("code 0", message)
            self.assertNotIn("Discard Last Frames", message)
            self.assertNotIn("blur effect", message)
            self.assertNotIn("Color Correction Strength", message)

    def test_trim_first_frames_bounds_keep_a_video(self) -> None:
        """0 et 10 sont dans la borne. Le défaut d'usine 0 aussi."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": 10},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": 0},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "t2v_1.3B", "sliding_window": True})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_trim_first_frames_float_is_truncated(self) -> None:
        """int tronque : 10.9 et -0.1 passent, 11.5 devient 11."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": 10.9},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": -0.1},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": 11.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 10", message)
            self.assertNotIn("must be an integer", message)

    def test_string_trim_11_is_skipped(self) -> None:
        """« 11 » et « 11.0 » sont l'entier 11. extra_settings refuse."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in ("11", "11.0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "sliding_window": True},
                    {"sliding_window_trim_first_frames": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("at most 10", message)
                self.assertNotIn("must be an integer", message)

    def test_string_trim_is_an_integer_error(self) -> None:
        """« large » et « 11.5 » ne passent pas par le int() du scheduler.

        extra_settings renvoie must be an integer et le CLI sort 0.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in ("large", "11.5"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "sliding_window": True},
                    {"sliding_window_trim_first_frames": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("must be an integer", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 10", message)

    def test_negative_trim_first_frames_is_skipped(self) -> None:
        """Un trim négatif est sous le minimum 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": -1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at least 0", str(caught.exception))

    def test_bool_trim_first_frames_is_skipped(self) -> None:
        """True n'est pas un entier pour extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": True},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a integer", str(caught.exception))

    def test_empty_trim_first_frames_keeps_a_video(self) -> None:
        """Une chaîne vide n'est pas une erreur pour validate_setting_value."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_sliding_window_off_ignores_trim_11(self) -> None:
        """sliding_window faux retire le contrôle, y compris pour t2v_1.3B."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False},
                {"sliding_window_trim_first_frames": 11},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"sliding_window_trim_first_frames": 11},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_injected_sliding_window_rejects_trim_11(self) -> None:
        """Le premier JSON qui injecte sliding_window refuse 11 sans la clé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for architecture in ("t2v", "vace_1.3B", "ti2v_2_2"):
                _write_model(
                    root,
                    {"architecture": architecture},
                    {"sliding_window_trim_first_frames": 11},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("at most 10", message)
                self.assertIn("query_model_def", message)
                self.assertIn(architecture, message)
                self.assertIn("code 0", message)

    def test_animate_injection_rejects_trim_11(self) -> None:
        """animate injecte sliding_window. Les lettres d'usine sont vidées."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "animate"},
                {
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 10", message)
            self.assertIn("animate", message)

    def test_finetune_keeps_injected_window_for_trim(self) -> None:
        """Un finetune t2v_1.3B ne retire pas le sliding_window injecté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 10", str(caught.exception))

    def test_finetune_t2v_after_t2v_1_3b_does_not_inject_trim(self) -> None:
        """Un t2v seulement en finetune ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v"},
                    "sliding_window_trim_first_frames": 11,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_false_clears_window_for_trim(self) -> None:
        """Un JSON suivant qui écrit sliding_window à faux l'efface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v", "sliding_window": True},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "sliding_window": False},
                    "sliding_window_trim_first_frames": 11,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_multitalk_factory_trim_keeps_a_video(self) -> None:
        """multitalk ne réécrit pas le trim. Le défaut 0 reste dans la borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "multitalk"},
                {"image_prompt_type": "", "audio_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "multitalk"},
                {
                    "image_prompt_type": "",
                    "audio_prompt_type": "",
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 10", str(caught.exception))

    def test_sky_df_still_rejects_trim_11(self) -> None:
        """sky_df cache le rejet, pas le trim. La casse ne compte pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "SKY_DF_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "t2v_1.3B", "sliding_window": True},
                        "sliding_window_discard_last_frames": 21,
                        "sliding_window_trim_first_frames": 11,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "SKY_DF_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 10", message)
            self.assertNotIn("Discard Last Frames", message)

    def test_diffusion_forcing_still_rejects_trim_11(self) -> None:
        """diffusion_forcing cache le rejet. Le trim 11 reste l'erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            defaults = root / "defaults"
            defaults.mkdir(parents=True)
            (defaults / "clip_diffusion_forcing.json").write_text(
                json.dumps(
                    {
                        "model": {
                            "architecture": "t2v_1.3B",
                            "sliding_window": True,
                            "color_correction": False,
                        },
                        "sliding_window_discard_last_frames": 21,
                        "sliding_window_trim_first_frames": 11,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "clip_diffusion_forcing")
            message = str(caught.exception)
            self.assertIn("at most 10", message)
            self.assertNotIn("Discard Last Frames", message)

    def test_discard_21_is_reported_before_trim(self) -> None:
        """Le rejet 21 sort avant le trim 11 quand il est visible."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {
                    "sliding_window_discard_last_frames": 21,
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Discard Last Frames", message)
            self.assertNotIn("Trim First Frames", message)

    def test_noise_151_is_reported_before_trim(self) -> None:
        """Le bruit 151 sort avant le trim 11."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "vace_class": True,
                    "color_correction": True,
                },
                {
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 151,
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertNotIn("Trim First Frames", message)

    def test_color_strength_is_reported_before_trim(self) -> None:
        """La force 2 sort avant le trim 11."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "color_correction": True,
                },
                {
                    "sliding_window_color_correction_strength": 2,
                    "sliding_window_trim_first_frames": 11,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Color Correction Strength", message)
            self.assertNotIn("Trim First Frames", message)

    def test_queue_trim_11_is_skipped(self) -> None:
        """La file à 11 remplace le trim d'usine 0."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v", "sliding_window": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_trim_first_frames=11,
                )
            message = str(caught.exception)
            self.assertIn("at most 10", message)
            self.assertIn("file", message.lower())

    def test_queue_trim_overrides_a_bad_json(self) -> None:
        """La file à 0 remplace un JSON à 11."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {"sliding_window_trim_first_frames": 11},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_trim_first_frames=0,
            )

    def test_discard_string_blocks_trim_before_extra_settings(self) -> None:
        """Un rejet « large » lève ValueError avant le trim 11.

        Le processus ne sort pas 0. sky_df cache le rejet dans extra_settings
        mais le frame scheduler l'évalue quand même.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {
                    "sliding_window_discard_last_frames": "large",
                    "sliding_window_trim_first_frames": 11,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            defaults = root / "defaults"
            (defaults / "SKY_DF_1.3B.json").write_text(
                json.dumps(
                    {
                        "model": {"architecture": "t2v_1.3B", "sliding_window": True},
                        "sliding_window_discard_last_frames": "large",
                        "sliding_window_trim_first_frames": 11,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wan2gp_runner._require_video_model_definition(root, "SKY_DF_1.3B")

    def test_discard_step_zero_blocks_trim(self) -> None:
        """discard_step 0 lève ZeroDivisionError avant extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sliding_window_defaults": {"discard_step": 0},
                },
                {"sliding_window_trim_first_frames": 11},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "sliding_window_defaults": {"discard_step": 4},
                },
                {"sliding_window_trim_first_frames": 11},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("at most 10", str(caught.exception))

    def test_trim_first_frames_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le trim sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "sliding_window": True,
                    "sub_parallel_windows": True,
                    "color_correction": True,
                    "sliding_window_defaults": {"overlap_max": 8},
                },
                {
                    "sub_parallel_window_size": 0,
                    "sub_parallel_window_overlap": 8,
                    "sliding_window_color_correction_strength": 0,
                    "sliding_window_overlap_noise": 0,
                    "sliding_window_discard_last_frames": 0,
                    "sliding_window_trim_first_frames": 11,
                },
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("11", message)
            self.assertIn("at most 10", message)
            self.assertIn("code 0", message)

    def test_top_p_above_one_is_skipped(self) -> None:
        """1.01 et 2 dépassent 1. Le trim 0 et sliding_window faux ne masquent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (1.01, 2, 2.0):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_p_slider": True, "sliding_window": False},
                    {"top_p": raw, "sliding_window_trim_first_frames": 0},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Top-p must be at most 1", message)
                self.assertIn("code 0", message)
                self.assertNotIn("Trim First Frames", message)
                self.assertNotIn("Top-k", message)

    def test_top_p_bounds_keep_a_video(self) -> None:
        """0, 1 et le défaut d'usine 0.9 sont dans la borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0, 1, 0.9, 1.0, "1.0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_p_slider": True},
                    {"top_p": raw},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "t2v_1.3B", "top_p_slider": True})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_top_p_string_and_type_errors(self) -> None:
        """« 1.01 » est le nombre 1.01. « large » et True ne sont pas des nombres."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": True},
                {"top_p": "1.01"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-p must be at most 1", str(caught.exception))
            for raw in ("large", True):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_p_slider": True},
                    {"top_p": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Top-p must be a number", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 1", message)

    def test_negative_and_empty_top_p(self) -> None:
        """-0.01 est sous 0. Une chaîne vide n'est pas une erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": True},
                {"top_p": -0.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-p must be at least 0", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": True},
                {"top_p": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_slider_off_ignores_top_p(self) -> None:
        """Sans top_p_slider, un top_p 2 est ignoré. top_k_slider seul aussi.

        t2v_1.3B n'injecte pas top_p_slider. Un top_k dans la borne n'est
        pas ce saut. NAG vrai et NAG_scale 21 sont le saut suivant : le
        message dit NAG Scale, pas Top-p.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"top_p": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": False},
                {"top_p": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": 0},
                {"top_p": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": True},
                {"top_p": 2, "top_k": 50},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_scale": 21, "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 20", message)
            self.assertNotIn("Top-p", message)

    def test_truthy_slider_rejects_top_p(self) -> None:
        """1 et la chaîne false sont vrais pour bool(), comme WanGP."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for flag in (1, "false"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_p_slider": flag},
                    {"top_p": 2},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("Top-p must be at most 1", str(caught.exception))

    def test_finetune_slider_rejects_top_p(self) -> None:
        """Un finetune qui écrit le drapeau et top_p 2 fait sauter la tâche."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "top_p_slider": True},
                    "top_p": 1.01,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-p must be at most 1", str(caught.exception))

    def test_later_false_clears_top_p_slider(self) -> None:
        """Un JSON suivant qui écrit top_p_slider à faux retire le contrôle."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": True},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "top_p_slider": False},
                    "top_p": 2,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_top_p_overrides_the_factory(self) -> None:
        """La file à 2 remplace l'usine 0.9. La file à 0.9 répare un JSON à 2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v_1.3B", "top_p_slider": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_top_p=2,
                )
            message = str(caught.exception)
            self.assertIn("Top-p must be at most 1", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": True},
                {"top_p": 2},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_top_p=0.9,
            )

    def test_earlier_extra_settings_win_over_top_p(self) -> None:
        """Le trim, le rejet, la couleur et le bruit sortent avant top_p."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "top_p_slider": True},
                {"sliding_window_trim_first_frames": 11, "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trim First Frames", message)
            self.assertNotIn("Top-p", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "top_p_slider": True},
                {"sliding_window_discard_last_frames": 21, "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Discard Last Frames", message)
            self.assertNotIn("Top-p", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "color_correction": True,
                    "top_p_slider": True,
                },
                {"sliding_window_color_correction_strength": 2, "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Color Correction Strength", message)
            self.assertNotIn("Top-p", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True, "top_p_slider": True},
                {"sliding_window_overlap_noise": 151, "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertNotIn("Top-p", message)

    def test_temperature_without_audio_does_not_mask_top_p(self) -> None:
        """audio_only faux ignore une température 2. top_p 1.01 reste l'erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "audio_only": False,
                    "temperature": True,
                    "top_p_slider": True,
                },
                {"temperature": 2, "top_p": 1.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-p must be at most 1", message)
            self.assertNotIn("Temperature", message)

    def test_audio_only_is_reported_before_top_p(self) -> None:
        """audio_only vrai est refusé avant extra_settings, donc avant top_p."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "audio_only": True, "top_p_slider": True},
                {"temperature": 2, "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("audio_only", message)
            self.assertNotIn("Top-p", message)

    def test_scheduler_crash_blocks_top_p(self) -> None:
        """Un rejet « large » ou un discard_step 0 lève avant extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "top_p_slider": True},
                {"sliding_window_discard_last_frames": "large", "top_p": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "top_p_slider": True,
                    "sliding_window_defaults": {"discard_step": 0},
                },
                {"top_p": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False, "top_p_slider": True},
                {"sliding_window_discard_last_frames": "large", "top_p": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-p must be at most 1", str(caught.exception))

    def test_top_p_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand top_p sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V", "top_p_slider": True},
                {"top_p": 2},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Top-p must be at most 1", message)
            self.assertIn("code 0", message)

    def test_top_k_above_100_is_skipped(self) -> None:
        """101 dépasse 100. Le libellé coupé est Top-k. top_p 0.9 ne masque pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (101, 101.5, "101", "101.0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_k_slider": True, "top_p_slider": False},
                    {"top_k": raw, "top_p": 0.9},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Top-k must be at most 100", message)
                self.assertIn("code 0", message)
                self.assertNotIn("0 = disabled", message)
                self.assertNotIn("Top-p", message)
                self.assertNotIn("NAG Scale", message)

    def test_top_k_bounds_keep_a_video(self) -> None:
        """0, 100, le défaut 50 et la troncature 100.9 ou -0.1 restent dans la borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0, 100, 50, 100.9, -0.1, "100", "50.0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_k_slider": True},
                    {"top_k": raw},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "t2v_1.3B", "top_k_slider": True})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_top_k_string_and_type_errors(self) -> None:
        """True donne « a integer ». « large » et « 101.5 » donnent « an integer »."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": True},
                {"top_k": True},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-k must be a integer", message)
            self.assertNotIn("at most 100", message)
            for raw in ("large", "101.5"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_k_slider": True},
                    {"top_k": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Top-k must be an integer", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 100", message)

    def test_negative_and_empty_top_k(self) -> None:
        """-1 est sous 0. Une chaîne vide n'est pas une erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": True},
                {"top_k": -1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-k must be at least 0", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": True},
                {"top_k": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_slider_off_ignores_top_k(self) -> None:
        """Sans top_k_slider, un top_k 101 est ignoré. top_p_slider seul aussi.

        t2v_1.3B n'injecte pas top_k_slider. NAG vrai et NAG_scale 21 sont
        le saut suivant : le message dit NAG Scale, pas Top-k.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"top_k": 101},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": False},
                {"top_k": 101},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": 0},
                {"top_k": 101},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_p_slider": True},
                {"top_k": 101, "top_p": 0.9},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_scale": 21, "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 20", message)
            self.assertNotIn("Top-k", message)

    def test_truthy_slider_rejects_top_k(self) -> None:
        """1 et la chaîne false sont vrais pour bool(), comme WanGP."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for flag in (1, "false"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "top_k_slider": flag},
                    {"top_k": 101},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("Top-k must be at most 100", str(caught.exception))

    def test_finetune_slider_rejects_top_k(self) -> None:
        """Un finetune qui écrit le drapeau et top_k 101 fait sauter la tâche."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "top_k_slider": True},
                    "top_k": 101,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-k must be at most 100", str(caught.exception))

    def test_later_false_clears_top_k_slider(self) -> None:
        """Un JSON suivant qui écrit top_k_slider à faux retire le contrôle."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": True},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "top_k_slider": False},
                    "top_k": 101,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_top_k_overrides_the_factory(self) -> None:
        """La file à 101 remplace l'usine 50. La file à 50 répare un JSON à 101."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v_1.3B", "top_k_slider": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_top_k=101,
                )
            message = str(caught.exception)
            self.assertIn("Top-k must be at most 100", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "top_k_slider": True},
                {"top_k": 101},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_top_k=50,
            )

    def test_earlier_extra_settings_win_over_top_k(self) -> None:
        """top_p, le trim, le rejet, la couleur et le bruit sortent avant top_k."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "top_p_slider": True,
                    "top_k_slider": True,
                },
                {"top_p": 2, "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-p must be at most 1", message)
            self.assertNotIn("Top-k", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "top_k_slider": True},
                {"sliding_window_trim_first_frames": 11, "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trim First Frames", message)
            self.assertNotIn("Top-k", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "top_k_slider": True},
                {"sliding_window_discard_last_frames": 21, "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Discard Last Frames", message)
            self.assertNotIn("Top-k", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "color_correction": True,
                    "top_k_slider": True,
                },
                {"sliding_window_color_correction_strength": 2, "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Color Correction Strength", message)
            self.assertNotIn("Top-k", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "vace_class": True, "top_k_slider": True},
                {"sliding_window_overlap_noise": 151, "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("at most 150", message)
            self.assertNotIn("Top-k", message)

    def test_scheduler_crash_blocks_top_k(self) -> None:
        """Un rejet « large » ou un discard_step 0 lève avant extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "top_k_slider": True},
                {"sliding_window_discard_last_frames": "large", "top_k": 101},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "top_k_slider": True,
                    "sliding_window_defaults": {"discard_step": 0},
                },
                {"top_k": 101},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False, "top_k_slider": True},
                {"sliding_window_discard_last_frames": "large", "top_k": 101},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Top-k must be at most 100", str(caught.exception))

    def test_top_k_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand top_k sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V", "top_k_slider": True},
                {"top_k": 101},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Top-k must be at most 100", message)
            self.assertIn("code 0", message)

    def test_nag_scale_above_20_is_skipped(self) -> None:
        """21 et 20.01 dépassent 20. top_k 50 et sliding_window faux ne masquent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (21, 20.01, "21", "21.0", 20.01):
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "NAG": True,
                        "top_k_slider": False,
                        "sliding_window": False,
                    },
                    {"NAG_scale": raw, "top_k": 50, "NAG_tau": 5.01},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Scale must be at most 20", message)
                self.assertIn("code 0", message)
                self.assertNotIn("Top-k", message)
                self.assertNotIn("NAG Tau", message)

    def test_nag_scale_bounds_keep_a_video(self) -> None:
        """1, 20 et l'usine 1 passent. NAG_alpha 2 reste dans la borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (1, 20, 1.0, "1", "20.0", "  20  "):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_scale": raw, "NAG_tau": 5, "NAG_alpha": 2},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_tau": 3.5, "NAG_alpha": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_nag_scale_below_one(self) -> None:
        """0.5 est sous 1. Une chaîne vide n'est pas une erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0.5, "0.5", 0):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_scale": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("NAG Scale must be at least 1", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_scale": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_nag_scale_type_errors(self) -> None:
        """True et « large » donnent « must be a number », pas une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (True, "large"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_scale": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Scale must be a number", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 20", message)

    def test_nag_off_ignores_scale(self) -> None:
        """Sans NAG, 21 est ignoré. t2v_1.3B, bernini et animate2 n'injectent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for model in (
                {"architecture": "t2v_1.3B"},
                {"architecture": "t2v_1.3B", "NAG": False},
                {"architecture": "t2v_1.3B", "NAG": 0},
                {"architecture": "alpha_lynx"},
            ):
                _write_model(root, model, {"NAG_scale": 21, "image_refs": ["ref.png"]})
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "bernini"},
                {"NAG_scale": 21, "video_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "bernini_1.3B"},
                {"NAG_scale": 21, "video_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "animate2"},
                {
                    "NAG_scale": 21,
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_truthy_nag_rejects_scale(self) -> None:
        """1 et la chaîne false sont vrais pour bool(), comme WanGP."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for flag in (1, "false"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": flag},
                    {"NAG_scale": 21},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("NAG Scale must be at most 20", str(caught.exception))

    def test_injected_architectures_reject_nag_scale(self) -> None:
        """t2v, vace, i2v, alpha, lynx et vista4d injectent NAG au premier JSON."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            cases = (
                ({"architecture": "t2v"}, {"NAG_scale": 21}),
                ({"architecture": "vace_1.3B"}, {"NAG_scale": 21}),
                ({"architecture": "i2v"}, {"NAG_scale": 21}),
                ({"architecture": "alpha"}, {"NAG_scale": 21}),
                ({"architecture": "lynx"}, {"NAG_scale": 21, "image_refs": ["ref.png"]}),
                ({"architecture": "vista4d"}, {"NAG_scale": 21}),
            )
            for model, settings in cases:
                _write_model(root, model, settings)
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Scale must be at most 20", message, model["architecture"])
                self.assertIn(str(model["architecture"]), message)

    def test_finetune_keeps_injected_nag(self) -> None:
        """Un finetune t2v_1.3B ne retire pas le NAG injecté par t2v."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "NAG_scale": 21,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Scale must be at most 20", str(caught.exception))

    def test_finetune_t2v_after_t2v_1_3b_does_not_inject_nag(self) -> None:
        """Un t2v seulement en finetune ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v"},
                    "NAG_scale": 21,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_false_clears_nag(self) -> None:
        """Un JSON suivant qui écrit NAG à faux retire l'injection."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "NAG": False},
                    "NAG_scale": 21,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_false_on_injector_clears_nag(self) -> None:
        """Le premier JSON qui écrit NAG à faux écrase la formule."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v", "NAG": False},
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_custom_nag_scale_bounds(self) -> None:
        """Un objet model.NAG_scale change le maximum, le minimum et le libellé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": {"max": 30}},
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": {"max": 10}},
                {"NAG_scale": 11},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 10", message)
            self.assertNotIn("at most 20", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "NAG_scale": {"label": "Custom NAG (boost)"},
                },
                {"NAG_scale": 21},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom NAG must be at most 20", message)
            self.assertNotIn("NAG Scale", message)
            self.assertNotIn("boost", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": {"max": "nope"}},
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": {"min": 2}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Scale must be at least 2", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "NAG_scale": {"min": 30, "max": 10},
                },
                {"NAG_scale": 21},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at least 30", message)
            self.assertNotIn("at most 10", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": {"label": None}},
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": 5},
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_nag_scale_overrides_the_factory(self) -> None:
        """La file à 21 remplace l'usine 1. La file à 1 répare un JSON à 21."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v_1.3B", "NAG": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_nag_scale=21,
                )
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 20", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_nag_scale=1,
            )

    def test_earlier_extra_settings_win_over_nag_scale(self) -> None:
        """top_k, top_p et le trim sortent avant NAG_scale."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "top_k_slider": True},
                {"top_k": 101, "NAG_scale": 21},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-k must be at most 100", message)
            self.assertNotIn("NAG Scale", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "top_p_slider": True},
                {"top_p": 2, "NAG_scale": 21},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-p must be at most 1", message)
            self.assertNotIn("NAG Scale", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "sliding_window": True},
                {"sliding_window_trim_first_frames": 11, "NAG_scale": 21},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trim First Frames", message)
            self.assertNotIn("NAG Scale", message)

    def test_scheduler_crash_blocks_nag_scale(self) -> None:
        """Un rejet « large » ou un discard_step 0 lève avant extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "NAG": True},
                {"sliding_window_discard_last_frames": "large", "NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "NAG": True,
                    "sliding_window_defaults": {"discard_step": 0},
                },
                {"NAG_scale": 21},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False, "NAG": True},
                {"sliding_window_discard_last_frames": "large", "NAG_scale": 21},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Scale must be at most 20", str(caught.exception))

    def test_nag_scale_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand NAG_scale sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V", "NAG": True},
                {"NAG_scale": 21},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 20", message)
            self.assertIn("code 0", message)

    def test_nag_tau_above_5_is_skipped(self) -> None:
        """5.01 dépasse 5. NAG_alpha 3 est hors borne, mais l'erreur de tau sort avant."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (5.01, "5.01", "5.010", 6):
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "NAG": True,
                        "sliding_window": False,
                    },
                    {"NAG_tau": raw, "NAG_scale": 1, "NAG_alpha": 3},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Tau must be at most 5", message)
                self.assertIn("code 0", message)
                self.assertNotIn("NAG Scale", message)
                self.assertNotIn("NAG Alpha", message)

    def test_nag_tau_bounds_keep_a_video(self) -> None:
        """1, 5 et l'usine 3.5 passent. NAG_alpha 2 reste dans la borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (1, 5, 3.5, "5.0", "  5  ", "3.5"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_tau": raw, "NAG_alpha": 2, "NAG_scale": 20},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_alpha": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_nag_tau_below_one(self) -> None:
        """0.5 est sous 1. Une chaîne vide n'est pas une erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0.5, "0.5", 0):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_tau": raw, "NAG_scale": 1},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("NAG Tau must be at least 1", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_tau": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_nag_tau_type_errors(self) -> None:
        """True et « large » donnent « must be a number », pas une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (True, "large"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_tau": raw, "NAG_scale": 1},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Tau must be a number", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 5", message)

    def test_nag_off_ignores_tau(self) -> None:
        """Sans NAG, 5.01 est ignoré. t2v_1.3B, bernini et animate2 n'injectent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for model in (
                {"architecture": "t2v_1.3B"},
                {"architecture": "t2v_1.3B", "NAG": False},
                {"architecture": "t2v_1.3B", "NAG": 0},
                {"architecture": "alpha_lynx"},
            ):
                _write_model(root, model, {"NAG_tau": 5.01, "image_refs": ["ref.png"]})
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "bernini"},
                {"NAG_tau": 5.01, "video_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "bernini_1.3B"},
                {"NAG_tau": 5.01, "video_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "animate2"},
                {
                    "NAG_tau": 5.01,
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_truthy_nag_rejects_tau(self) -> None:
        """1 et la chaîne false sont vrais pour bool(), comme WanGP."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for flag in (1, "false"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": flag},
                    {"NAG_tau": 5.01, "NAG_scale": 1},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("NAG Tau must be at most 5", str(caught.exception))

    def test_injected_architectures_reject_nag_tau(self) -> None:
        """t2v, vace, i2v, alpha, lynx et vista4d injectent NAG au premier JSON."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            cases = (
                ({"architecture": "t2v"}, {"NAG_tau": 5.01, "NAG_scale": 1}),
                ({"architecture": "vace_1.3B"}, {"NAG_tau": 5.01, "NAG_scale": 1}),
                ({"architecture": "i2v"}, {"NAG_tau": 5.01, "NAG_scale": 1}),
                ({"architecture": "alpha"}, {"NAG_tau": 5.01, "NAG_scale": 1}),
                (
                    {"architecture": "lynx"},
                    {"NAG_tau": 5.01, "NAG_scale": 1, "image_refs": ["ref.png"]},
                ),
                ({"architecture": "vista4d"}, {"NAG_tau": 5.01, "NAG_scale": 1}),
            )
            for model, settings in cases:
                _write_model(root, model, settings)
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Tau must be at most 5", message, model["architecture"])
                self.assertIn(str(model["architecture"]), message)

    def test_finetune_keeps_injected_nag_for_tau(self) -> None:
        """Un finetune t2v_1.3B ne retire pas le NAG injecté par t2v."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "NAG_tau": 5.01,
                    "NAG_scale": 1,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Tau must be at most 5", str(caught.exception))

    def test_finetune_t2v_after_t2v_1_3b_does_not_inject_nag_for_tau(self) -> None:
        """Un t2v seulement en finetune ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v"},
                    "NAG_tau": 5.01,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_false_clears_nag_for_tau(self) -> None:
        """Un JSON suivant qui écrit NAG à faux retire l'injection."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "NAG": False},
                    "NAG_tau": 5.01,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_false_on_injector_clears_nag_for_tau(self) -> None:
        """Le premier JSON qui écrit NAG à faux écrase la formule."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v", "NAG": False},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_custom_nag_tau_bounds(self) -> None:
        """Un objet model.NAG_tau change le maximum, le minimum et le libellé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"max": 6}},
                {"NAG_tau": 5.01, "NAG_scale": 1},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"max": 4}},
                {"NAG_tau": 5, "NAG_scale": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Tau must be at most 4", message)
            self.assertNotIn("at most 5", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "NAG_tau": {"label": "Custom Tau (boost)"},
                },
                {"NAG_tau": 5.01, "NAG_scale": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom Tau must be at most 5", message)
            self.assertNotIn("NAG Tau", message)
            self.assertNotIn("boost", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"max": "nope"}},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"max": True}},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"min": 4}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Tau must be at least 4", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "NAG_tau": {"min": 6, "max": 4},
                },
                {"NAG_tau": 5.01, "NAG_scale": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Tau must be at least 6", message)
            self.assertNotIn("at most 4", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"label": "("}},
                {"NAG_tau": 5.01, "NAG_scale": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG_tau must be at most 5", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": {"label": None}},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": 4},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": 5},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": 1},
                {"NAG_tau": 5.01, "NAG_scale": 1},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_nag_tau_overrides_the_factory(self) -> None:
        """La file à 5.01 remplace l'usine 3.5. La file à 3.5 répare un JSON à 5.01."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v_1.3B", "NAG": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_nag_tau=5.01,
                )
            message = str(caught.exception)
            self.assertIn("NAG Tau must be at most 5", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_nag_tau=3.5,
            )

    def test_earlier_extra_settings_win_over_nag_tau(self) -> None:
        """NAG_scale, top_k et le trim sortent avant NAG_tau."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_scale": 21, "NAG_tau": 5.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 20", message)
            self.assertNotIn("NAG Tau", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "top_k_slider": True},
                {"top_k": 101, "NAG_tau": 5.01, "NAG_scale": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-k must be at most 100", message)
            self.assertNotIn("NAG Tau", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "sliding_window": True},
                {"sliding_window_trim_first_frames": 11, "NAG_tau": 5.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trim First Frames", message)
            self.assertNotIn("NAG Tau", message)

    def test_scheduler_crash_blocks_nag_tau(self) -> None:
        """Un rejet « large » ou un discard_step 0 lève avant extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "NAG": True},
                {"sliding_window_discard_last_frames": "large", "NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "NAG": True,
                    "sliding_window_defaults": {"discard_step": 0},
                },
                {"NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False, "NAG": True},
                {"sliding_window_discard_last_frames": "large", "NAG_tau": 5.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Tau must be at most 5", str(caught.exception))

    def test_nag_tau_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand NAG_tau sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V", "NAG": True},
                {"NAG_tau": 5.01, "NAG_scale": 1},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("NAG Tau must be at most 5", message)
            self.assertIn("code 0", message)

    def test_nag_alpha_above_2_is_skipped(self) -> None:
        """2.01 dépasse 2. Un tau dans la borne ne masque pas ce saut."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (2.01, "2.01", "2.010", 3):
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "NAG": True,
                        "sliding_window": False,
                    },
                    {"NAG_alpha": raw, "NAG_scale": 1, "NAG_tau": 5},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Alpha must be at most 2", message)
                self.assertIn("code 0", message)
                self.assertNotIn("NAG Scale", message)
                self.assertNotIn("NAG Tau", message)
                self.assertNotIn("Pace", message)

    def test_nag_alpha_bounds_keep_a_video(self) -> None:
        """0, 2 et l'usine 0.5 passent. pace 9 n'est pas ce saut."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0, 2, 0.5, "2.0", "  2  ", "0.5", -0.0):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_alpha": raw, "NAG_tau": 5, "NAG_scale": 20, "pace": 9},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"pace": 9},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_nag_alpha_below_zero(self) -> None:
        """-0.01 est sous 0. Une chaîne vide n'est pas une erreur."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (-0.01, "-0.01"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_alpha": raw, "NAG_scale": 1, "NAG_tau": 3.5},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("NAG Alpha must be at least 0", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_alpha": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_alpha": "  "},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_nag_alpha_type_errors(self) -> None:
        """True et « large » donnent « must be a number », pas une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (True, "large"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": True},
                    {"NAG_alpha": raw, "NAG_scale": 1, "NAG_tau": 3.5},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Alpha must be a number", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 2", message)

    def test_nag_off_ignores_alpha(self) -> None:
        """Sans NAG, 2.01 est ignoré. t2v_1.3B, bernini et animate2 n'injectent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for model in (
                {"architecture": "t2v_1.3B"},
                {"architecture": "t2v_1.3B", "NAG": False},
                {"architecture": "t2v_1.3B", "NAG": 0},
                {"architecture": "alpha_lynx"},
            ):
                _write_model(root, model, {"NAG_alpha": 2.01, "image_refs": ["ref.png"]})
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "bernini"},
                {"NAG_alpha": 2.01, "video_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "bernini_1.3B"},
                {"NAG_alpha": 2.01, "video_prompt_type": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "animate2"},
                {
                    "NAG_alpha": 2.01,
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_truthy_nag_rejects_alpha(self) -> None:
        """1 et la chaîne false sont vrais pour bool(), comme WanGP."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for flag in (1, "false"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "NAG": flag},
                    {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                self.assertIn("NAG Alpha must be at most 2", str(caught.exception))

    def test_injected_architectures_reject_nag_alpha(self) -> None:
        """t2v, vace, i2v, alpha, lynx et vista4d injectent NAG au premier JSON."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            cases = (
                ({"architecture": "t2v"}, {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5}),
                ({"architecture": "vace_1.3B"}, {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5}),
                ({"architecture": "i2v"}, {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5}),
                ({"architecture": "alpha"}, {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5}),
                (
                    {"architecture": "lynx"},
                    {
                        "NAG_alpha": 2.01,
                        "NAG_scale": 1,
                        "NAG_tau": 3.5,
                        "image_refs": ["ref.png"],
                    },
                ),
                ({"architecture": "vista4d"}, {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5}),
            )
            for model, settings in cases:
                _write_model(root, model, settings)
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("NAG Alpha must be at most 2", message, model["architecture"])
                self.assertIn(str(model["architecture"]), message)

    def test_finetune_keeps_injected_nag_for_alpha(self) -> None:
        """Un finetune t2v_1.3B ne retire pas le NAG injecté par t2v."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Alpha must be at most 2", str(caught.exception))

    def test_finetune_t2v_after_t2v_1_3b_does_not_inject_nag_for_alpha(self) -> None:
        """Un t2v seulement en finetune ne rappelle pas query_model_def."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                finetune={
                    "model": {"architecture": "t2v"},
                    "NAG_alpha": 2.01,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_later_false_clears_nag_for_alpha(self) -> None:
        """Un JSON suivant qui écrit NAG à faux retire l'injection."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v"},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "NAG": False},
                    "NAG_alpha": 2.01,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_explicit_false_on_injector_clears_nag_for_alpha(self) -> None:
        """Le premier JSON qui écrit NAG à faux écrase la formule."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v", "NAG": False},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_custom_nag_alpha_bounds(self) -> None:
        """Un objet model.NAG_alpha change le maximum, le minimum et le libellé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"max": 3}},
                {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"max": 1}},
                {"NAG_alpha": 2, "NAG_scale": 1, "NAG_tau": 3.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Alpha must be at most 1", message)
            self.assertNotIn("at most 2", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "NAG_alpha": {"label": "Custom Alpha (boost)"},
                },
                {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom Alpha must be at most 2", message)
            self.assertNotIn("NAG Alpha", message)
            self.assertNotIn("boost", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"max": "nope"}},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"max": True}},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"min": 1}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Alpha must be at least 1", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "NAG_alpha": {"min": 3, "max": 1},
                },
                {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Alpha must be at least 3", message)
            self.assertNotIn("at most 1", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"label": "("}},
                {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG_alpha must be at most 2", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": {"label": None}},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_alpha": 1},
                {"NAG_alpha": 2.01, "NAG_tau": 5.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_tau": 4},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "NAG_scale": 5},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_nag_alpha_overrides_the_factory(self) -> None:
        """La file à 2.01 remplace l'usine 0.5. La file à 0.5 répare un JSON à 2.01."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(root, {"architecture": "t2v_1.3B", "NAG": True})
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_nag_alpha=2.01,
                )
            message = str(caught.exception)
            self.assertIn("NAG Alpha must be at most 2", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_nag_alpha=0.5,
            )

    def test_earlier_extra_settings_win_over_nag_alpha(self) -> None:
        """NAG_tau, NAG_scale, top_k et le trim sortent avant NAG_alpha."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_tau": 5.01, "NAG_alpha": 2.01, "NAG_scale": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Tau must be at most 5", message)
            self.assertNotIn("NAG Alpha", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {"NAG_scale": 21, "NAG_alpha": 2.01, "NAG_tau": 3.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Scale must be at most 20", message)
            self.assertNotIn("NAG Alpha", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "top_k_slider": True},
                {"top_k": 101, "NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Top-k must be at most 100", message)
            self.assertNotIn("NAG Alpha", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True, "sliding_window": True},
                {"sliding_window_trim_first_frames": 11, "NAG_alpha": 2.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Trim First Frames", message)
            self.assertNotIn("NAG Alpha", message)

    def test_scheduler_crash_blocks_nag_alpha(self) -> None:
        """Un rejet « large » ou un discard_step 0 lève avant extra_settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True, "NAG": True},
                {"sliding_window_discard_last_frames": "large", "NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "NAG": True,
                    "sliding_window_defaults": {"discard_step": 0},
                },
                {"NAG_alpha": 2.01},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": False, "NAG": True},
                {"sliding_window_discard_last_frames": "large", "NAG_alpha": 2.01},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("NAG Alpha must be at most 2", str(caught.exception))

    def test_nag_alpha_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand NAG_alpha sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V", "NAG": True},
                {"NAG_alpha": 2.01, "NAG_scale": 1, "NAG_tau": 3.5},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("NAG Alpha must be at most 2", message)
            self.assertIn("code 0", message)

    def test_pace_above_1_is_skipped(self) -> None:
        """9, « 9 » et 1.01 dépassent 1. La racine pace n'est pas lue."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (9, "9", 1.01, "  2  "):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                    {"custom_settings": {"pace": raw}},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Pace must be at most 1", message)
                self.assertIn("code 0", message)
                self.assertNotIn("Exaggeration", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"pace": 9},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_pace_bounds_keep_a_video(self) -> None:
        """0.2, 1, le vide et une valeur absente passent. Sans l'id, exaggeration 3 n'est pas lu."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0.2, 1, "0.2", "  0.5  ", ""):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                    {"custom_settings": {"pace": raw, "exaggeration": 3}},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"exaggeration": 3}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_pace_below_min(self) -> None:
        """0.19 est sous 0.2."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"pace": 0.19}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be at least 0.2", str(caught.exception))

    def test_pace_type_errors(self) -> None:
        """True et « large » donnent « must be a number », pas une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (True, "large"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                    {"custom_settings": {"pace": raw}},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Pace must be a number", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 1", message)

    def test_pace_hidden_ids_are_ignored(self) -> None:
        """Pace, une valeur qui n'est pas une liste, et le sixième objet ne comptent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            hidden = (
                {"architecture": "t2v_1.3B"},
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "Pace"}]},
                {"architecture": "t2v_1.3B", "custom_settings": {"id": "pace"}},
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "x"}] * 5 + [{"id": "pace"}],
                },
            )
            for model in hidden:
                _write_model(root, model, {"custom_settings": {"pace": 9}, "pace": 9})
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": ["pace"]},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_pace_scan_keeps_the_fifth_and_skips_non_dicts(self) -> None:
        """Le cinquième objet compte. Un élément qui n'est pas un dict est sauté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "x"}] * 4 + [{"id": " pace "}],
                },
                {"custom_settings": {"pace": 9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be at most 1", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": ["nope", {"id": "pace"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be at most 1", str(caught.exception))

    def test_pace_custom_bounds_and_label(self) -> None:
        """L'objet change le maximum, le minimum et le libellé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "max": 0.5, "label": "Custom Pace (boost)"},
                    ],
                },
                {"custom_settings": {"pace": 0.6}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom Pace must be at most 0.5", message)
            self.assertNotIn("boost", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "label": "("}],
                },
                {"custom_settings": {"pace": 9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("pace must be at most 1", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "name": "Named Pace"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Named Pace must be at most 1", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "label": "  "}],
                },
                {"custom_settings": {"pace": 9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("pace must be at most 1", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "min": 0.5, "max": 0.4}],
                },
                {"custom_settings": {"pace": 0.1}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Pace must be at least 0.5", message)
            self.assertNotIn("at most 0.4", message)
            for broken in ({"max": "nope"}, {"max": True}):
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "custom_settings": [{"id": "pace", **broken}],
                    },
                    {"custom_settings": {"pace": 9}},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "text"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_pace_integer_type(self) -> None:
        """Le type integer stringifie 1.5. 1.0 passe, 2.0 reste une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": 1.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Pace must be an integer", message)
            self.assertIn("code 0", message)
            self.assertNotIn("at least 0.2", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": 0.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be an integer", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": 1.0}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": 2.0}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be at most 1", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": "large"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be an integer", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": "1.5"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be an integer", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "integer"}],
                },
                {"custom_settings": {"pace": True}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be a integer", str(caught.exception))

    def test_pace_finetune_keeps_or_clears_the_id(self) -> None:
        """Un finetune qui ne réécrit pas la liste laisse l'id. Une liste vide l'efface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"pace": 9}},
                finetune={"model": {"architecture": "t2v_1.3B", "name": "suite"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Pace must be at most 1", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"pace": 9}},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "custom_settings": []},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"pace": 9}},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "custom_settings": {"pace": 0.5},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_custom_settings_override_pace(self) -> None:
        """La file remplace le dict. None ne l'efface pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"pace": 0.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_settings={"pace": 9},
                )
            message = str(caught.exception)
            self.assertIn("Pace must be at most 1", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "pace"}]},
                {"custom_settings": {"pace": 9}},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_custom_settings={"pace": 0.5},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_settings=None,
                )
            self.assertIn("Pace must be at most 1", str(caught.exception))

    def test_earlier_errors_win_over_pace(self) -> None:
        """NAG_alpha et le frame scheduler sortent avant pace."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "custom_settings": [{"id": "pace"}],
                },
                {
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                    "custom_settings": {"pace": 9},
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Alpha must be at most 2", message)
            self.assertNotIn("Pace must", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "custom_settings": [{"id": "pace"}],
                },
                {
                    "sliding_window_discard_last_frames": "large",
                    "custom_settings": {"pace": 9},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": False,
                    "NAG_alpha": 4,
                    "custom_settings": [{"id": "pace"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "switch_threshold": 4,
                    "custom_settings": [{"id": "pace"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "temperature_slider": 4,
                    "custom_settings": [{"id": "pace"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_pace_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand pace sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "custom_settings": [{"id": "pace"}],
                },
                {"custom_settings": {"pace": 9}},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Pace must be at most 1", message)
            self.assertIn("code 0", message)

    def test_exaggeration_above_2_is_skipped(self) -> None:
        """3, « 3 » et 2.01 dépassent 2. La racine exaggeration n'est pas lue."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (3, "3", 2.01, "  3  "):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                    {"custom_settings": {"exaggeration": raw, "pace": 0.5}},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Exaggeration must be at most 2", message)
                self.assertIn("code 0", message)
                self.assertNotIn("Pace must", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"exaggeration": 3},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_exaggeration_bounds_keep_a_video(self) -> None:
        """0.25, 2, le vide et une valeur absente passent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0.25, 2, "0.25", "  1  ", "", "2.0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                    {"custom_settings": {"exaggeration": raw}},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"pace": 0.5}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace"}, {"id": "exaggeration"}],
                },
                {"custom_settings": {"pace": 0.5, "exaggeration": 1}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_exaggeration_below_min(self) -> None:
        """0.24 est sous 0.25."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"exaggeration": 0.24}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be at least 0.25", str(caught.exception))

    def test_exaggeration_type_errors(self) -> None:
        """True et « large » donnent « must be a number », pas une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (True, "large"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                    {"custom_settings": {"exaggeration": raw}},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Exaggeration must be a number", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 2", message)

    def test_exaggeration_hidden_ids_are_ignored(self) -> None:
        """Exaggeration, une valeur qui n'est pas une liste, et le sixième objet ne comptent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            hidden = (
                {"architecture": "t2v_1.3B"},
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "Exaggeration"}]},
                {"architecture": "t2v_1.3B", "custom_settings": {"id": "exaggeration"}},
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "x"}] * 5 + [{"id": "exaggeration"}],
                },
            )
            for model in hidden:
                _write_model(
                    root,
                    model,
                    {"custom_settings": {"exaggeration": 3}, "exaggeration": 3},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": ["exaggeration"]},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_exaggeration_scan_keeps_the_fifth_and_skips_non_dicts(self) -> None:
        """Le cinquième objet compte. Un élément qui n'est pas un dict est sauté."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "x"}] * 4 + [{"id": " exaggeration "}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be at most 2", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": ["nope", {"id": "exaggeration"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be at most 2", str(caught.exception))

    def test_exaggeration_custom_bounds_and_label(self) -> None:
        """L'objet change le maximum, le minimum et le libellé."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "exaggeration", "max": 1, "label": "Custom Exaggeration (boost)"},
                    ],
                },
                {"custom_settings": {"exaggeration": 1.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom Exaggeration must be at most 1", message)
            self.assertNotIn("boost", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "label": "("}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("exaggeration must be at most 2", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "name": "Named Exaggeration"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Named Exaggeration must be at most 2", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "label": "  "}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("exaggeration must be at most 2", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "min": 1, "max": 0.5}],
                },
                {"custom_settings": {"exaggeration": 0.1}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Exaggeration must be at least 1", message)
            self.assertNotIn("at most 0.5", message)
            for broken in ({"max": "nope"}, {"max": True}):
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "custom_settings": [{"id": "exaggeration", **broken}],
                    },
                    {"custom_settings": {"exaggeration": 3}},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "text"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_exaggeration_integer_type(self) -> None:
        """Le type integer stringifie 1.5 et 2.9. 2.0 reste dans la borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (1.5, 2.9, 0.1, 3.2):
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "custom_settings": [{"id": "exaggeration", "type": "integer"}],
                    },
                    {"custom_settings": {"exaggeration": raw}},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn("Exaggeration must be an integer", message)
                self.assertIn("code 0", message)
                self.assertNotIn("at most 2", message)
                self.assertNotIn("at least 0.25", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "integer"}],
                },
                {"custom_settings": {"exaggeration": 2.0}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "integer"}],
                },
                {"custom_settings": {"exaggeration": "large"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be an integer", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "integer"}],
                },
                {"custom_settings": {"exaggeration": "1.5"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be an integer", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "integer"}],
                },
                {"custom_settings": {"exaggeration": True}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be a integer", str(caught.exception))

    def test_collect_int_type_rejects_a_fraction_before_bounds(self) -> None:
        """Le type int refuse 1.5 avant les bornes. Un entier 9 reste une borne."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "int"}],
                },
                {"custom_settings": {"pace": 1.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom Setting 1 Expected an integer value.", message)
            self.assertIn("code 0", message)
            self.assertNotIn("at most 1", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "int", "label": "Custom Pace (boost)"},
                    ],
                },
                {"custom_settings": {"pace": 1.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Custom Pace (boost) Expected an integer value.", message)
            self.assertNotIn("Custom Pace must", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": ["nope", {"id": "pace", "type": "int"}],
                },
                {"custom_settings": {"pace": 0.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Custom Setting 2 Expected an integer value.", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "int"}],
                },
                {"custom_settings": {"pace": 1.0}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "int"}],
                },
                {"custom_settings": {"pace": 9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Pace must be at most 1", message)
            self.assertNotIn("Expected an integer value.", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "int"}],
                },
                {"custom_settings": {"exaggeration": 2.9}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(
                "Custom Setting 1 Expected an integer value.",
                str(caught.exception),
            )
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "exaggeration", "type": "int"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Exaggeration must be at most 2", message)
            self.assertNotIn("Expected an integer value.", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "integer"},
                        {"id": "exaggeration"},
                    ],
                },
                {"custom_settings": {"pace": 1.5, "exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Pace must be an integer", message)
            self.assertNotIn("Exaggeration must", message)

    def test_exaggeration_finetune_keeps_or_clears_the_id(self) -> None:
        """Un finetune qui ne réécrit pas la liste laisse l'id. Une liste vide l'efface."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"exaggeration": 3}},
                finetune={"model": {"architecture": "t2v_1.3B", "name": "suite"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Exaggeration must be at most 2", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"exaggeration": 3}},
                finetune={
                    "model": {"architecture": "t2v_1.3B", "custom_settings": []},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"exaggeration": 3}},
                finetune={
                    "model": {"architecture": "t2v_1.3B"},
                    "custom_settings": {"exaggeration": 1},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_queue_custom_settings_override_exaggeration(self) -> None:
        """La file remplace le dict. None ne l'efface pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"exaggeration": 1}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_settings={"exaggeration": 3},
                )
            message = str(caught.exception)
            self.assertIn("Exaggeration must be at most 2", message)
            self.assertIn("file", message.lower())
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_settings": [{"id": "exaggeration"}]},
                {"custom_settings": {"exaggeration": 3}},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_custom_settings={"exaggeration": 1},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_settings=None,
                )
            self.assertIn("Exaggeration must be at most 2", str(caught.exception))

    def test_earlier_errors_win_over_exaggeration(self) -> None:
        """pace, NAG_alpha et le frame scheduler sortent avant exaggeration."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace"}, {"id": "exaggeration"}],
                },
                {"custom_settings": {"pace": 9, "exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Pace must be at most 1", message)
            self.assertNotIn("Exaggeration must", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "custom_settings": [{"id": "exaggeration"}],
                },
                {
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                    "custom_settings": {"exaggeration": 3},
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("NAG Alpha must be at most 2", message)
            self.assertNotIn("Exaggeration must", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "custom_settings": [{"id": "exaggeration"}],
                },
                {
                    "sliding_window_discard_last_frames": "large",
                    "custom_settings": {"exaggeration": 3},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": False,
                    "NAG_alpha": 4,
                    "custom_settings": [{"id": "exaggeration"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "switch_threshold": 4,
                    "custom_settings": [{"id": "exaggeration"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_both_ids_report_exaggeration_when_pace_is_in_range(self) -> None:
        """Les deux ids : pace 0.5 laisse l'erreur d'exaggeration 3."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace"}, {"id": "exaggeration"}],
                },
                {"custom_settings": {"pace": 0.5, "exaggeration": 3}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Exaggeration must be at most 2", message)
            self.assertNotIn("Pace must", message)

    def test_exaggeration_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand exaggeration sort de la borne."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "custom_settings": [{"id": "exaggeration"}],
                },
                {"custom_settings": {"exaggeration": 3}},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Exaggeration must be at most 2", message)
            self.assertIn("code 0", message)

    def test_dropdown_outside_choices_is_skipped(self) -> None:
        """Une valeur hors liste saute avant extra_settings. Le libellé n'est pas tronqué."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {"custom_settings": {"pace": "nope"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn(
                "Custom Setting 1 Expected one of the dropdown choices.",
                message,
            )
            self.assertIn("code 0", message)
            self.assertNotIn("Pace must", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {
                            "id": "pace",
                            "type": "dropdown",
                            "label": "Custom Pace (boost)",
                            "choices": ["slow", "fast"],
                        }
                    ],
                },
                {"custom_settings": {"pace": "nope"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn(
                "Custom Pace (boost) Expected one of the dropdown choices.",
                message,
            )
            self.assertIn("boost", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        "nope",
                        {
                            "id": "pace",
                            "type": "dropdown",
                            "choices": [["Slow", "slow"]],
                        },
                    ],
                },
                {"custom_settings": {"pace": "fast"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(
                "Custom Setting 2 Expected one of the dropdown choices.",
                str(caught.exception),
            )
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "NAG": True,
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                        {"id": "exaggeration"},
                    ],
                },
                {
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                    "custom_settings": {"pace": "nope", "exaggeration": 3},
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Expected one of the dropdown choices.", message)
            self.assertNotIn("NAG Alpha", message)
            self.assertNotIn("Exaggeration must", message)

    def test_dropdown_keeps_a_video_when_the_value_is_allowed(self) -> None:
        """Égalité, str, vide, choix vides, menu masqué et sixième objet passent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            allowed = (
                {"pace": "slow"},
                {"pace": " fast "},
                {"pace": "slow"},
                {"pace": None},
                {"pace": ""},
                {"pace": "   "},
                {"pace": 1},
                {"pace": "1"},
                {},
            )
            for raw in allowed[:6]:
                _write_model(
                    root,
                    {
                        "architecture": "t2v_1.3B",
                        "custom_settings": [
                            {
                                "id": "pace",
                                "type": "dropdown",
                                "choices": ["slow", "fast"],
                            }
                        ],
                    },
                    {"custom_settings": raw},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": [["Slow", "slow"]]},
                    ],
                },
                {"custom_settings": {"pace": "slow"}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": [["Slow", "slow"]]},
                    ],
                },
                {"custom_settings": {"pace": "Slow"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Expected one of the dropdown choices.", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": [1, 2]},
                    ],
                },
                {"custom_settings": {"pace": "1"}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            for model in (
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": []},
                    ],
                },
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": "slow"},
                    ],
                },
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown"},
                    ],
                },
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "x"}] * 5
                    + [
                        {
                            "id": "pace",
                            "type": "dropdown",
                            "choices": ["slow", "fast"],
                        }
                    ],
                },
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {
                            "id": "pace",
                            "type": "dropdown",
                            "choices": ["slow", "fast"],
                            "video_prompt_type": "V",
                        }
                    ],
                },
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {
                            "id": "pace",
                            "type": "dropdown",
                            "choices": ["slow", "fast"],
                            "model_mode": [2],
                        }
                    ],
                },
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {
                            "id": "Pace",
                            "type": "dropdown",
                            "choices": ["slow", "fast"],
                        }
                    ],
                },
            ):
                items = model["custom_settings"]
                bag: dict[str, object] = {"pace": "nope"}
                if (
                    isinstance(items, list)
                    and items
                    and isinstance(items[0], dict)
                    and items[0].get("id") == "Pace"
                ):
                    bag = {"Pace": "nope"}
                _write_model(root, model, {"custom_settings": bag})
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "sliding_window": True,
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {
                    "sliding_window_discard_last_frames": "large",
                    "custom_settings": {"pace": "nope"},
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "int"},
                        {"id": "mood", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {"custom_settings": {"pace": 1.5, "mood": "nope"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Expected an integer value.", message)
            self.assertNotIn("dropdown choices", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {
                            "id": "Pace",
                            "type": "dropdown",
                            "choices": ["slow", "fast"],
                        }
                    ],
                },
                {"custom_settings": {"pace": "nope"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Expected one of the dropdown choices.", str(caught.exception))

    def test_dropdown_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le menu n'a pas la valeur."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "name": "T2V",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {"custom_settings": {"pace": "nope"}},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Expected one of the dropdown choices.", message)
            self.assertIn("code 0", message)

    def test_custom_sigmas_text_is_skipped(self) -> None:
        """Une chaîne illisible saute après le parse custom. Wan injecte le drapeau."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            cases = (
                ("nope", "Custom Sigmas phase 1: enter numbers separated by commas."),
                (12, "Custom Sigmas must be text:"),
                ("0.2,0.9", "values must be in strictly descending order."),
                ("1,0.5", "from 0 (inclusive) to 1 (exclusive)."),
                ("0", "provide at least one positive sigma"),
                ("0.9;0.2", "requires 1 phase slots separated by 0 ';'"),
                ("nan", "from 0 (inclusive) to 1 (exclusive)."),
                ("inf", "from 0 (inclusive) to 1 (exclusive)."),
            )
            for raw, expected in cases:
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": raw},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                message = str(caught.exception)
                self.assertIn(expected, message)
                self.assertIn("code 0", message)
                self.assertNotIn("NAG Alpha", message)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas": "false"},
                {"custom_sigmas": "nope"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("enter numbers separated by commas.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("requires 2 phase slots separated by 1 ';'", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9", "guidance_phases": "2~"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("requires 2 phase slots separated by 1 ';'", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;nope", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(
                "Custom Sigmas phase 2: enter numbers separated by commas.",
                str(caught.exception),
            )
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2", "guidance_phases": "1.5"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("invalid literal for int()", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {
                    "custom_sigmas": "nope",
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("enter numbers separated by commas.", message)
            self.assertNotIn("NAG Alpha", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {"custom_sigmas": "nope", "custom_settings": {"pace": "nope"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Expected one of the dropdown choices.", message)
            self.assertNotIn("Custom Sigmas", message)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [{"id": "pace", "type": "int"}],
                },
                {"custom_sigmas": "nope", "custom_settings": {"pace": 1.5}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            message = str(caught.exception)
            self.assertIn("Expected an integer value.", message)
            self.assertNotIn("Custom Sigmas", message)

    def test_custom_sigmas_keeps_a_video_when_the_schedule_is_usable(self) -> None:
        """Vide, espaces, ordre descendant et drapeau faux ne sautent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in ("", "   ", "0.9,0.2", "0.9, 0.2", "0.9,0", " 0.9,0 "):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": raw},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(root, {"architecture": "t2v_1.3B", "name": "T2V"})
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            for model in (
                {"architecture": "t2v_1.3B", "custom_sigmas": False},
                {"architecture": "t2v_1.3B", "custom_sigmas": 0},
                {"architecture": "t2v_1.3B", "custom_sigmas": ""},
                {"architecture": "plain_arch"},
            ):
                _write_model(root, model, {"custom_sigmas": "nope"})
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "nope"},
                finetune={"model": {"custom_sigmas": False}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "nope"},
                finetune={"model": {"architecture": "plain_arch"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("enter numbers separated by commas.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {
                    "custom_sigmas": "nope",
                    "sliding_window_discard_last_frames": "large",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "nope", "guidance_phases": None},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2"},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_custom_sigmas="",
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_sigmas="nope",
                )
            self.assertIn("enter numbers separated by commas.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "nope"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_sigmas=None,
                )
            self.assertIn("enter numbers separated by commas.", str(caught.exception))

    def test_custom_sigmas_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand les sigmas ne sont pas un calendrier."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V"},
                {"custom_sigmas": "nope"},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("enter numbers separated by commas.", message)
            self.assertIn("code 0", message)

    def test_incompatible_sigma_scheduler_is_skipped(self) -> None:
        """lcm et causvid sautent quand un créneau de sigmas est défini."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            message = (
                "The selected scheduler is not compatible with Custom Sigmas. "
                "Clear Custom Sigmas or select a compatible scheduler: "
                "Default, unipc, euler, dpm++."
            )
            for solver in ("lcm", "causvid", "LCM", " unipc", None, 12):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": "0.9,0.2", "sample_solver": solver},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                text = str(caught.exception)
                self.assertIn(message, text)
                self.assertIn("code 0", text)
                self.assertNotIn("NAG Alpha", text)
                self.assertNotIn("descending order between 0 and 1", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0", "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("not compatible with Custom Sigmas", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("not compatible with Custom Sigmas", text)
            self.assertNotIn("descending order between 0 and 1", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas": "false"},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("not compatible with Custom Sigmas", str(caught.exception))
            _write_model(
                root,
                {"architecture": "plain_arch", "custom_sigmas": True, "custom_sigmas_schedulers": []},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "unipc"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("scheduler: .", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "plain_arch",
                    "custom_sigmas": True,
                    "custom_sigmas_schedulers": ["", "unipc"],
                },
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Default, unipc.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {
                    "custom_sigmas": "0.9,0.2",
                    "sample_solver": "lcm",
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("not compatible with Custom Sigmas", text)
            self.assertNotIn("NAG Alpha", text)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {
                    "custom_sigmas": "0.9,0.2",
                    "sample_solver": "lcm",
                    "custom_settings": {"pace": "nope"},
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("Expected one of the dropdown choices.", text)
            self.assertNotIn("not compatible with Custom Sigmas", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "nope", "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("enter numbers separated by commas.", text)
            self.assertNotIn("not compatible with Custom Sigmas", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
                finetune={"model": {"architecture": "plain_arch"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("not compatible with Custom Sigmas", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_sample_solver="lcm",
                )
            self.assertIn("not compatible with Custom Sigmas", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_sample_solver=None,
                )
            self.assertIn("not compatible with Custom Sigmas", str(caught.exception))

    def test_compatible_sigma_scheduler_keeps_a_video(self) -> None:
        """unipc, euler, dpm++ et une chaîne vide de sigmas ne sautent pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for solver in ("unipc", "euler", "dpm++", ""):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": "0.9,0.2", "sample_solver": solver},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            for raw in ("", "   ", "0.9,0"):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": raw, "sample_solver": "lcm"},
                )
                if raw == "0.9,0":
                    with self.assertRaises(Wan2GPError):
                        wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                else:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas": 0},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "plain_arch", "custom_sigmas": True},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_schedulers": None},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_schedulers": 1},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_schedulers": ["lcm", "unipc"]},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
                finetune={"model": {"custom_sigmas_schedulers": False}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "nope", "guidance_phases": None, "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {
                    "custom_sigmas": "0.9,0.2",
                    "sample_solver": "lcm",
                    "sliding_window_discard_last_frames": "large",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "multitalk"},
                {
                    "custom_sigmas": "0.9,0.2",
                    "audio_prompt_type": "",
                    "image_prompt_type": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "animate2"},
                {
                    "custom_sigmas": "0.9,0.2",
                    "image_prompt_type": "",
                    "video_prompt_type": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_sample_solver="",
            )
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_custom_sigmas="",
                queue_sample_solver="lcm",
            )

    def test_incompatible_sigma_scheduler_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le solveur refuse les sigmas."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V"},
                {"custom_sigmas": "0.9,0.2", "sample_solver": "lcm"},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("not compatible with Custom Sigmas", message)
            self.assertIn("code 0", message)

    def test_continuous_custom_sigmas_are_skipped(self) -> None:
        """Un calendrier continu qui ne descend pas saute avant le GPU."""
        descending = "Custom Sigmas requires phase switches in descending order between 0 and 1."
        crosses = (
            "Custom Sigmas phase 1 crosses a phase switch. "
            "This model uses one descending schedule across all phases."
        )
        crosses_phase_2 = (
            "Custom Sigmas phase 2 crosses a phase switch. "
            "This model uses one descending schedule across all phases."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw, phases in (
                ("0.9;", 2),
                ("0.9; ", 2),
                ("0.9;;", 3),
                ("0.9;0.2;0.5", 3),
            ):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": raw, "guidance_phases": phases, "sample_solver": "unipc"},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                text = str(caught.exception)
                self.assertIn(descending, text)
                self.assertIn("code 0", text)
                self.assertNotIn("crosses a phase switch", text)
                self.assertNotIn("not compatible with Custom Sigmas", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.2;0.9", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn(crosses, text)
            self.assertNotIn(descending, text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;0.5,0.1;0.4", "guidance_phases": 3, "sample_solver": "unipc"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(crosses_phase_2, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": "2~", "sample_solver": "euler"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(descending, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_continuous": "false"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(descending, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_continuous": 1},
                {"custom_sigmas": "0.2;0.9", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(crosses, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {
                    "custom_sigmas": "0.9;",
                    "guidance_phases": 2,
                    "switch_threshold": 900,
                    "sample_solver": "dpm++",
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn(crosses, text)
            self.assertNotIn(descending, text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {
                    "custom_sigmas": "0.9;",
                    "guidance_phases": 2,
                    "sample_solver": "unipc",
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn(descending, text)
            self.assertNotIn("NAG Alpha", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "lcm"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("not compatible with Custom Sigmas", text)
            self.assertNotIn(descending, text)
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {
                    "custom_sigmas": "0.9;",
                    "guidance_phases": 2,
                    "custom_settings": {"pace": "nope"},
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("Expected one of the dropdown choices.", text)
            self.assertNotIn(descending, text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;nope", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("enter numbers separated by commas.", text)
            self.assertNotIn(descending, text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9,0.2"},
                finetune={"model": {"architecture": "plain_arch"}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"guidance_phases": 2, "sample_solver": "unipc"},
                finetune={"model": {"architecture": "plain_arch"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_sigmas="0.9;",
                )
            self.assertIn(descending, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "unipc"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_custom_sigmas=None,
                )
            self.assertIn(descending, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_switch_threshold=900,
                )
            self.assertIn(crosses, str(caught.exception))

    def test_continuous_custom_sigmas_keep_a_video(self) -> None:
        """Un calendrier continu valide, ou un drapeau faux, ne saute pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw, phases in (
                ("0.9;0.2", 2),
                ("0.9,0.5;0.4,0.1", 2),
                ("0.9,0.2", 1),
                ("0.9;0.5;0.2", 3),
            ):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": raw, "guidance_phases": phases, "sample_solver": "unipc"},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            for raw in ("", "   "):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"custom_sigmas": raw, "guidance_phases": 2, "sample_solver": "lcm"},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_continuous": False},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "unipc"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            for flag in (0, None, ""):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "custom_sigmas_continuous": flag},
                    {"custom_sigmas": "0.2;0.9", "guidance_phases": 2},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "plain_arch", "custom_sigmas": True},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "lcm"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {
                    "architecture": "plain_arch",
                    "custom_sigmas": True,
                    "custom_sigmas_continuous": True,
                },
                {
                    "custom_sigmas": "0.9;",
                    "guidance_phases": 2,
                    "switch_threshold": 500,
                    "sample_solver": "unipc",
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("descending order between 0 and 1", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2},
                finetune={"model": {"custom_sigmas_continuous": False}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_switch_scale": None},
                {"custom_sigmas": "0.9;", "guidance_phases": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_switch_scale": 0},
                {"custom_sigmas": "0.9;0.2", "guidance_phases": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas_switch_scale": "1000"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "switch_threshold": None},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "switch_threshold": "nope"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {
                    "custom_sigmas": "0.9;",
                    "guidance_phases": 2,
                    "switch_threshold": 100,
                    "sample_solver": "unipc",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas": 0},
                {"custom_sigmas": "0.9;", "guidance_phases": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {
                    "custom_sigmas": "0.9;",
                    "guidance_phases": 2,
                    "sliding_window_discard_last_frames": "large",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "unipc"},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_custom_sigmas="",
            )
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "switch_threshold": 0},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_switch_threshold=100,
            )
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "switch_threshold": 500},
            )
            wan2gp_runner._require_video_model_definition(
                root,
                "t2v_1.3B",
                queue_switch_threshold=None,
            )

    def test_continuous_custom_sigmas_stop_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand les phases continues se croisent."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "sample_solver": "unipc"},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("descending order between 0 and 1", message)
            self.assertIn("code 0", message)

    def test_guidance_switch_is_skipped(self) -> None:
        """Un seuil de guidance illisible saute avant le GPU."""
        integer = "Guidance Switch Threshold must be an integer."
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw, phases in (
                ("large", 2),
                ("large", 3),
                ("1001.5", 2),
                ("large", "2"),
                ("large", 2.9),
                ("large", "2~"),
            ):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"switch_threshold": raw, "guidance_phases": phases, "custom_sigmas": ""},
                )
                with self.assertRaises(Wan2GPError) as caught:
                    wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
                text = str(caught.exception)
                self.assertIn(integer, text)
                self.assertIn("code 0", text)
                self.assertIn("« Guidance Switch Threshold must be an integer. »", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"switch_threshold": 1001, "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be at most 1000.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"switch_threshold": -1, "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be at least 0.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"switch_threshold": True, "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a integer.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"switch_threshold": 1001.5, "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be at most 1000.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "multiple_submodels": "false"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Model / Guidance Switch Threshold must be an integer.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "URLs2": "https://example.invalid/a"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Model / Guidance Switch Threshold must be an integer.", str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "switch_threshold": {"max": 10, "label": "Custom Switch (boost)"},
                },
                {"switch_threshold": 11, "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("Custom Switch must be at most 10.", str(caught.exception))
            self.assertNotIn("boost", str(caught.exception).split("Le message est")[-1])
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": {"label": "("}},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("switch_threshold must be an integer.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": {"min": 2}},
                {"guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be at least 2.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": {"min": 30, "max": 10}},
                {"switch_threshold": 15, "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be at least 30.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": {"type": "number"}},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("must be a number.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "lock_guidance_phases": "false"},
                {"switch_threshold": "large", "guidance_phases": 1, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "visible_phases": 0, "inference_steps2": True},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "visible_phases": 0,
                    "switch_threshold": {"label": "X"},
                },
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn("X must be an integer.", str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "guidance_max_phases": "3"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas": 0},
                {"switch_threshold": "large", "guidance_phases": None},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "NAG": True},
                {
                    "switch_threshold": "large",
                    "guidance_phases": 2,
                    "custom_sigmas": "",
                    "NAG_alpha": 2.01,
                    "NAG_scale": 1,
                    "NAG_tau": 3.5,
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn(integer, text)
            self.assertNotIn("NAG Alpha", text)
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"guidance_phases": 2, "custom_sigmas": "", "switch_threshold": 0},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_switch_threshold="large",
                )
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"guidance_phases": 2, "custom_sigmas": "", "switch_threshold": "large"},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_switch_threshold=None,
                )
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"guidance_phases": 2, "custom_sigmas": "", "guidance_scale": "large"},
                finetune={"model": {"architecture": "plain_arch"}},
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(
                    root,
                    "t2v_1.3B",
                    queue_guidance_scale=5,
                    queue_switch_threshold="large",
                )
            self.assertIn(integer, str(caught.exception))
            _write_model(
                root,
                {
                    "architecture": "t2v_1.3B",
                    "custom_settings": [
                        {"id": "pace", "type": "dropdown", "choices": ["slow", "fast"]},
                    ],
                },
                {
                    "switch_threshold": "large",
                    "guidance_phases": 2,
                    "custom_sigmas": "",
                    "custom_settings": {"pace": "nope"},
                },
            )
            with self.assertRaises(Wan2GPError) as caught:
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            text = str(caught.exception)
            self.assertIn("Expected one of the dropdown choices.", text)
            self.assertNotIn(integer, text)

    def test_guidance_switch_keeps_a_video(self) -> None:
        """Un seuil dans la borne, ou un réglage invisible, ne saute pas."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "wan"
            for raw in (0, 1000, 1000.9, -0.1, "1000.0", "0", "", "   ", None):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B"},
                    {"switch_threshold": raw, "guidance_phases": 2, "custom_sigmas": ""},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"switch_threshold": "large", "guidance_phases": 1, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "visible_phases": 1},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            for max_phases in (0, 1, True, None):
                _write_model(
                    root,
                    {"architecture": "t2v_1.3B", "guidance_max_phases": max_phases},
                    {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
                )
                wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "guidance_max_phases": "nope"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "plain_arch"},
                {"switch_threshold": "large", "guidance_phases": 2},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": "nope"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "temperature_slider": "nope"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {
                    "switch_threshold": "large",
                    "guidance_phases": 2,
                    "custom_sigmas": "",
                    "guidance_scale": "large",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {
                    "switch_threshold": "large",
                    "guidance_phases": 2,
                    "custom_sigmas": "",
                    "guidance2_scale": 0,
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": {"type": "text"}},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "switch_threshold": {"max": "nope"}},
                {"switch_threshold": 1001, "guidance_phases": 2, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "lock_guidance_phases": 0},
                {"switch_threshold": "large", "guidance_phases": 1, "custom_sigmas": ""},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "custom_sigmas": 0},
                {"switch_threshold": "large", "guidance_phases": "large"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;0.2", "guidance_phases": 2, "switch_threshold": "large"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"custom_sigmas": "0.9;", "guidance_phases": 2, "switch_threshold": "nope"},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "sliding_window": True},
                {
                    "switch_threshold": "large",
                    "guidance_phases": 2,
                    "custom_sigmas": "",
                    "sliding_window_discard_last_frames": "large",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {
                    "switch_threshold": 100,
                    "switch_threshold2": 200,
                    "guidance_phases": 3,
                    "custom_sigmas": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {
                    "switch_threshold": 0,
                    "switch_threshold2": "large",
                    "guidance_phases": 1,
                    "custom_sigmas": "",
                },
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")
            _write_model(
                root,
                {"architecture": "t2v_1.3B"},
                {"guidance_phases": 2, "custom_sigmas": "", "switch_threshold": "large"},
                finetune={"model": {"guidance_max_phases": 0}},
            )
            wan2gp_runner._require_video_model_definition(root, "t2v_1.3B")

    def test_guidance_switch_stops_before_the_gpu(self) -> None:
        """Le processus n'est pas lancé quand le seuil n'est pas un entier."""
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "wan"
            root.mkdir()
            (root / "wgp.py").write_text("settings_version = 2.81\n", encoding="utf-8")
            _write_model(
                root,
                {"architecture": "t2v_1.3B", "name": "T2V"},
                {"switch_threshold": "large", "guidance_phases": 2, "custom_sigmas": ""},
            )

            def refuse_gpu(*_args: object, **_kwargs: object) -> _FakeProcess:
                raise AssertionError("GPU")

            with mock.patch(
                "src.wan2gp_runner.subprocess.Popen",
                side_effect=_popen_except_ffmpeg(refuse_gpu),
            ):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 7),
                            settings=replace(config.wan, root=root),
                            output_dir=base / "out",
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            message = str(caught.exception)
            self.assertIn("Guidance Switch Threshold must be an integer.", message)
            self.assertIn("code 0", message)


def _write_model(
    root: Path,
    model: dict[str, object],
    settings: dict[str, object] | None = None,
    finetune: dict[str, object] | None = None,
) -> None:
    """Écrit ``defaults/t2v_1.3B.json`` et, au besoin, le finetune du même nom."""
    defaults = root / "defaults"
    defaults.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"model": model}
    if settings:
        payload.update(settings)
    (defaults / "t2v_1.3B.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    finetunes = root / "finetunes"
    if finetune is None:
        if finetunes.exists():
            for path in finetunes.glob("*.json"):
                path.unlink()
        return
    finetunes.mkdir(exist_ok=True)
    (finetunes / "t2v_1.3B.json").write_text(json.dumps(finetune) + "\n", encoding="utf-8")


def _saved_env(*keys: str) -> dict[str, str | None]:
    return {key: os.environ.get(key) for key in keys}


def _restore_env(saved: dict[str, str | None]) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _real_libstdcxx() -> Path | None:
    """``libstdc++.so.6`` du système, celui que ``CDLL`` peut charger."""
    ldconfig = shutil.which("ldconfig")
    if ldconfig is None:
        return None
    completed = subprocess.run(
        [ldconfig, "-p"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if "=>" not in line:
            continue
        left, _, right = line.partition("=>")
        if left.strip().split(" ", 1)[0] != "libstdc++.so.6":
            continue
        candidate = Path(right.strip())
        if candidate.is_file():
            return candidate
    return None


def _link_ffmpeg_setup(local: Path, real: Path) -> None:
    """``shared/ffmpeg_setup.py`` du clone pointe vers un autre arbre WanGP."""
    (real / "shared").mkdir(parents=True, exist_ok=True)
    (real / "shared" / "ffmpeg_setup.py").write_text("# module résolu\n", encoding="utf-8")
    (local / "shared").mkdir(parents=True)
    (local / "shared" / "__init__.py").write_text("", encoding="utf-8")
    (local / "shared" / "ffmpeg_setup.py").symlink_to(real / "shared" / "ffmpeg_setup.py")
    (local / "wgp.py").write_text("# clone\n", encoding="utf-8")


def _needed_libavcodec(ffmpeg: str) -> str | None:
    return _needed_shared_library(ffmpeg, "libavcodec.so")


def _needed_names(path: str) -> set[str] | None:
    readelf = shutil.which("readelf")
    if readelf is None:
        return None
    completed = subprocess.run(
        [readelf, "-d", path],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    found: set[str] = set()
    for line in completed.stdout.splitlines():
        if "Shared library:" not in line:
            continue
        start = line.find("[")
        end = line.find("]", start + 1)
        if start >= 0 and end > start:
            found.add(line[start + 1 : end])
    return found


def _resolved_library(ffmpeg: str, soname: str) -> str | None:
    ldd = shutil.which("ldd")
    if ldd is None:
        return None
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
    completed = subprocess.run(
        [ldd, ffmpeg],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(soname) or "=>" not in stripped:
            continue
        target = stripped.split("=>", 1)[1].strip().split(" ", 1)[0]
        if target and Path(target).is_file():
            return target
    return None


def _transitive_soname(ffmpeg: str) -> str | None:
    """Soname chargé via libavcodec, absent des NEEDED directs du binaire."""
    direct = _needed_names(ffmpeg)
    codec = _needed_libavcodec(ffmpeg)
    if direct is None or codec is None:
        return None
    resolved = _resolved_library(ffmpeg, codec)
    if resolved is None:
        return None
    nested = _needed_names(resolved)
    if nested is None:
        return None
    extra = sorted(nested - direct)
    for name in extra:
        if name.startswith("libx264.so"):
            return name
    return extra[0] if extra else None


def _hwcap_level(ffmpeg: str) -> str | None:
    """Premier niveau glibc-hwcaps annoncé par l'interpréteur du binaire."""
    readelf = shutil.which("readelf")
    if readelf is None:
        return None
    listed = subprocess.run(
        [readelf, "-l", ffmpeg],
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        return None
    marker = "Requesting program interpreter:"
    interpreter = None
    for line in listed.stdout.splitlines():
        if marker not in line:
            continue
        interpreter = line.split(marker, 1)[1].strip().rstrip("]")
        break
    if not interpreter:
        return None
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
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
        return None
    if completed.returncode != 0:
        return None
    header = "Subdirectories of glibc-hwcaps directories, in priority order:"
    capture = False
    for line in completed.stdout.splitlines():
        if not capture:
            if line.strip() == header:
                capture = True
            continue
        if not line.startswith((" ", "\t")):
            break
        token = line.strip().split(" ", 1)[0]
        if token:
            return token
    return None


def _interpreter_basename(ffmpeg: str) -> str | None:
    readelf = shutil.which("readelf")
    if readelf is None:
        return None
    completed = subprocess.run(
        [readelf, "-l", ffmpeg],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    marker = "Requesting program interpreter:"
    for line in completed.stdout.splitlines():
        if marker not in line:
            continue
        raw = line.split(marker, 1)[1].strip().rstrip("]")
        name = Path(raw).name
        return name or None
    return None


def _needed_libavdevice(ffmpeg: str) -> str | None:
    return _needed_shared_library(ffmpeg, "libavdevice.so")


def _needed_shared_library(ffmpeg: str, stem: str) -> str | None:
    readelf = shutil.which("readelf")
    if readelf is None:
        return None
    completed = subprocess.run(
        [readelf, "-d", ffmpeg],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if "Shared library:" not in line or stem not in line:
            continue
        start = line.find("[")
        end = line.find("]", start + 1)
        if start >= 0 and end > start:
            return line[start + 1 : end]
    return None


def _popen_except_ffmpeg(fake):
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        program = Path(str(command[0])).name
        # readelf, ldd, ldconfig et ld.so --help listent les sonames ou les
        # niveaux glibc-hwcaps. python -c reproduit preload_preferred_libstdcxx.
        # Ils ne lancent pas le GPU. wgp.py, lui, n'a pas d'argument -c.
        if "-c" in command or program in {"ffmpeg", "ffprobe", "readelf", "ldd", "ldconfig"} or program.startswith(
            ("ld-linux", "ld64.so", "ld.so")
        ):
            return real_popen(command, **kwargs)
        return fake(command, **kwargs)

    return popen


class _FakeProcess:
    def __init__(self, log: str = "Video saved\n") -> None:
        self.stdout = io.StringIO(log)
        self.returncode = 0

    def wait(self, timeout: int | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.returncode = -9


def _encode(path: Path, seconds: float, frames: int) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=64x64:d={seconds}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or "ffmpeg n'a pas créé la vidéo de test")


def _encode_mjpeg(path: Path) -> None:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:d=1",
            "-frames:v",
            "8",
            "-c:v",
            "mjpeg",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or "ffmpeg n'a pas créé le MP4 MJPEG")


if __name__ == "__main__":
    unittest.main()
