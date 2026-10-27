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
        # niveaux glibc-hwcaps. Ils ne lancent pas le GPU.
        if program in {"ffmpeg", "ffprobe", "readelf", "ldd", "ldconfig"} or program.startswith(
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
