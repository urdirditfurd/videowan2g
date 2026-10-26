"""Réglages Wan2GP : grille d'images, file ZIP, échec sans MP4, sans GPU."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_pipeline_config
from src.errors import Wan2GPError
from src.pipeline import RunOptions, run
from src.prompt_builder import build_prompt, load_catalog
from src.wan2gp_runner import (
    GenerationRequest,
    ensure_ffmpeg,
    generate,
    parse_resolution,
    require_produced_mp4,
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
            self.assertEqual(payload["image_prompt_type"], "")
            self.assertEqual(payload["multi_prompts_gen_type"], "FG")
            self.assertEqual(payload["settings_version"], 2.81)
            self.assertNotIn("image_start", payload)
            self.assertEqual(payload["video_length"], 81)
            self.assertEqual(payload["resolution"], "624x624")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["force_fps"], "16")
            self.assertEqual(payload["override_profile"], 4)
            self.assertIn("capybara", str(payload["prompt"]))
            self.assertIn("--process", result.command)
            self.assertIn("--output-dir", result.command)
            self.assertIn("--verbose", result.command)
            self.assertEqual(result.command[result.command.index("--verbose") + 1], "2")
            self.assertTrue(str(result.queue_path).endswith(".zip"))
            self.assertIn(str(result.queue_path), result.command)
            self.assertIn("--t2v-1-3B", result.command)
            self.assertIn("--profile", result.command)
            self.assertIn("4", result.command)
            self.assertIn("--attention", result.command)
            self.assertIn("sdpa", result.command)
            self.assertIn("42", result.command)
            self.assertNotIn("--i2v-1-3B", result.command)
            with zipfile.ZipFile(result.queue_path) as archive:
                manifest = json.loads(archive.read("queue.json"))
            self.assertEqual(manifest[0]["params"]["model_type"], "t2v_1.3B")
            self.assertEqual(manifest[0]["params"]["image_mode"], 0)
            self.assertNotIn("image_start", manifest[0]["params"])

    def test_image_start_is_refused(self) -> None:
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
            self.assertIn("Fun_InP_1.3B", str(caught.exception))
            self.assertIn("t2v_1.3B", str(caught.exception))

    def test_i2v_model_is_refused(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        settings = replace(config.wan, model_type="fun_inp_1.3B")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(Wan2GPError) as caught:
                generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 3),
                        settings=settings,
                        output_dir=Path(directory),
                        aspect="9:16",
                        dry_run=True,
                    )
                )
            self.assertIn("fun_inp_1.3B", str(caught.exception))

    def test_extra_settings_cannot_force_image_mode(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        settings = replace(
            config.wan,
            extra_settings={
                "model_type": "fun_inp_1.3B",
                "image_mode": 1,
                "image_start": "/tmp/still.png",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            result = generate(
                GenerationRequest(
                    prompt=build_prompt(theme, 9),
                    settings=settings,
                    output_dir=Path(directory),
                    aspect="9:16",
                    dry_run=True,
                )
            )
            self.assertEqual(result.settings["model_type"], "t2v_1.3B")
            self.assertEqual(result.settings["image_mode"], 0)
            self.assertNotIn("image_start", result.settings)

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
            self.assertTrue(any(str(part).endswith(".zip") for part in manifest["command"]))


class Mp4CheckTest(unittest.TestCase):
    def test_ffmpeg_can_encode_mp4(self) -> None:
        ffmpeg, ffprobe = ensure_ffmpeg()
        self.assertTrue(Path(ffmpeg).is_file())
        self.assertTrue(Path(ffprobe).is_file())

    def test_jpeg_alone_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "frame.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            with self.assertRaises(Wan2GPError) as caught:
                require_produced_mp4(folder, before_mp4={}, before_stills={})
            message = str(caught.exception)
            self.assertIn("Aucun MP4", message)
            self.assertIn("frame.jpg", message)
            self.assertIn("JPEG", message)

    def test_real_mp4_is_accepted(self) -> None:
        ffmpeg, _ffprobe = ensure_ffmpeg()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video = folder / "loop.mp4"
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
                    "color=c=black:s=32x32:r=8:d=0.25",
                    "-frames:v",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-an",
                    str(video),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            found = require_produced_mp4(folder, before_mp4={}, before_stills={})
            self.assertEqual(found, video)

    def test_process_that_only_writes_jpeg_fails(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / "Wan2GP"
            root.mkdir()
            (root / "wgp.py").write_text("# stub\n", encoding="utf-8")
            settings = replace(config.wan, root=root, python=sys.executable)
            real_popen = subprocess.Popen

            def fake_popen(command: list[str] | tuple[str, ...], **kwargs: object) -> object:
                args = list(command)
                if "--output-dir" not in args:
                    return real_popen(command, **kwargs)  # type: ignore[arg-type]
                out = Path(args[args.index("--output-dir") + 1])
                out.mkdir(parents=True, exist_ok=True)
                (out / "still.jpeg").write_bytes(b"\xff\xd8\xff\xd9")
                process = mock.Mock()
                process.stdout = iter(())
                process.wait.return_value = 0
                return process

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=fake_popen):
                with self.assertRaises(Wan2GPError) as caught:
                    generate(
                        GenerationRequest(
                            prompt=build_prompt(theme, 5),
                            settings=settings,
                            output_dir=output,
                            aspect="9:16",
                            dry_run=False,
                        )
                    )
            self.assertIn("Aucun MP4", str(caught.exception))
            self.assertIn("still.jpeg", str(caught.exception))

    def test_process_that_writes_mp4_is_accepted(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["cozy_cafe"]
        ffmpeg, _ffprobe = ensure_ffmpeg()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            root = output / "Wan2GP"
            root.mkdir()
            (root / "wgp.py").write_text("# stub\n", encoding="utf-8")
            settings = replace(config.wan, root=root, python=sys.executable)
            real_popen = subprocess.Popen

            def fake_popen(command: list[str] | tuple[str, ...], **kwargs: object) -> object:
                args = list(command)
                if "--output-dir" not in args:
                    return real_popen(command, **kwargs)  # type: ignore[arg-type]
                out = Path(args[args.index("--output-dir") + 1])
                out.mkdir(parents=True, exist_ok=True)
                video = out / "clip.mp4"
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
                        "color=c=black:s=32x32:r=8:d=0.25",
                        "-frames:v",
                        "2",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-an",
                        str(video),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    raise AssertionError(completed.stderr)
                process = mock.Mock()
                process.stdout = io.StringIO("")
                process.wait.return_value = 0
                return process

            with mock.patch("src.wan2gp_runner.subprocess.Popen", side_effect=fake_popen):
                result = generate(
                    GenerationRequest(
                        prompt=build_prompt(theme, 6),
                        settings=settings,
                        output_dir=output,
                        aspect="9:16",
                        dry_run=False,
                    )
                )
            self.assertIsNotNone(result.raw_video)
            assert result.raw_video is not None
            self.assertEqual(result.raw_video.suffix, ".mp4")
            self.assertTrue(result.raw_video.is_file())
            self.assertGreater(result.raw_video.stat().st_size, 256)


if __name__ == "__main__":
    unittest.main()
