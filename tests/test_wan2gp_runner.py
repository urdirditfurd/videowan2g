"""Réglages Wan2GP : grille d'images, VRAM, seed, sans GPU."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config_loader import load_pipeline_config
from src.errors import Wan2GPError
from src.pipeline import RunOptions, run
from src.prompt_builder import build_prompt, load_catalog
from src.wan2gp_runner import GenerationRequest, generate, parse_resolution, snap_frame_count

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
            self.assertEqual(payload["video_length"], 81)
            self.assertEqual(payload["resolution"], "624x624")
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["force_fps"], "16")
            self.assertEqual(payload["override_profile"], 4)
            self.assertIn("capybara", str(payload["prompt"]))
            self.assertIn("--process", result.command)
            self.assertIn("--t2v-1-3B", result.command)
            self.assertIn("--profile", result.command)
            self.assertIn("4", result.command)
            self.assertIn("--attention", result.command)
            self.assertIn("sdpa", result.command)
            self.assertIn("42", result.command)
            self.assertNotIn("--i2v-1-3B", result.command)

    def test_image_mode_selects_the_light_i2v_model(self) -> None:
        config = load_pipeline_config(ROOT / "config" / "pipeline.yaml")
        theme = load_catalog(ROOT / "config" / "themes.yaml")["night_train"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            image = output / "still.png"
            image.write_bytes(b"png")
            result = generate(
                GenerationRequest(
                    prompt=build_prompt(theme, 7),
                    settings=config.wan,
                    output_dir=output,
                    aspect="9:16",
                    image_path=image,
                    dry_run=True,
                )
            )
            self.assertEqual(result.settings["model_type"], "fun_inp_1.3B")
            self.assertEqual(result.settings["image_prompt_type"], "S")
            self.assertEqual(result.settings["image_start"], str(image.resolve()))
            self.assertIn("--i2v-1-3B", result.command)

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


if __name__ == "__main__":
    unittest.main()
