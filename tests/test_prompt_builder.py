"""Le prompt porte toujours la mascotte, le mouvement lent et une variation stable."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.errors import ThemeError
from src.prompt_builder import build_prompt, load_catalog

THEMES = Path(__file__).resolve().parents[1] / "config" / "themes.yaml"


class PromptBuilderTest(unittest.TestCase):
    def test_every_theme_keeps_the_mascot_and_the_loop(self) -> None:
        catalog = load_catalog(THEMES)
        self.assertGreaterEqual(len(catalog), 8)
        for theme in catalog.values():
            self.assertEqual(theme.motion.speed, "slow")
            for seed in range(len(theme.variations)):
                built = build_prompt(theme, seed)
                text = built.prompt.lower()
                self.assertIn("capybara", text)
                self.assertIn("slow gentle natural motion", text)
                self.assertIn("looping animation", text)
                self.assertIn("locked tripod", text)
                self.assertIn(theme.background_motion.lower(), text)
                self.assertEqual(built.variation, theme.variations[seed])
                self.assertIn("camera shake", built.negative_prompt)

    def test_same_seed_is_stable_and_another_seed_changes_the_detail(self) -> None:
        theme = load_catalog(THEMES)["rainy_window"]
        first = build_prompt(theme, 3)
        again = build_prompt(theme, 3)
        other = build_prompt(theme, 4)
        self.assertEqual(first.prompt, again.prompt)
        self.assertNotEqual(first.variation, other.variation)
        self.assertIn("rain", first.prompt.lower())

    def test_fast_motion_is_rejected(self) -> None:
        raw = """
themes:
  rush:
    title: Trop vite
    base_prompt: A room
    background_motion: clouds race
    mascot_action: runs
    lighting: noon
    palette: red
    motion:
      speed: fast
      ambience: sprint
    variations: [un livre, une tasse, une plante]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "themes.yaml"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaises(ThemeError):
                load_catalog(path)


if __name__ == "__main__":
    unittest.main()
