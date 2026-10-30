"""La finition rapproche la dernière image de la première."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.post_processor import PostProcessConfig, post_process, probe_video

FFMPEG = shutil.which("ffmpeg")


@unittest.skipUnless(FFMPEG, "ffmpeg est requis")
class LoopTest(unittest.TestCase):
    def test_crossfade_makes_the_endpoints_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.mp4"
            destination = folder / "loop.mp4"
            _clip(source, duration=2)
            result = post_process(
                source,
                destination,
                PostProcessConfig(crossfade_seconds=0.35, grain=4, warm_grade=True, grain_seed=7),
            )
            self.assertLess(result.endpoint_mae_after, result.endpoint_mae_before / 2)
            self.assertLess(result.endpoint_mae_after, 8)
            self.assertLess(result.output_duration, result.source_duration - 0.2)
            self.assertGreater(result.output_duration, result.source_duration - 0.6)
            info = probe_video(destination)
            self.assertGreater(info.frames, 8)

    def test_optional_audio_is_muxed_from_a_local_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "source.mp4"
            audio = folder / "bed.wav"
            destination = folder / "with-audio.mp4"
            _clip(source, duration=2)
            _tone(audio)
            post_process(
                source,
                destination,
                PostProcessConfig(
                    crossfade_seconds=0.35,
                    grain=0,
                    warm_grade=False,
                    audio_path=audio,
                    audio_volume=0.2,
                ),
            )
            probed = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_streams",
                        "-of",
                        "json",
                        str(destination),
                    ],
                    text=True,
                )
            )
            kinds = {stream["codec_type"] for stream in probed["streams"]}
            self.assertIn("video", kinds)
            self.assertIn("audio", kinds)


def _clip(path: Path, duration: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=320x240:rate=16:duration={duration}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _tone(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=3",
            "-c:a",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )


if __name__ == "__main__":
    unittest.main()
