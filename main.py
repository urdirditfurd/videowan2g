"""Point d'entrée : un thème, un seed, une boucle capybara."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from src.config_loader import Aspect, Backend, PipelineConfig, load_pipeline_config
from src.errors import PipelineError
from src.pipeline import RunOptions, list_themes, run


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    if args.list:
        for theme_id, title in list_themes():
            print(f"{theme_id:16}  {title}")
        return 0
    if not args.theme:
        print("Indiquer --theme, ou --list pour le catalogue.", file=sys.stderr)
        return 2
    if args.image is not None:
        print(
            "L'option --image est refusée. Fun_InP_1.3B (I2V) peut écrire un JPEG et quitter "
            "sans MP4. La génération reste en text-to-video t2v_1.3B.",
            file=sys.stderr,
        )
        return 2
    try:
        config = _apply_cli(load_pipeline_config(), args.backend)
        result = run(
            RunOptions(
                theme_id=args.theme,
                seed=args.seed,
                dry_run=args.dry_run,
                aspect=_aspect(args.aspect),
                frames=args.frames,
                resolution=args.resolution,
                image_path=None,
                audio_path=args.audio,
            ),
            config=config,
        )
    except PipelineError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    print(f"seed={result.seed}")
    if result.video_path is not None:
        print(result.video_path)
    else:
        print(result.manifest_path)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Génère une boucle lofi Ghibli avec une mascotte capybara, via Wan2GP."
    )
    parser.add_argument("--theme", help="Identifiant du thème, par exemple rainy_window.")
    parser.add_argument("--list", action="store_true", help="Affiche le catalogue puis s'arrête.")
    parser.add_argument("--seed", type=int, help="Seed positif. Absent : un seed aléatoire est choisi.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Écrit le prompt et les réglages Wan2GP, sans lancer le GPU.",
    )
    parser.add_argument("--aspect", choices=["9:16", "1:1"], help="Cadrage Shorts ou carré.")
    parser.add_argument("--frames", type=int, help="Nombre d'images demandé, aligné ensuite sur la grille Wan.")
    parser.add_argument("--resolution", help="Résolution WIDTHxHEIGHT, multiple de 16. Prioritaire sur --aspect.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Refusé. L'I2V Fun_InP_1.3B peut produire un JPEG au lieu d'un MP4.",
    )
    parser.add_argument("--audio", type=Path, help="Piste locale libre de droits à mixer sous la boucle.")
    parser.add_argument(
        "--backend",
        choices=["subprocess", "api"],
        help="Les deux lancent wgp.py --process. La valeur api ne garde plus le modèle en mémoire.",
    )
    return parser


def _apply_cli(config: PipelineConfig, backend: str | None) -> PipelineConfig:
    if backend is None:
        return config
    chosen = _backend(backend)
    return replace(config, wan=replace(config.wan, backend=chosen))


def _backend(value: str) -> Backend:
    match value:
        case "subprocess":
            return "subprocess"
        case "api":
            return "api"
        case other:
            raise PipelineError(f"Backend inconnu: {other}.")


def _aspect(value: str | None) -> Aspect | None:
    match value:
        case None:
            return None
        case "9:16":
            return "9:16"
        case "1:1":
            return "1:1"
        case other:
            raise PipelineError(f"Cadrage inconnu: {other}.")


if __name__ == "__main__":
    sys.exit(main())
