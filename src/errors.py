"""Erreurs du pipeline, distinctes des échecs Python génériques."""


class PipelineError(RuntimeError):
    """Échec attendu : thème, fichier, FFmpeg ou génération."""


class ThemeError(PipelineError):
    """Catalogue de thèmes illisible ou incomplet."""


class Wan2GPError(PipelineError):
    """Wan2GP absent, invalide, ou la génération a échoué."""
