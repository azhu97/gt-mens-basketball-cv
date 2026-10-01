"""Torch device selection: CUDA, then Apple MPS, then CPU."""


def select_device(preferred: str | None = None) -> str:
    """Return ``preferred`` if given, else the best available device string for ultralytics."""
    if preferred:
        return preferred
    import torch  # imported lazily: torch import is slow and only needed here

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
