"""Caption sinks — consume CaptionEvents, render frames to a destination."""
from .base import CaptionSink

__all__ = ["CaptionSink"]


def make_sink(kind: str, **kwargs) -> CaptionSink:
    """Factory: select a sink by string kind. Lazy-imports each implementation."""
    if kind == "png":
        from .png import PngSink

        return PngSink(**kwargs)
    if kind == "canvas":
        from .canvas_ws import CanvasWsSink

        return CanvasWsSink(**kwargs)
    if kind == "pyvirtualcam":
        from .pyvirtualcam import PyVirtualCamSink

        return PyVirtualCamSink(**kwargs)
    raise ValueError(f"unknown sink kind: {kind}")
