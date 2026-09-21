"""From a video file to raw per-frame detections.

:func:`extract_keypoints` decodes a video and asks a
:class:`~linesafe.backends.base.PoseBackend` for the detections of every
frame, producing exactly what a keypoints file holds.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .backends.base import PoseBackend
from .detections import Detection, RawDetections


def extract_keypoints(
    video_path: Path,
    backend: PoseBackend,
    source: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> RawDetections:
    """Run ``backend`` over every frame of ``video_path``.

    Args:
        video_path: the clip to decode.
        backend: anything satisfying the :class:`PoseBackend` protocol; it is
            *not* closed here — the caller owns it.
        source: name recorded in the keypoints file; defaults to the video stem.
        progress: called as ``progress(done, total)`` after each frame.

    Returns:
        A :class:`RawDetections` holding every detection of every frame — the
        exact payload of a version-1 keypoints file.

    Raises:
        FileNotFoundError: the video does not exist.
        ValueError: the video cannot be opened, reports ``fps <= 0`` or
            degenerate dimensions, or decodes to zero frames. A clip whose
            frame rate is unknown cannot be timed, so it is refused rather than
            analyzed at a guessed 30 fps.
    """
    # Imported here so that importing this module (and the test suite) never
    # pays for the OpenCV import.
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not fps > 0:
            raise ValueError(
                f"{video_path} reports fps {fps}; cannot time a session without it"
            )
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            raise ValueError(
                f"{video_path} reports a {width}x{height} frame size"
            )
        total = max(0, int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

        frames: list[list[Detection]] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(backend.infer(frame_bgr, len(frames)))
            if progress is not None:
                progress(len(frames), max(total, len(frames)))
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")

    return RawDetections(
        source=source if source is not None else video_path.stem,
        fps=fps,
        width=width,
        height=height,
        backend=backend.name,
        frames=frames,
    )
