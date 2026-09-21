"""Ultralytics yolov8-pose backend — the reference implementation on a Mac.

``ultralytics`` and ``torch`` are imported inside :meth:`UltralyticsBackend.__init__`
so that importing :mod:`linesafe` (and running the test suite) never needs them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..detections import Detection


class UltralyticsBackend:
    """Runs a yolov8-pose checkpoint through the ultralytics predict API.

    Args:
        weights: checkpoint name or path, e.g. ``"yolov8m-pose.pt"``.
        device: ``"mps"`` / ``"cuda"`` / ``"cpu"``; ``None`` picks the best
            available (MPS on Apple silicon, then CUDA, then CPU).
        conf: detection confidence threshold.
        imgsz: inference resolution (the long side; ultralytics letterboxes).
    """

    def __init__(
        self,
        weights: str = "yolov8m-pose.pt",
        device: str | None = None,
        conf: float = 0.25,
        imgsz: int = 640,
    ):
        import torch  # noqa: PLC0415 - lazy: keeps the suite torch-free
        from ultralytics import YOLO  # noqa: PLC0415

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.weights = str(weights)
        self.device = device
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.name = f"ultralytics/{Path(self.weights).stem}"
        self.model = YOLO(self.weights)

    def infer(self, frame_bgr: np.ndarray, idx: int) -> list[Detection]:
        """All people detected in ``frame_bgr``; empty list when there are none."""
        if self.model is None:
            raise RuntimeError("backend already closed")
        results = self.model.predict(
            frame_bgr,
            verbose=False,
            device=self.device,
            conf=self.conf,
            imgsz=self.imgsz,
        )
        if not results:
            return []
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        # MPS tensors cannot be handed to numpy directly: .cpu() first.
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        kpts_xy = keypoints.xy.cpu().numpy()
        if keypoints.conf is None:
            # Faking full confidence here would switch off every downstream
            # conf_min gate and gap fill, hiding a wrong checkpoint.
            raise ValueError(
                f"checkpoint provides no keypoint confidence; cannot build "
                f"Detection (weights={self.weights!r})"
            )
        kpts_conf = keypoints.conf.cpu().numpy()

        return [
            Detection(
                kpts=kpts_xy[i],
                conf=kpts_conf[i],
                bbox=(
                    float(xyxy[i][0]),
                    float(xyxy[i][1]),
                    float(xyxy[i][2]),
                    float(xyxy[i][3]),
                ),
                score=float(scores[i]),
            )
            for i in range(len(xyxy))
        ]

    def close(self) -> None:
        """Drop the model so its device memory can be reclaimed."""
        self.model = None
