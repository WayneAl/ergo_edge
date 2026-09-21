"""Hailo (UGen300 NPU) pose backend — **UNVERIFIED ADAPTER**.

.. warning::

   **UNVERIFIED.** This module was written against the *API shape* of HailoRT
   5.3 pyHailoRT (``VDevice``, ``HEF``, ``ConfigureParams``, ``InferVStreams``)
   from documentation only. There is no Hailo device on the development
   machine, so no line of :meth:`HailoBackend.infer` has ever been executed
   against real hardware or a real ``.hef``. Nothing here should be described
   as working. Expect the vstream names, the output tensor layout, the input
   **channel order** (this adapter feeds RGB, because yolov8 is RGB-trained,
   but the HEF's own compilation settings may already include a BGR→RGB step —
   confirm before trusting any result) and the post-processing split (some
   ``yolov8-pose`` HEFs already run NMS on-chip and emit decoded results
   instead of the raw ``(56, 8400)`` tensor) to need correction on first
   contact with a UGen300; the decode below deliberately raises with the shape
   it received rather than guessing.

   Verified pieces: only that importing this module needs no Hailo software,
   and that constructing :class:`HailoBackend` without pyHailoRT raises
   :class:`HailoUnavailable`.

The COCO-17 output contract is the same as every other backend, so once the
adapter is corrected on hardware nothing downstream changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..detections import Detection
from ..keypoints import N_KPTS

#: yolov8-pose raw head: 4 box + 1 score + 17*3 keypoint channels.
_POSE_CHANNELS = 4 + 1 + N_KPTS * 3
#: IoU threshold of the host-side NMS fallback.
_NMS_IOU = 0.45


class HailoUnavailable(RuntimeError):
    """Raised when the Hailo runtime (pyHailoRT) or device is not available."""


class HailoDecodeError(ValueError):
    """Raised when the device's output does not match the layout decoded here.

    Distinct from :class:`HailoUnavailable`: the runtime and the device are
    fine, the *adapter* does not understand what came back.
    """


def _letterbox(
    frame_bgr: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to ``size`` (w, h) with grey.

    Returns ``(padded, ratio, (pad_x, pad_y))`` so coordinates can be mapped
    back with ``(v - pad) / ratio``.
    """
    import cv2  # noqa: PLC0415 - only needed when a Hailo device is in play

    dst_w, dst_h = size
    src_h, src_w = frame_bgr.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"empty frame of shape {frame_bgr.shape}")
    ratio = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * ratio)), int(round(src_h * ratio))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((dst_h, dst_w, 3), 114, dtype=np.uint8)
    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    top, left = int(round(pad_y)), int(round(pad_x))
    padded[top : top + new_h, left : left + new_w] = resized
    return padded, ratio, (float(left), float(top))


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (n, 4) array of xyxy boxes."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS by score; returns the kept indices, highest score first."""
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thr]
    return keep


class HailoBackend:
    """yolov8-pose on a Hailo NPU. **Unverified** — see the module docstring.

    Args:
        hef_path: compiled ``.hef`` network for the device.
        conf: detection confidence threshold.
    """

    name = "hailo/yolov8m_pose"

    def __init__(self, hef_path: str = "yolov8m_pose.hef", conf: float = 0.25):
        try:
            import hailo_platform  # noqa: PLC0415 - lazy: no Hailo on dev machines
        except ImportError as exc:
            raise HailoUnavailable(
                "pyHailoRT not installed; see README §UGen300"
            ) from exc

        self.hef_path = str(hef_path)
        self.conf = float(conf)
        self._hailo = hailo_platform
        try:
            self._device = hailo_platform.VDevice()
            hef = hailo_platform.HEF(self.hef_path)
            configure_params = hailo_platform.ConfigureParams.create_from_hef(
                hef, interface=hailo_platform.HailoStreamInterface.PCIe
            )
            self._network_group = self._device.configure(hef, configure_params)[0]
            self._network_group_params = self._network_group.create_params()
            self._in_params = hailo_platform.InputVStreamParams.make(
                self._network_group, format_type=hailo_platform.FormatType.UINT8
            )
            self._out_params = hailo_platform.OutputVStreamParams.make(
                self._network_group, format_type=hailo_platform.FormatType.FLOAT32
            )
            in_info = hef.get_input_vstream_infos()[0]
            self._input_name = in_info.name
            in_h, in_w = int(in_info.shape[0]), int(in_info.shape[1])
        except Exception as exc:  # pragma: no cover - needs hardware
            raise HailoUnavailable(
                f"could not open Hailo device with {self.hef_path!r}: {exc}"
            ) from exc
        self.imgsz = (in_w, in_h)

    def infer(self, frame_bgr: np.ndarray, idx: int) -> list[Detection]:
        """Detections for one frame. **Unverified** — see the module docstring."""
        import cv2  # noqa: PLC0415 - only needed when a Hailo device is in play

        padded, ratio, pad = _letterbox(frame_bgr, self.imgsz)
        # yolov8 is RGB-trained; ultralytics converts internally, nothing here
        # does. See the module docstring: the HEF may convert too.
        batch = np.expand_dims(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), axis=0)
        with self._network_group.activate(self._network_group_params):
            with self._hailo.InferVStreams(
                self._network_group, self._in_params, self._out_params
            ) as pipeline:
                outputs = pipeline.infer({self._input_name: batch})
        return self._decode(outputs, ratio, pad)

    def _decode(
        self, outputs: dict[str, Any], ratio: float, pad: tuple[float, float]
    ) -> list[Detection]:
        """Turn the raw yolov8-pose head into ``Detection``s in frame pixels.

        **Unverified.** Accepts the raw ``(1, 56, n)`` / ``(1, n, 56)`` tensor;
        anything else raises :class:`HailoDecodeError` with the shape it
        actually got instead of guessing a layout.
        """
        if len(outputs) != 1:
            raise HailoDecodeError(
                f"expected 1 output vstream, got {sorted(outputs)}"
            )
        raw = np.asarray(next(iter(outputs.values())), dtype=np.float32)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 2:
            raise HailoDecodeError(
                f"unsupported Hailo pose output of shape {raw.shape}; this "
                "adapter decodes only the raw yolov8-pose head"
            )
        if raw.shape[0] == _POSE_CHANNELS:
            preds = raw.T
        elif raw.shape[1] == _POSE_CHANNELS:
            preds = raw
        else:
            raise HailoDecodeError(
                f"unsupported Hailo pose output of shape {raw.shape}; expected "
                f"{_POSE_CHANNELS} channels (4 box + 1 score + {N_KPTS}*3 kpts)"
            )

        scores = preds[:, 4]
        keep_conf = scores >= self.conf
        preds, scores = preds[keep_conf], scores[keep_conf]
        if preds.shape[0] == 0:
            return []

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.stack(
            [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1
        )
        kpts = preds[:, 5:].reshape(-1, N_KPTS, 3)

        pad_x, pad_y = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        kpts[:, :, 0] = (kpts[:, :, 0] - pad_x) / ratio
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_y) / ratio

        return [
            Detection(
                kpts=kpts[i, :, :2],
                conf=kpts[i, :, 2],
                bbox=(
                    float(boxes[i, 0]),
                    float(boxes[i, 1]),
                    float(boxes[i, 2]),
                    float(boxes[i, 3]),
                ),
                score=float(scores[i]),
            )
            for i in _nms(boxes, scores, _NMS_IOU)
        ]

    def close(self) -> None:
        """Release the virtual device. **Unverified** — see the module docstring."""
        device = getattr(self, "_device", None)
        if device is None:
            return
        self._device = None
        release = getattr(device, "release", None)
        if release is not None:  # pragma: no cover - needs hardware
            release()
