"""1€ filter smoothing for keypoint trajectories.

The 1€ filter (Géry Casiez, Nicolas Roussel, Daniel Vogel, CHI 2012) is a
first-order low-pass whose cutoff rises with the estimated speed of the signal:
it removes jitter while the golfer is still and gets out of the way during the
downswing, where lag would move the impact frame.

Cutoffs are in Hz, so the defaults assume pixel coordinates: at address the
hands barely move and the cutoff stays near ``min_cutoff``; through impact the
hand speed is hundreds of px/s and ``beta`` pushes the cutoff far above the
frame rate, leaving the trajectory essentially untouched.

Nothing here invents data: a sample the caller marks as low-confidence is
skipped (the filter is not updated and the previous filtered value is repeated)
rather than being smoothed into the track, and a non-finite input raises
instead of poisoning the filter state.

Variant note: the speed estimate is taken against the *previous filtered*
value (the Tollander/`OneEuroFilter`-Python variant), not against the previous
raw sample as in the Casiez/Roussel/Vogel reference ``lastValue()``. The
filtered reference is quieter on noisy keypoints, at the cost of slightly
understating the speed right after a step.
"""

from __future__ import annotations

import math

import numpy as np

from .keypoints import N_KPTS


class OneEuroFilter:
    """Scalar 1€ filter.

    Args:
        freq: nominal sampling rate in Hz, used when a call passes no
            timestamp (or for the sample right after a :meth:`reset`).
        min_cutoff: cutoff in Hz at zero speed — lower means smoother but laggier.
        beta: speed coefficient in Hz per unit of signal speed — higher means
            less lag during fast motion.
        d_cutoff: cutoff in Hz of the low-pass on the speed estimate itself.
    """

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        if not freq > 0:
            raise ValueError(f"OneEuroFilter.freq must be > 0, got {freq}")
        if not min_cutoff > 0:
            raise ValueError(
                f"OneEuroFilter.min_cutoff must be > 0, got {min_cutoff}"
            )
        if beta < 0:
            raise ValueError(f"OneEuroFilter.beta must be >= 0, got {beta}")
        if not d_cutoff > 0:
            raise ValueError(
                f"OneEuroFilter.d_cutoff must be > 0, got {d_cutoff}"
            )
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(te: float, cutoff: float) -> float:
        """Exponential-smoothing factor for sample interval ``te`` and ``cutoff``."""
        r = 2.0 * math.pi * cutoff * te
        return r / (r + 1.0)

    def reset(self) -> None:
        """Forget all state; the next sample is passed through unchanged."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @property
    def value(self) -> float | None:
        """Last filtered value, or ``None`` before the first sample."""
        return self._x_prev

    def __call__(self, x: float, t: float | None = None) -> float:
        """Filter one sample taken at time ``t`` (seconds); returns the estimate."""
        x = float(x)
        if not math.isfinite(x):
            raise ValueError(f"OneEuroFilter got a non-finite sample: {x}")

        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = None if t is None else float(t)
            return x

        te = 1.0 / self.freq
        if t is not None and self._t_prev is not None:
            dt = float(t) - self._t_prev
            if dt <= 0:
                raise ValueError(
                    f"OneEuroFilter timestamps must increase: {self._t_prev} -> {t}"
                )
            te = dt

        a_d = self._alpha(te, self.d_cutoff)
        # Speed against the previous *filtered* value (Tollander variant), not
        # against the previous raw sample as in the Casiez reference.
        dx = (x - self._x_prev) / te
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(te, cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = None if t is None else float(t)
        return x_hat


def smooth_track_kpts(
    kpts: np.ndarray,
    conf: np.ndarray,
    fps: float,
    min_cutoff: float = 1.5,
    beta: float = 0.05,
    conf_min: float = 0.3,
) -> np.ndarray:
    """Smooth a keypoint track with one 1€ filter per (keypoint, axis).

    Args:
        kpts: ``(n, 17, 2)`` pixel coordinates.
        conf: ``(n, 17)`` per-keypoint confidences.
        fps: frame rate, used both as the filter frequency and to build the
            per-frame timestamps, so skipped frames still advance time.
        min_cutoff, beta: 1€ parameters, see :class:`OneEuroFilter`.
        conf_min: samples with ``conf < conf_min`` are skipped — the filter is
            not updated and the previous filtered value is repeated, so a
            dropped or hallucinated keypoint cannot drag the track with it.
            Skipped samples *before* the first good one pass through unchanged
            (there is nothing to repeat yet).

    Returns:
        ``(n, 17, 2)`` float32 smoothed coordinates. Never introduces NaN or
        inf; a non-finite input raises ``ValueError`` instead.
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    conf = np.asarray(conf, dtype=np.float32)
    if kpts.ndim != 3 or kpts.shape[1:] != (N_KPTS, 2):
        raise ValueError(
            f"smooth_track_kpts expects kpts of shape (n, {N_KPTS}, 2), "
            f"got {kpts.shape}"
        )
    if conf.shape != kpts.shape[:2]:
        raise ValueError(
            f"smooth_track_kpts expects conf of shape {kpts.shape[:2]}, "
            f"got {conf.shape}"
        )
    if not fps > 0:
        raise ValueError(f"smooth_track_kpts needs fps > 0, got {fps}")
    if not np.isfinite(kpts).all():
        raise ValueError("smooth_track_kpts got non-finite keypoints")
    # NaN would slip past `conf < conf_min` as "confident enough" and smooth a
    # broken sample straight into the track.
    if not np.isfinite(conf).all():
        raise ValueError("smooth_track_kpts got non-finite confidences")

    n = kpts.shape[0]
    out = np.empty_like(kpts)
    if n == 0:
        return out

    filters = [
        [
            OneEuroFilter(freq=fps, min_cutoff=min_cutoff, beta=beta)
            for _ in range(2)
        ]
        for _ in range(N_KPTS)
    ]

    for i in range(n):
        t = i / fps
        for k in range(N_KPTS):
            if conf[i, k] < conf_min:
                for axis in range(2):
                    last = filters[k][axis].value
                    out[i, k, axis] = (
                        kpts[i, k, axis] if last is None else last
                    )
                continue
            for axis in range(2):
                out[i, k, axis] = filters[k][axis](float(kpts[i, k, axis]), t)

    return out
