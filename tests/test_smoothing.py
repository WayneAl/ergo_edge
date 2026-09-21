import math

import numpy as np
import pytest

from linesafe.keypoints import L_WRIST, N_KPTS
from linesafe.smoothing import OneEuroFilter, smooth_track_kpts


def test_one_euro_constant_signal_passthrough():
    """A constant input must come out constant: no warm-up ramp, no overshoot."""
    fps = 30.0
    f = OneEuroFilter(freq=fps)
    out = [f(5.0, i / fps) for i in range(30)]

    assert out[0] == 5.0
    for value in out:
        assert value == pytest.approx(5.0)


def test_one_euro_reduces_noise():
    """Filtered signal is closer to the clean sine than the noisy input is.

    Amplitude is a free parameter (the plan fixes only the 1 Hz sine, the 30 Hz
    sampling rate and the N(0, 0.2) noise).  0.3 puts the signal in the
    noise-dominated regime the filter is meant for.  Note the crossover: with
    the default ``min_cutoff=1.5`` Hz, a 1 Hz sine of amplitude above ~0.45
    loses more to the filter's phase lag than it gains in noise rejection —
    real keypoint trajectories stay on the winning side because their pixel
    speeds push the adaptive cutoff far above 1.5 Hz.
    """
    fps = 30.0
    n = 300
    amplitude = 0.3
    t = np.arange(n) / fps
    clean = amplitude * np.sin(2 * math.pi * 1.0 * t)
    noisy = clean + np.random.default_rng(0).normal(0.0, 0.2, n)

    f = OneEuroFilter(freq=fps)
    filtered = np.array([f(float(x), float(ts)) for x, ts in zip(noisy, t)])

    def rms(a: np.ndarray) -> float:
        return float(np.sqrt(np.mean((a - clean) ** 2)))

    assert rms(filtered) < rms(noisy)


def test_one_euro_beta_reduces_lag():
    """``beta`` must actually open the cutoff during fast motion.

    A fixed-cutoff first-order low-pass lags a ramp by a constant offset; the
    1€ filter's speed term shrinks that offset. Without this the other tests
    still pass with ``beta`` ignored — and a laggy filter would move the
    detected impact frame.
    """
    fps = 30.0
    n = 10
    true = np.linspace(0.0, 100.0, n)
    t = np.arange(n) / fps

    def lag(beta: float) -> float:
        f = OneEuroFilter(freq=fps, beta=beta)
        out = np.array([f(float(x), float(ts)) for x, ts in zip(true, t)])
        return float(np.mean(np.abs(out[-3:] - true[-3:])))

    lag_fixed = lag(0.0)  # measured ~31 px
    lag_adaptive = lag(0.05)  # measured ~2.9 px
    assert lag_adaptive * 5.0 < lag_fixed


def test_smooth_track_skips_low_conf():
    """A zero-confidence spike must not move the output at all."""
    fps = 30.0
    n = 20
    spike_at = 10

    kpts = np.zeros((n, N_KPTS, 2), dtype=np.float32)
    kpts[:, L_WRIST, 0] = np.arange(n, dtype=np.float32) * 4.0  # steady drift
    kpts[:, L_WRIST, 1] = 100.0
    conf = np.ones((n, N_KPTS), dtype=np.float32)

    kpts[spike_at, L_WRIST, 0] = 9999.0
    kpts[spike_at, L_WRIST, 1] = -9999.0
    conf[spike_at, L_WRIST] = 0.0

    out = smooth_track_kpts(kpts, conf, fps)

    assert out.shape == (n, N_KPTS, 2)
    assert np.isfinite(out).all()
    assert out[spike_at, L_WRIST, 0] == out[spike_at - 1, L_WRIST, 0]
    assert out[spike_at, L_WRIST, 1] == out[spike_at - 1, L_WRIST, 1]
    # and the spike leaves no trace afterwards
    assert out[spike_at + 1, L_WRIST, 0] < kpts[spike_at + 1, L_WRIST, 0] + 1.0
