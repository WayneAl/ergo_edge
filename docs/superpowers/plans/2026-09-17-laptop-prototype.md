# LineSafe laptop prototype — implementation plan

**Goal:** a side-view webcam session scored live with REBA and RULA on a laptop, with an overlay window,
SQLite events, a replay path for tests, and a LAN dashboard page.
**Architecture:** pose backends and the 1€ filter are copied from golf_coach; everything after the backend
is new and pure (tracker → geometry → REBA/RULA → activity → events), with I/O only in the CLI, overlay,
store and web modules. `linesafe replay` runs the identical pipeline from a keypoints file, so every test
is camera-free and torch-free.
**Spec:** `docs/superpowers/specs/2026-09-17-laptop-prototype-design.md` — executors read both.
**Stack:** Python 3.11, uv, hatchling, numpy ≥ 2, opencv-python ≥ 4.10, typer ≥ 0.12; extras
`pose` = ultralytics ≥ 8.3, `web` = fastapi ≥ 0.115 + uvicorn ≥ 0.30; dev group pytest ≥ 8, httpx ≥ 0.27.

## Global constraints (every task inherits these)
- Repo `/Users/waynekuo/Documents/GitHub/ergo_edge`; work only in the worktree path the orchestrator gives you.
- Package `src/linesafe/`; `requires-python = ">=3.11,<3.14"`; `.python-version` = `3.11`.
- No dependencies beyond the Stack line. torch/ultralytics/fastapi are imported lazily; `uv run pytest`
  must pass without the `pose` extra installed code paths being executed, without a camera, without network.
- golf_coach (`/Users/waynekuo/Documents/GitHub/golf_coach`, commit 078f217) is a read-only copy source.
  Never modify it; never `import golfcoach`.
- Units: angles in degrees, time in seconds (never frame counts in any params object), pixels in image
  coordinates with x right and y **down**.
- Fail loud: out-of-range inputs raise `ValueError` naming the field; a missing measurement is a
  `Missing(reason)` value, never 0, NaN or a silent default. The only defaults for non-visual inputs live in
  `StationConfig`.
- `tests/fixtures/worksheet_tables.py` is verified ground truth; never edit it; production code never imports it.
- English identifiers, comments and UI text.
- Git: commit on the task branch with `git add <explicit paths>`; message `<type>: <what>` ending with the
  line `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. No pushing (orchestrator
  pushes). No repo-wide formatter.
- Held back for the orchestrator/Wayne: opening a camera, recording video, anything needing a person in front
  of a camera, pushing, network calls.

## File map
| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml`, `.python-version`, `.gitignore` (append) | project, deps, scripts, pytest config | 1 |
| `src/linesafe/__init__.py` | version string | 1 |
| `src/linesafe/keypoints.py` | COCO-17 indices, `N_KPTS`, `NAMES`, `SKELETON` (copied) | 1 |
| `src/linesafe/detections.py` | `Detection`, `RawDetections`, keypoints file v1 save/load (copied from golf `types.py`) | 1 |
| `src/linesafe/smoothing.py` | `OneEuroFilter` (copied whole file) | 1 |
| `src/linesafe/backends/{__init__,base,replay,ultralytics_backend,hailo_backend}.py` | pose backends (copied) | 1 |
| `src/linesafe/capture.py` | `extract_keypoints(video, backend) -> RawDetections` (copied `run_backend`) | 1 |
| `src/linesafe/tables.py` | REBA/RULA lookup tables + range-checked lookups | 2 |
| `src/linesafe/pose.py` | `PoseFrame` | 3 |
| `src/linesafe/geometry.py` | view, facing, joint angles | 3 |
| `src/linesafe/config.py` | `StationConfig`, `load_station` | 4 |
| `src/linesafe/reba.py` | REBA segment scores, `RebaScore`, `score_reba` | 4 |
| `stations/example.toml` | sample station config | 4 |
| `src/linesafe/track.py` | streaming `Tracker` | 5 |
| `src/linesafe/activity.py` | activity flags over time | 6 |
| `src/linesafe/events.py` | High-risk event detector | 6 |
| `src/linesafe/store.py` | SQLite `EventStore` | 6 |
| `src/linesafe/rula.py` | RULA segment scores, `RulaScore`, `score_rula` | 7 |
| `src/linesafe/pipeline.py` | `Pipeline.step` composition | 8 |
| `src/linesafe/overlay.py` | OpenCV drawing | 8 |
| `src/linesafe/cli.py` | typer app: `run`, `replay`, `extract` (+ `web` in 9) | 8, 9 |
| `tests/synth.py` | synthetic side-view poses and sessions from angles | 8 |
| `src/linesafe/web.py` | FastAPI app + inline dashboard page | 9 |
| `tests/test_*.py` | one test module per source module | each |

---

### Task 1: Project scaffold and copied pose front half (glue)

**Files:** Create `pyproject.toml`, `.python-version`, `src/linesafe/__init__.py`, `src/linesafe/keypoints.py`,
`src/linesafe/detections.py`, `src/linesafe/smoothing.py`, `src/linesafe/backends/__init__.py`,
`src/linesafe/backends/base.py`, `src/linesafe/backends/replay.py`, `src/linesafe/backends/ultralytics_backend.py`,
`src/linesafe/backends/hailo_backend.py`, `src/linesafe/capture.py`, `tests/test_smoothing.py`,
`tests/test_detections.py`, `tests/test_backends.py` · Modify `.gitignore` (append).
(`tests/__init__.py`, `tests/fixtures/__init__.py`, `tests/fixtures/worksheet_tables.py` already exist.)

**Interfaces — Produces:**
- `linesafe.keypoints`: `NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST,
  R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE` (= 0..16), `N_KPTS = 17`, `NAMES: list[str]`,
  `SKELETON: list[tuple[int, int]]`.
- `linesafe.detections`: `KEYPOINTS_FORMAT_VERSION = 1`, `_check_kpts_conf(kpts, conf, owner) -> (np.ndarray, np.ndarray)`,
  `Detection(kpts, conf, bbox, score)` with `.area`, `RawDetections(source, fps, width, height, backend, frames)` with `.n`,
  `save_keypoints(path, raw)`, `load_keypoints(path) -> RawDetections`.
- `linesafe.smoothing.OneEuroFilter(freq, min_cutoff=1.5, beta=0.05, d_cutoff=1.0)`; call `f(x, t)`; `.reset()`; `.value`.
- `linesafe.backends`: `PoseBackend` protocol (`name`, `infer(frame_bgr, idx) -> list[Detection]`, `close()`),
  `make_backend(name, **kwargs)`; `ReplayBackend(keypoints_path)`; `UltralyticsBackend`; `HailoBackend`, `HailoUnavailable`.
- `linesafe.capture.extract_keypoints(video_path, backend, source=None, progress=None) -> RawDetections`.

**Steps:**
- [ ] 1. `pyproject.toml` modelled on golf's: `name = "linesafe"`, `version = "0.1.0"`,
  `description = "Continuous REBA/RULA ergonomic risk scoring from pose estimation"`, dependencies
  `numpy>=2`, `opencv-python>=4.10`, `typer>=0.12`; `[project.optional-dependencies]` `pose = ["ultralytics>=8.3"]`,
  `web = ["fastapi>=0.115", "uvicorn>=0.30"]`; `[dependency-groups] dev = ["pytest>=8", "httpx>=0.27"]`;
  `[project.scripts] linesafe = "linesafe.cli:app"`; hatchling with `packages = ["src/linesafe"]`;
  `[tool.pytest.ini_options] testpaths = ["tests"]`. `.python-version` = `3.11`.
  Append to `.gitignore`: `pilot/`, `*.db`, `*.mp4`, `yolov8*-pose.pt`.
  `src/linesafe/__init__.py`: `__version__ = "0.1.0"`. → verify: `uv sync --all-extras` exits 0.
- [ ] 2. Copy golf `src/golfcoach/keypoints.py` verbatim to `src/linesafe/keypoints.py` (docstring may drop golf wording).
- [ ] 3. Create `src/linesafe/detections.py` by copying from golf `src/golfcoach/types.py` exactly these
  symbols and their imports: `KEYPOINTS_FORMAT_VERSION`, `_XY_DP`, `_CONF_DP`, `_as_int`, `_check_kpts_conf`,
  `Detection`, `RawDetections`, `_write_json`, `save_keypoints`, `load_keypoints`. Nothing golf-specific.
- [ ] 4. Copy golf `smoothing.py` whole; copy golf `backends/*.py` whole; change every `..types` import to
  `..detections` and `golfcoach` names in docstrings to `linesafe`. Copy golf `tracking.run_backend` into
  `capture.py` renamed `extract_keypoints`, error text "cannot time a swing" → "cannot time a session".
- [ ] 5. Tests: copy golf `tests/test_smoothing.py` (adjust imports; drop tests of anything not copied).
  From golf `tests/test_backends.py` and `tests/test_types.py` copy only the tests exercising copied
  symbols (Detection validation, keypoints round-trip, ReplayBackend, make_backend unknown name raising,
  Hailo/Ultralytics import-free construction checks if present). Any test that used a golf fixture file must
  instead write a 3-frame keypoints file into `tmp_path` via `save_keypoints`. Do not copy golf fixtures.
  → verify: `uv run pytest -q` all green; paste the count.
- [ ] 6. → verify: `uv run python -c "import linesafe.backends, linesafe.capture; print('ok')"` prints ok, and
  `uv run python -c "import sys, linesafe.backends; print('torch' in sys.modules)"` prints `False`.
- [ ] 7. Commit: `chore: scaffold linesafe package with pose front half copied from golf_coach`.

**Rules for this task:** smallest diff — copy, don't redesign; keep golf's fail-loud checks verbatim.

---

### Task 2: REBA and RULA lookup tables (logic, TDD)

**Files:** Create `src/linesafe/tables.py`, `tests/test_tables.py`.

**Interfaces — Produces** (all raise `ValueError` naming the argument when out of range, ints only):
- `reba_table_a(neck: int, trunk: int, legs: int) -> int` — neck 1–3, trunk 1–5, legs 1–4.
- `reba_table_b(upper_arm: int, lower_arm: int, wrist: int) -> int` — upper_arm 1–6, lower_arm 1–2, wrist 1–3.
- `reba_table_c(score_a: int, score_b: int) -> int` — both 1–12.
- `rula_table_a(upper_arm: int, lower_arm: int, wrist: int, wrist_twist: int) -> int` — 1–6, 1–3, 1–4, 1–2.
- `rula_table_b(neck: int, trunk: int, legs: int) -> int` — 1–6, 1–6, 1–2.
- `rula_table_c(wrist_arm: int, neck_trunk_leg: int) -> int` — both ≥ 1; row index clamps to 8 ("8+"),
  column clamps to 7 ("7+"); < 1 raises.

**Steps:**
- [ ] 1. Write `tests/test_tables.py` (below) → verify: `uv run pytest tests/test_tables.py -q` FAILS with
  `ModuleNotFoundError: linesafe.tables`.
- [ ] 2. Implement `tables.py`: store the six tables as module-level tuples-of-tuples (immutable), values copied
  from `tests/fixtures/worksheet_tables.py` (layout of that file is documented in its comments: REBA_A is
  `[neck][trunk][legs]`; RULA_A rows are `[upper_arm][lower_arm] -> [w1t1, w1t2, w2t1, w2t2, w3t1, w3t2, w4t1, w4t2]`;
  RULA_B rows are `[neck] -> [t1l1, t1l2, …, t6l1, t6l2]`). Do not import the fixture. → verify: same command PASSES.
- [ ] 3. Full suite green. Commit: `feat: REBA and RULA lookup tables checked against worksheet ground truth`.

```python
import itertools
import pytest
from linesafe import tables as T
from tests.fixtures import worksheet_tables as W

def test_reba_table_a_every_cell():
    for n, t, g in itertools.product(range(1, 4), range(1, 6), range(1, 5)):
        assert T.reba_table_a(n, t, g) == W.REBA_A[n][t - 1][g - 1], (n, t, g)

def test_reba_table_b_every_cell():
    for u, l, w in itertools.product(range(1, 7), range(1, 3), range(1, 4)):
        assert T.reba_table_b(u, l, w) == W.REBA_B[u][l - 1][w - 1], (u, l, w)

def test_reba_table_c_every_cell():
    for a, b in itertools.product(range(1, 13), range(1, 13)):
        assert T.reba_table_c(a, b) == W.REBA_C[a - 1][b - 1], (a, b)

def test_rula_table_a_every_cell():
    for u, l, w, tw in itertools.product(range(1, 7), range(1, 4), range(1, 5), range(1, 3)):
        assert T.rula_table_a(u, l, w, tw) == W.RULA_A[u][l - 1][(w - 1) * 2 + (tw - 1)], (u, l, w, tw)

def test_rula_table_b_every_cell():
    for n, t, g in itertools.product(range(1, 7), range(1, 7), range(1, 3)):
        assert T.rula_table_b(n, t, g) == W.RULA_B[n][(t - 1) * 2 + (g - 1)], (n, t, g)

def test_rula_table_c_every_cell_and_clamps():
    for a, b in itertools.product(range(1, 9), range(1, 8)):
        assert T.rula_table_c(a, b) == W.RULA_C[a - 1][b - 1], (a, b)
    assert T.rula_table_c(11, 9) == W.RULA_C[7][6]

@pytest.mark.parametrize("call, name", [
    (lambda: T.reba_table_a(0, 1, 1), "neck"), (lambda: T.reba_table_a(1, 6, 1), "trunk"),
    (lambda: T.reba_table_a(1, 1, 5), "legs"), (lambda: T.reba_table_b(7, 1, 1), "upper_arm"),
    (lambda: T.reba_table_b(1, 3, 1), "lower_arm"), (lambda: T.reba_table_b(1, 1, 4), "wrist"),
    (lambda: T.reba_table_c(13, 1), "score_a"), (lambda: T.reba_table_c(1, 0), "score_b"),
    (lambda: T.rula_table_a(1, 1, 5, 1), "wrist"), (lambda: T.rula_table_a(1, 1, 1, 3), "wrist_twist"),
    (lambda: T.rula_table_b(7, 1, 1), "neck"), (lambda: T.rula_table_c(0, 1), "wrist_arm"),
])
def test_out_of_range_raises_naming_argument(call, name):
    with pytest.raises(ValueError, match=name):
        call()

def test_non_int_raises():
    with pytest.raises(ValueError):
        T.reba_table_a(1.0, 1, 1)
```

---

### Task 3: PoseFrame and joint-angle geometry (logic, TDD)

**Files:** Create `src/linesafe/pose.py`, `src/linesafe/geometry.py`, `tests/test_geometry.py`.

**Interfaces — Consumes:** `linesafe.keypoints` indices; `linesafe.detections._check_kpts_conf`.
**Produces:**
```python
# pose.py
@dataclass(frozen=True)
class PoseFrame:
    t: float
    kpts: np.ndarray   # (17, 2) float32, validated with _check_kpts_conf
    conf: np.ndarray   # (17,) float32
    bbox: tuple[float, float, float, float]

# geometry.py
class View(str, Enum): SIDE = "side"; FRONT = "front"
@dataclass(frozen=True)
class Measured: deg: float; conf: float        # conf = min conf of the keypoints used
@dataclass(frozen=True)
class Missing: reason: str
Angle = Measured | Missing
LEFT, RIGHT = 0, 1
@dataclass(frozen=True)
class GeometryParams:
    conf_min: float = 0.3
    front_ratio_min: float = 0.55   # shoulder width / torso length at or above this = FRONT
    legs_level_frac: float = 0.10   # ankle height difference <= this * torso length = both feet down
    twist_frac: float = 0.20        # |shoulder width - hip width| / torso length above this = twisted
    neck_offset_deg: float = 0.0    # subtracted from neck flexion; calibrated in the pilot check
@dataclass(frozen=True)
class Angles:
    t: float
    view: View
    facing: int | None                      # +1 facing +x (right), -1 facing left, None unknown
    trunk_flex: Angle                       # + flexion, - extension
    trunk_twisted: bool | Missing
    neck_flex: Angle                        # relative to trunk; + flexion, - extension
    upper_arm: tuple[Angle, Angle]          # (left, right); relative to trunk; + flexion, - extension
    lower_arm: tuple[Angle, Angle]          # elbow flexion, 0 = straight
    knee: tuple[Angle, Angle]               # knee flexion, 0 = straight
    legs_bilateral: bool | Missing
def classify_view(pose: PoseFrame, params: GeometryParams = GeometryParams()) -> View
def facing(pose: PoseFrame, params: GeometryParams = GeometryParams()) -> int | None
def compute_angles(pose: PoseFrame, params: GeometryParams = GeometryParams(), view_ok: bool = True) -> Angles
```

**Definitions (implement exactly):** a keypoint is *confident* when `conf >= conf_min`.
- `shoulder_mid` / `hip_mid`: mean of the confident shoulders / hips (one is enough). None confident → every
  trunk-relative angle is `Missing("no confident shoulders")` / `Missing("no confident hips")`.
- `torso = shoulder_mid - hip_mid`, `L = |torso|`; `L < 1` → trunk-relative angles `Missing("degenerate torso")`.
- `u = torso / L` (trunk up). `facing`: if nose and ≥ 1 ear confident and `|nose.x - ear_mean.x| >= 2` →
  `sign(nose.x - ear_mean.x)`; elif nose confident and `|nose.x - shoulder_mid.x| >= 2` →
  `sign(nose.x - shoulder_mid.x)`; else None. Signed angles need facing: None → `Missing("facing unknown")`.
- `s = facing`; trunk forward unit `f = (-u.y * s, u.x * s)`.
- `trunk_flex = degrees(atan2(s * torso.x, -torso.y))`.
- `neck_flex = degrees(atan2(dot(h, f), dot(h, u))) - neck_offset_deg`, `h = ear_mean - shoulder_mid`
  (confident ears only; none → `Missing("no confident ear")`).
- `upper_arm[side] = degrees(atan2(dot(v, f), dot(v, -u)))`, `v = elbow - shoulder` of that side
  (shoulder or elbow not confident → `Missing("l_elbow not confident")` using `NAMES`); then if the result is
  `<= -90` add 360, so the range is (-90, 270] — an overhead reach past 180° stays a large flexion instead of
  wrapping to a negative extension (amended after Task 3 review; shoulder extension never reaches -90°).
- `lower_arm[side] = 180 - interior angle at elbow between (shoulder - elbow) and (wrist - elbow)`.
- `knee[side] = 180 - interior angle at knee between (hip - knee) and (ankle - knee)`.
- `legs_bilateral`: both ankles confident → `|ankle_l.y - ankle_r.y| <= legs_level_frac * L`; else `Missing("ankle not confident")`.
- `trunk_twisted`: both shoulders and both hips confident → `abs(|lsh.x-rsh.x| - |lhip.x-rhip.x|) / L > twist_frac`;
  else `Missing("shoulders or hips not confident")`. Computed regardless of `view_ok`.
- `classify_view`: both shoulders confident and `L >= 1` → FRONT if `|lsh.x - rsh.x| / L >= front_ratio_min`
  else SIDE; otherwise SIDE.
- `view_ok=False` → `trunk_flex`, `neck_flex`, `upper_arm`, `lower_arm`, `knee` are all
  `Missing("front view on a side-view station")`; `trunk_twisted` and `legs_bilateral` computed as usual.
- Interior angle uses `acos(clip(dot / (|a||b|), -1, 1))`; a segment shorter than 1 px (same threshold as the
  torso) → `Missing("degenerate <segment>")`. `PoseFrame` rejects non-finite kpts/conf with `ValueError`;
  `GeometryParams.conf_min` must satisfy `0 < conf_min <= 1`.

**Named failure modes owned here (spec):** wrong camera view → flexion angles Missing (test 11); a missing
keypoint is `Missing` with a reason, never 0 (test 10).

**Steps:**
- [ ] 1. Write `tests/test_geometry.py` (below) → verify FAIL (`ModuleNotFoundError`).
- [ ] 2. Implement `pose.py` and `geometry.py` → verify `uv run pytest tests/test_geometry.py -q` PASS; full suite green.
- [ ] 3. Commit: `feat: side-view joint angles with view and facing detection`.

```python
import math
import numpy as np
import pytest
from linesafe import keypoints as K
from linesafe.pose import PoseFrame
from linesafe.geometry import (GeometryParams, Measured, Missing, View, LEFT, RIGHT,
                               classify_view, compute_angles, facing)

def pose(points, conf=0.9, low=()):
    k = np.zeros((17, 2), np.float32); c = np.full(17, conf, np.float32)
    for i, (x, y) in points.items():
        k[i] = (x, y)
    for i in low:
        c[i] = 0.0
    return PoseFrame(t=0.0, kpts=k, conf=c, bbox=(0.0, 0.0, 400.0, 600.0))

def upright(over=()):
    p = {K.NOSE: (115, 178), K.L_EAR: (100, 180), K.R_EAR: (100, 180),
         K.L_SHOULDER: (102, 200), K.R_SHOULDER: (98, 200), K.L_HIP: (102, 300), K.R_HIP: (98, 300),
         K.L_ELBOW: (102, 250), K.R_ELBOW: (98, 250), K.L_WRIST: (102, 300), K.R_WRIST: (98, 300),
         K.L_KNEE: (102, 400), K.R_KNEE: (98, 400), K.L_ANKLE: (102, 500), K.R_ANKLE: (98, 500)}
    p.update(dict(over)); return p

def deg(a):
    assert isinstance(a, Measured), a
    return a.deg

def test_upright_is_all_zero():
    a = compute_angles(pose(upright()))
    assert a.view is View.SIDE and a.facing == 1
    assert deg(a.trunk_flex) == pytest.approx(0, abs=0.01)
    assert deg(a.neck_flex) == pytest.approx(0, abs=0.01)
    for side in (LEFT, RIGHT):
        assert deg(a.upper_arm[side]) == pytest.approx(0, abs=0.01)
        assert deg(a.lower_arm[side]) == pytest.approx(0, abs=0.01)
        assert deg(a.knee[side]) == pytest.approx(0, abs=0.01)
    assert a.legs_bilateral is True and a.trunk_twisted is False

def lean(deg_, facing_=1):
    dx, dy = 100 * math.sin(math.radians(deg_)) * facing_, -100 * math.cos(math.radians(deg_))
    sh = (100 + dx, 300 + dy)
    ear = (sh[0], sh[1] - 20); nose = (ear[0] + 15 * facing_, ear[1] - 2)
    return upright({K.L_SHOULDER: sh, K.R_SHOULDER: sh, K.L_HIP: (100, 300), K.R_HIP: (100, 300),
                      K.L_EAR: ear, K.R_EAR: ear, K.NOSE: nose})

def test_trunk_flexion_45_facing_right():
    assert deg(compute_angles(pose(lean(45))).trunk_flex) == pytest.approx(45, abs=0.01)

def test_trunk_flexion_45_facing_left_is_still_positive():
    a = compute_angles(pose(lean(45, facing_=-1)))
    assert a.facing == -1 and deg(a.trunk_flex) == pytest.approx(45, abs=0.01)

def test_trunk_extension_is_negative():
    assert deg(compute_angles(pose(lean(-20))).trunk_flex) == pytest.approx(-20, abs=0.01)

def test_upper_arm_forward_90_and_back_30():
    p = upright({K.L_ELBOW: (152, 200), K.R_ELBOW: (98 - 25, 200 + 43.30127)})
    a = compute_angles(pose(p))
    assert deg(a.upper_arm[LEFT]) == pytest.approx(90, abs=0.01)
    assert deg(a.upper_arm[RIGHT]) == pytest.approx(-30, abs=0.01)

def test_upper_arm_is_relative_to_trunk():
    p = lean(45)
    sh = p[K.L_SHOULDER]
    p[K.L_ELBOW] = (sh[0], sh[1] + 50)            # hanging straight down in the image
    assert deg(compute_angles(pose(p)).upper_arm[LEFT]) == pytest.approx(45, abs=0.01)

def test_elbow_90():
    p = upright({K.L_SHOULDER: (100, 200), K.L_ELBOW: (100, 250), K.L_WRIST: (150, 250)})
    assert deg(compute_angles(pose(p)).lower_arm[LEFT]) == pytest.approx(90, abs=0.01)

def test_knee_60():
    p = upright({K.L_HIP: (100, 300), K.L_KNEE: (100, 400), K.L_ANKLE: (186.60254, 450)})
    assert deg(compute_angles(pose(p)).knee[LEFT]) == pytest.approx(60, abs=0.01)

def test_neck_flexion_30():
    ear = (100 + 20 * math.sin(math.radians(30)), 200 - 20 * math.cos(math.radians(30)))
    p = upright({K.L_SHOULDER: (100, 200), K.R_SHOULDER: (100, 200), K.L_HIP: (100, 300),
                   K.R_HIP: (100, 300), K.L_EAR: ear, K.R_EAR: ear, K.NOSE: (ear[0] + 15, ear[1])})
    assert deg(compute_angles(pose(p)).neck_flex) == pytest.approx(30, abs=0.01)

def test_missing_elbow_is_missing_with_reason():
    a = compute_angles(pose(upright(), low=(K.L_ELBOW,)))
    assert isinstance(a.upper_arm[LEFT], Missing) and "l_elbow" in a.upper_arm[LEFT].reason
    assert isinstance(a.upper_arm[RIGHT], Measured)

def test_front_view_detected_and_blocks_flexion():
    p = upright({K.L_SHOULDER: (60, 200), K.R_SHOULDER: (140, 200)})
    assert classify_view(pose(p)) is View.FRONT
    a = compute_angles(pose(p), view_ok=False)
    assert isinstance(a.trunk_flex, Missing) and "front" in a.trunk_flex.reason
    assert isinstance(a.knee[LEFT], Missing)

def test_facing_unknown_blocks_signed_angles():
    a = compute_angles(pose(upright(), low=(K.NOSE,)))
    assert a.facing is None
    assert isinstance(a.trunk_flex, Missing) and "facing" in a.trunk_flex.reason
    assert isinstance(a.lower_arm[LEFT], Measured)       # unsigned angles do not need facing

def test_one_foot_raised_is_not_bilateral():
    a = compute_angles(pose(upright({K.R_ANKLE: (98, 440)})))
    assert a.legs_bilateral is False

def test_twist_proxy():
    p = upright({K.L_SHOULDER: (60, 200), K.R_SHOULDER: (140, 200), K.L_HIP: (95, 300), K.R_HIP: (105, 300)})
    assert compute_angles(pose(p)).trunk_twisted is True

def test_facing_needs_nose_offset():
    p = upright({K.NOSE: (101, 178)})
    assert facing(pose(p)) is None
```

---

### Task 4: StationConfig and REBA scoring (logic, TDD)

**Files:** Create `src/linesafe/config.py`, `src/linesafe/reba.py`, `stations/example.toml`, `tests/test_config.py`, `tests/test_reba.py`.

**Interfaces — Consumes:** `tables.reba_table_a/b/c`; `geometry.Angles, Measured, Missing, LEFT, RIGHT`.
**Produces:**
```python
# config.py
@dataclass(frozen=True)
class StationConfig:
    station_id: str
    reba_load: int = 0            # 0: < 5 kg, 1: 5-10 kg, 2: > 10 kg
    reba_shock: bool = False      # +1 shock or rapid build-up of force
    reba_coupling: int = 0        # 0 good, 1 fair, 2 poor, 3 unacceptable
    reba_wrist: int = 1           # final REBA wrist score 1-3 (incl. deviation/twist)
    arm_supported: bool = False   # -1 upper arm in REBA and RULA
    rula_wrist: int = 1           # 1-4 (incl. bent-from-midline)
    rula_wrist_twist: int = 1     # 1-2
    rula_force: int = 0           # 0-3
    roi: tuple[int, int, int, int] | None = None   # x1, y1, x2, y2 pixels; None = whole frame
def load_station(path: Path) -> StationConfig
```
`__post_init__` range-checks every field (`ValueError` naming it); `station_id` non-empty; roi has x1<x2, y1<y2.
TOML shape (`load_station` reads a `[station]` table; keys equal field names, `id` maps to `station_id`; unknown
key → `ValueError` naming it):
```toml
[station]
id = "S1"
reba_load = 1
reba_coupling = 1
reba_wrist = 2
rula_wrist = 2
rula_force = 1
# roi = [0, 0, 1920, 1080]
```
```python
# reba.py
class Band(str, Enum):
    NEGLIGIBLE = "negligible"; LOW = "low"; MEDIUM = "medium"; HIGH = "high"; VERY_HIGH = "very high"
def reba_band(total: int) -> Band                  # 1 | 2-3 | 4-7 | 8-10 | 11-15, else ValueError
UPRIGHT_TOL_DEG = 5.0
def trunk_score(flex_deg: float, twisted: bool) -> int
def neck_score(flex_deg: float) -> int
def legs_score(bilateral: bool, knee_flex_deg: float | None) -> int
def upper_arm_score(flex_deg: float, supported: bool) -> int
def lower_arm_score(flex_deg: float) -> int
ASSUMED_SIDE_VIEW: tuple[str, ...] = ("trunk side-bend", "neck twist", "neck side-bend",
                                      "shoulder raised", "upper arm abducted")
@dataclass(frozen=True)
class RebaScore:
    t: float
    total: int                 # 1-15
    band: Band
    score_a: int               # table A + load
    score_b: int               # table B + coupling
    table_c: int
    activity: int              # 0-3
    parts: dict[str, int]      # keys: trunk, neck, legs, upper_arm, lower_arm, wrist, load, coupling
    side: str                  # "left" | "right" | "none" — arm used for group B
    drivers: tuple[str, ...]
    partial: bool
    missing: tuple[str, ...]
    assumed: tuple[str, ...]   # always ASSUMED_SIDE_VIEW in v1
def score_reba(angles: Angles, cfg: StationConfig, activity: int = 0) -> RebaScore | None
```
**Scoring rules (Hignett & McAtamney 2000; boundaries fixed here):**
- trunk: `|f| <= 5` → 1; `5 < f <= 20` → 2; `-20 <= f < -5` → 2; `20 < f <= 60` → 3; `f < -20` → 3; `f > 60` → 4; `+1` if twisted.
- neck: `-5 <= n <= 20` → 1; `n > 20` → 2; `n < -5` → 2.
- legs: bilateral → 1 else 2; knee `30 <= k <= 60` → +1; `k > 60` → +2; `k is None` → +0.
- upper arm: `-20 <= u <= 20` → 1; `20 < u <= 45` or `u < -20` → 2; `45 < u <= 90` → 3; `u > 90` → 4; supported → −1, floor 1.
- lower arm: `60 <= l <= 100` → 1 else 2.
- wrist = `cfg.reba_wrist`; load = `cfg.reba_load + int(cfg.reba_shock)`; coupling = `cfg.reba_coupling`.
- `score_a = reba_table_a(neck, trunk, legs) + load`; `score_b = reba_table_b(upper, lower, wrist) + coupling`;
  `table_c = reba_table_c(min(score_a, 12), min(score_b, 12))`; `total = table_c + activity`; `activity` must be 0–3
  (else `ValueError`); assert `1 <= total <= 15`.
- Missing handling: `trunk_flex` Missing → return `None`. `trunk_twisted` Missing → treated as not twisted, add
  `"trunk twist"` to `missing` (does not set partial). `neck_flex` Missing → neck 1, `"neck"` in missing, partial.
  `legs_bilateral` Missing → legs base 1, `"legs"` in missing, partial. Knee: max of Measured knees; none → +0,
  `"knee"` in missing (not partial). Arms: for each side whose upper arm is Measured compute upper score and lower
  score (lower Missing → 2 and `"lower arm <side>"` in missing, partial); pick the side with the larger
  `reba_table_b` value, tie → larger upper-arm flexion, tie → left. No Measured upper arm → upper 1, lower 2,
  side "none", `"upper arm"` in missing, partial.
- Drivers, in this order, only when they apply: trunk ≥ 3 `f"trunk flexion {f:.0f}°"` (negative: `f"trunk extension {-f:.0f}°"`);
  twisted `"trunk twist"`; neck 2 `f"neck flexion {n:.0f}°"`; upper arm ≥ 3 `f"upper arm {side} {u:.0f}°"`;
  legs ≥ 2 `f"knee flexion {k:.0f}°"` if knee added else `"one-leg stance"`; load > 0 `"load (station)"`;
  coupling > 0 `"coupling (station)"`; activity > 0 `f"activity +{activity}"`.

**Named failure mode owned here:** a needed angle Missing → score computed from available parts, `partial=True`,
listed in `missing`; no person/trunk → no score (never total 1).

**Steps:**
- [ ] 1. Write `tests/test_config.py` and `tests/test_reba.py` (below) → verify FAIL.
- [ ] 2. Implement → verify PASS; full suite green. Create `stations/example.toml` with the TOML above and a
  test that `load_station("stations/example.toml")` parses it.
- [ ] 3. Commit: `feat: station config and REBA scoring with explicit missing-angle handling`.

```python
# tests/test_reba.py
import pytest
from linesafe.config import StationConfig
from linesafe.geometry import Angles, Measured, Missing, View
from linesafe import reba as R

def angles(trunk=0.0, twisted=False, neck=0.0, ua=(0.0, 0.0), la=(0.0, 0.0), knee=(0.0, 0.0), bilateral=True):
    m = lambda v: v if isinstance(v, Missing) else Measured(float(v), 0.9)
    return Angles(t=1.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=twisted, neck_flex=m(neck),
                  upper_arm=(m(ua[0]), m(ua[1])), lower_arm=(m(la[0]), m(la[1])), knee=(m(knee[0]), m(knee[1])),
                  legs_bilateral=bilateral)

CFG = StationConfig(station_id="S1")

@pytest.mark.parametrize("f,exp", [(0, 1), (5, 1), (5.01, 2), (20, 2), (20.01, 3), (60, 3), (60.01, 4),
                                   (-5, 1), (-5.01, 2), (-20, 2), (-20.01, 3)])
def test_trunk_bands(f, exp):
    assert R.trunk_score(f, False) == exp and R.trunk_score(f, True) == exp + 1

@pytest.mark.parametrize("n,exp", [(0, 1), (20, 1), (20.01, 2), (-5, 1), (-5.01, 2)])
def test_neck_bands(n, exp):
    assert R.neck_score(n) == exp

@pytest.mark.parametrize("u,sup,exp", [(-20, False, 1), (20, False, 1), (20.01, False, 2), (45, False, 2),
                                       (45.01, False, 3), (90, False, 3), (90.01, False, 4), (-20.01, False, 2),
                                       (0, True, 1), (60, True, 2)])
def test_upper_arm_bands(u, sup, exp):
    assert R.upper_arm_score(u, sup) == exp

@pytest.mark.parametrize("l,exp", [(59.99, 2), (60, 1), (100, 1), (100.01, 2), (0, 2)])
def test_lower_arm_bands(l, exp):
    assert R.lower_arm_score(l) == exp

@pytest.mark.parametrize("bi,k,exp", [(True, None, 1), (False, None, 2), (True, 29.99, 1), (True, 30, 2),
                                      (True, 60, 2), (True, 60.01, 3), (False, 70, 4)])
def test_legs(bi, k, exp):
    assert R.legs_score(bi, k) == exp

@pytest.mark.parametrize("total,band", [(1, R.Band.NEGLIGIBLE), (2, R.Band.LOW), (3, R.Band.LOW), (4, R.Band.MEDIUM),
                                        (7, R.Band.MEDIUM), (8, R.Band.HIGH), (10, R.Band.HIGH),
                                        (11, R.Band.VERY_HIGH), (15, R.Band.VERY_HIGH)])
def test_bands(total, band):
    assert R.reba_band(total) is band

@pytest.mark.parametrize("bad", [0, 16])
def test_band_out_of_range(bad):
    with pytest.raises(ValueError):
        R.reba_band(bad)

def test_upright_neutral_is_negligible():
    s = R.score_reba(angles(), CFG)
    # trunk1 neck1 legs1 -> A=1; upper1 lower2 (straight arm) wrist1 -> B=1; C[1][1]=1
    assert (s.score_a, s.score_b, s.table_c, s.total, s.band) == (1, 1, 1, 1, R.Band.NEGLIGIBLE)
    assert s.partial is False and s.drivers == ()

def test_lifting_box_is_high():
    cfg = StationConfig(station_id="S1", reba_load=2, reba_coupling=1, reba_wrist=2)
    s = R.score_reba(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90), knee=(40, 35)), cfg)
    # trunk4 neck2 legs1+1=2 -> A table 6 + load 2 = 8; right arm: upper4 lower1 wrist2 -> 5 + coupling 1 = 6
    assert s.parts == {"trunk": 4, "neck": 2, "legs": 2, "upper_arm": 4, "lower_arm": 1, "wrist": 2, "load": 2, "coupling": 1}
    assert (s.score_a, s.score_b, s.table_c, s.total, s.band, s.side) == (8, 6, 10, 10, R.Band.HIGH, "right")
    assert s.drivers == ("trunk flexion 64°", "neck flexion 30°", "upper arm right 95°", "knee flexion 40°",
                         "load (station)", "coupling (station)")

def test_activity_adds_to_total():
    cfg = StationConfig(station_id="S1", reba_load=2, reba_coupling=1, reba_wrist=2)
    s = R.score_reba(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90), knee=(40, 35)), cfg, activity=3)
    assert s.total == 13 and s.band is R.Band.VERY_HIGH and s.drivers[-1] == "activity +3"

def test_missing_trunk_gives_no_score():
    assert R.score_reba(angles(trunk=Missing("facing unknown")), CFG) is None

def test_missing_neck_is_partial():
    s = R.score_reba(angles(neck=Missing("no confident ear")), CFG)
    assert s.parts["neck"] == 1 and s.partial is True and "neck" in s.missing

def test_missing_twist_is_listed_not_partial():
    s = R.score_reba(angles(twisted=Missing("hips not confident")), CFG)
    assert "trunk twist" in s.missing and s.partial is False

def test_no_upper_arm_is_partial_side_none():
    s = R.score_reba(angles(ua=(Missing("x"), Missing("y"))), CFG)
    assert s.side == "none" and s.parts["upper_arm"] == 1 and s.partial is True

def test_activity_out_of_range():
    with pytest.raises(ValueError):
        R.score_reba(angles(), CFG, activity=4)

def test_assumed_is_listed():
    assert R.score_reba(angles(), CFG).assumed == R.ASSUMED_SIDE_VIEW
```
```python
# tests/test_config.py
import pytest
from linesafe.config import StationConfig, load_station

@pytest.mark.parametrize("field,value", [("reba_load", 3), ("reba_coupling", 4), ("reba_wrist", 0), ("reba_wrist", 4),
                                         ("rula_wrist", 5), ("rula_wrist_twist", 3), ("rula_force", 4)])
def test_range_checks_name_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        StationConfig(station_id="S1", **{field: value})

def test_empty_station_id():
    with pytest.raises(ValueError, match="station_id"):
        StationConfig(station_id="")

def test_bad_roi():
    with pytest.raises(ValueError, match="roi"):
        StationConfig(station_id="S1", roi=(10, 10, 5, 20))

def test_load_station(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[station]\nid = "S7"\nreba_load = 1\nroi = [0, 0, 640, 480]\n')
    c = load_station(p)
    assert c.station_id == "S7" and c.reba_load == 1 and c.roi == (0, 0, 640, 480)

def test_unknown_key(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[station]\nid = "S7"\nlaod = 1\n')
    with pytest.raises(ValueError, match="laod"):
        load_station(p)

def test_example_file_parses():
    assert load_station("stations/example.toml").station_id == "S1"
```

---

### Task 5: Streaming tracker (logic, TDD)

**Files:** Create `src/linesafe/track.py`, `tests/test_track.py`.

**Interfaces — Consumes:** `Detection`; `PoseFrame`; `OneEuroFilter`; `keypoints.L_SHOULDER/R_SHOULDER`.
**Produces:**
```python
@dataclass(frozen=True)
class TrackerParams:
    conf_min: float = 0.3
    iou_min: float = 0.3
    lost_s: float = 0.5        # no matching detection for longer than this -> drop the lock and re-seed
    hold_s: float = 0.1        # a keypoint below conf_min keeps its last smoothed value this long
    min_cutoff: float = 1.5
    beta: float = 0.05
    freq_hint: float = 30.0    # OneEuroFilter(freq=...) before timestamps exist
class Tracker:
    def __init__(self, params: TrackerParams = TrackerParams(), roi: tuple[int, int, int, int] | None = None) -> None
    @property
    def locked(self) -> bool
    def update(self, t: float, dets: list[Detection]) -> PoseFrame | None
```
**Behaviour (implement exactly):**
- Not locked: candidates = detections whose both shoulders have `conf >= conf_min` and whose bbox centre lies in
  `roi` (if set). Seed on the largest `area`, ties by `score`, then list order. None → return None.
- Locked: match = detection with the highest IoU against the last bbox, if IoU ≥ `iou_min` (roi not applied to
  matching). No match: if `t - last_matched_t > lost_s` → unlock, reset all filters, and try to seed from `dets` in
  this same call; else return None (lock kept).
- For the matched/seeded detection, per keypoint: `conf >= conf_min` → x and y each through their own
  `OneEuroFilter(freq_hint, min_cutoff, beta)` with timestamp `t`; store last value, last conf, last good time.
  Below `conf_min`: if a previous value exists and `t - last_good_t <= hold_s` → output last value with last good
  conf; else output last value (or the raw point if none) with conf 0.0.
- Output `PoseFrame(t, kpts, conf, bbox=det.bbox)`. Timestamps must strictly increase per `update` call that
  processes a detection; otherwise the filter's `ValueError` propagates.

**Named failure mode owned here:** tracker loses the person > 0.5 s → re-seed on the largest confident-shoulder
bbox inside the ROI (tests 4, 5); a bystander never steals the lock while the target is matched (test 3).

**Steps:**
- [ ] 1. Tests below → FAIL. - [ ] 2. Implement → PASS, suite green. - [ ] 3. Commit `feat: streaming single-person tracker with 1€ smoothing`.

```python
import numpy as np
import pytest
from linesafe.detections import Detection
from linesafe.track import Tracker, TrackerParams
from linesafe import keypoints as K

def det(x1, y1, x2, y2, score=0.9, shoulder_conf=0.9, conf=0.9, dx=0.0):
    k = np.zeros((17, 2), np.float32)
    k[:, 0] = (x1 + x2) / 2 + dx; k[:, 1] = np.linspace(y1, y2, 17)
    c = np.full(17, conf, np.float32); c[K.L_SHOULDER] = c[K.R_SHOULDER] = shoulder_conf
    return Detection(kpts=k, conf=c, bbox=(x1, y1, x2, y2), score=score)

def test_seeds_largest_with_confident_shoulders():
    tr = Tracker()
    big_bad = det(0, 0, 400, 800, shoulder_conf=0.1); small_ok = det(500, 0, 700, 400)
    p = tr.update(0.0, [big_bad, small_ok])
    assert p.bbox == small_ok.bbox and tr.locked

def test_roi_excludes_outside_centre():
    tr = Tracker(roi=(0, 0, 450, 1000))
    p = tr.update(0.0, [det(500, 0, 900, 800), det(0, 0, 200, 400)])
    assert p.bbox == (0.0, 0.0, 200.0, 400.0)

def test_follows_target_not_bigger_bystander():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    p = tr.update(1 / 30, [det(900, 0, 1500, 900), det(5, 0, 205, 400)])
    assert p.bbox == (5.0, 0.0, 205.0, 400.0)

def test_short_miss_keeps_lock():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    assert tr.update(0.2, []) is None and tr.locked
    p = tr.update(0.3, [det(900, 0, 1500, 900), det(2, 0, 202, 400)])
    assert p.bbox == (2.0, 0.0, 202.0, 400.0)

def test_long_miss_reseeds_on_largest():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    assert tr.update(0.3, []) is None
    p = tr.update(0.9, [det(900, 0, 1500, 900), det(1600, 0, 1700, 200)])   # nothing overlaps the old bbox
    assert p.bbox == (900.0, 0.0, 1500.0, 900.0)

def test_first_frame_passes_through_and_constant_stays():
    tr = Tracker(); d = det(0, 0, 200, 400)
    p0 = tr.update(0.0, [d]); p1 = tr.update(1 / 30, [d])
    np.testing.assert_allclose(p0.kpts, d.kpts); np.testing.assert_allclose(p1.kpts, d.kpts, atol=1e-4)

def test_low_conf_keypoint_held_then_dropped():
    tr = Tracker(); d = det(0, 0, 200, 400)
    tr.update(0.0, [d])
    low = det(0, 0, 200, 400); low.conf[K.L_ELBOW] = 0.0   # Detection stores float32 arrays
    p = tr.update(0.05, [low])
    assert p.conf[K.L_ELBOW] == pytest.approx(0.9, abs=1e-6)
    np.testing.assert_allclose(p.kpts[K.L_ELBOW], d.kpts[K.L_ELBOW], atol=1e-4)
    p = tr.update(0.2, [low])
    assert p.conf[K.L_ELBOW] == 0.0

def test_non_increasing_time_raises():
    tr = Tracker(); d = det(0, 0, 200, 400)
    tr.update(1.0, [d])
    with pytest.raises(ValueError):
        tr.update(1.0, [d])
```
(If `Detection` freezes its arrays so `low.conf[...] = 0` fails, build `low` with a `conf` argument instead;
report which you did.)

---

### Task 6: Activity flags, event detector, SQLite store (logic, TDD)

**Files:** Create `src/linesafe/activity.py`, `src/linesafe/events.py`, `src/linesafe/store.py`,
`tests/test_activity.py`, `tests/test_events.py`, `tests/test_store.py`.

**Interfaces — Consumes:** `geometry.Angles, Measured`; `reba.RebaScore, Band`.
**Produces:**
```python
# activity.py
@dataclass(frozen=True)
class ActivityParams:
    window_s: float = 60.0          # REBA static / repetition window
    static_band_deg: float = 10.0   # held = max - min within this band
    rep_amp_deg: float = 15.0       # a reversal needs this much travel from the running extremum
    rep_per_min: int = 4            # REBA: more than this many actions per minute -> +1
    rapid_window_s: float = 1.0
    rapid_deg: float = 45.0         # trunk range within rapid_window_s at or above this -> rapid
    rapid_hold_s: float = 2.0       # the rapid flag stays on this long after the last trigger
    rula_static_s: float = 600.0    # RULA muscle use: static longer than 10 min (worksheet)
    max_gap_s: float = 1.0          # a gap longer than this breaks "held"
@dataclass(frozen=True)
class ActivityFlags:
    static: bool
    repeated: bool
    rapid: bool
    rula_muscle_use: bool
    @property
    def reba_points(self) -> int    # static + repeated + rapid
class ActivityTracker:
    def __init__(self, params: ActivityParams = ActivityParams()) -> None
    def update(self, t: float, angles: Angles | None) -> ActivityFlags
```
Signals: `trunk = angles.trunk_flex.deg` when Measured; `arm = max(deg of Measured upper arms)` when any.
`angles is None` or a Missing signal adds no sample (a gap). Everything is O(1) amortised per frame — at 30 FPS a
10-minute history must never be rescanned. Per signal keep:
- *run* state `run_start, run_min, run_max, last_t`. On a sample `(t, v)`: if `last_t is None` or
  `t - last_t > max_gap_s` or `max(run_max, v) - min(run_min, v) > static_band_deg` → `run_start = t`,
  `run_min = run_max = v`; else widen `run_min/run_max`. `last_t = t`.
  *held(signal, span)* = `last_t is not None and last_t >= t - max_gap_s and run_start <= t - span`.
  `static = held(trunk, window_s) or held(arm, window_s)`.
- *zig-zag* state `dir` (0 unknown, +1, −1), `ext`, `ref`, and a deque of reversal times. On a sample `v`
  (a gap longer than `max_gap_s` first resets `dir = 0, ref = v`): `dir == 0`: `v - ref >= rep_amp_deg` →
  `dir = +1, ext = v`; `ref - v >= rep_amp_deg` → `dir = -1, ext = v`. `dir == +1`: `v > ext` → `ext = v`; elif
  `ext - v >= rep_amp_deg` → append `t` to reversals, `dir = -1, ext = v`. `dir == -1` mirrored. Drop reversal
  times `< t - window_s`. `actions = len(reversals) // 2`.
  `repeated = max(actions(trunk), actions(arm)) > rep_per_min`.
- *rapid*: a deque of trunk samples within `[t - rapid_window_s, t]`; `max - min >= rapid_deg` → `last_rapid_t = t`;
  `rapid = last_rapid_t is not None and t - last_rapid_t <= rapid_hold_s`.
- `rula_muscle_use = held(trunk, rula_static_s) or held(arm, rula_static_s) or max(actions) >= rep_per_min`.

```python
# events.py
@dataclass(frozen=True)
class EventParams:
    enter_total: int = 8     # REBA High
    enter_s: float = 3.0
    exit_s: float = 3.0      # below High this long closes the event
@dataclass(frozen=True)
class Event:
    station: str
    t_start: float
    t_end: float             # last sample at or above enter_total
    peak: int
    band: Band               # reba_band(peak)
    drivers: tuple[str, ...] # drivers of the first sample that reached peak
    duration_s: float        # t_end - t_start
class EventDetector:
    def __init__(self, station: str, params: EventParams = EventParams()) -> None
    @property
    def active(self) -> bool
    def update(self, t: float, score: RebaScore | None) -> Event | None   # returns the event when it closes
    def flush(self) -> Event | None                                        # closes an active event; pending -> None
```
States: IDLE → (above) PENDING(start=t) → (above and `t - start >= enter_s`) ACTIVE. PENDING and not above → IDLE.
ACTIVE: above → `last_above = t`, clear `below_since`, track peak; not above → `below_since = below_since or t`;
`t - below_since >= exit_s` → close and return the Event, go IDLE. `above = score is not None and score.total >= enter_total`.
Peak/drivers tracking starts at PENDING start.

```python
# store.py
class EventStore:
    def __init__(self, path: Path | str) -> None     # sqlite3, check_same_thread=False, PRAGMA journal_mode=WAL; creates tables
    def add_event(self, e: Event) -> int
    def set_status(self, station: str, t: float, reba_total: int | None, band: str | None,
                   rula_total: int | None, drivers: tuple[str, ...], partial: bool, fps: float) -> None   # upsert by station
    def events(self, limit: int = 100, since: float | None = None) -> list[dict]   # newest first; keys = column names, drivers as list
    def status(self) -> list[dict]
    def close(self) -> None
```
Schema: `events(id INTEGER PRIMARY KEY, station TEXT NOT NULL, t_start REAL NOT NULL, t_end REAL NOT NULL,
peak INTEGER NOT NULL, band TEXT NOT NULL, drivers TEXT NOT NULL, duration_s REAL NOT NULL)`;
`status(station TEXT PRIMARY KEY, t REAL NOT NULL, reba_total INTEGER, band TEXT, rula_total INTEGER,
drivers TEXT NOT NULL, partial INTEGER NOT NULL, fps REAL NOT NULL)`. `drivers` stored as JSON.

**Steps:**
- [ ] 1. Tests below → FAIL. - [ ] 2. Implement the three modules → PASS, suite green.
- [ ] 3. Commit: `feat: REBA activity flags, High-risk event detector and SQLite event store`.

```python
# tests/test_activity.py
import math
from linesafe.activity import ActivityTracker, ActivityParams
from linesafe.geometry import Angles, Measured, Missing, View

FPS = 30
def ang(trunk, arm=None):
    m = lambda v: Missing("x") if v is None else Measured(float(v), 0.9)
    return Angles(t=0.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=False, neck_flex=m(0),
                  upper_arm=(m(arm), Missing("x")), lower_arm=(m(90), m(90)), knee=(m(0), m(0)), legs_bilateral=True)

def run(fn, seconds, tracker=None, t0=0.0):
    tr = tracker or ActivityTracker(); out = None
    for i in range(int(seconds * FPS) + 1):
        t = t0 + i / FPS
        out = tr.update(t, fn(t))
    return tr, out

def test_static_after_one_minute_not_before():
    _, f = run(lambda t: ang(30), 59)
    assert f.static is False
    _, f = run(lambda t: ang(30), 61)
    assert f.static is True and f.reba_points >= 1

def test_gap_breaks_static():
    _, f = run(lambda t: None if 20 <= t < 22 else ang(30), 61)
    assert f.static is False

def test_repeated_fast_cycles():
    _, f = run(lambda t: ang(15 + 15 * math.sin(2 * math.pi * t / 10)), 70)   # 6 cycles/min, 30° p-p
    assert f.repeated is True

def test_slow_cycles_not_repeated():
    _, f = run(lambda t: ang(15 + 15 * math.sin(2 * math.pi * t / 30)), 70)   # 2 cycles/min
    assert f.repeated is False

def test_rapid_trunk_change_holds_two_seconds():
    tr = ActivityTracker()
    for i in range(0, 31):
        t = i / FPS; tr.update(t, ang(0))
    for i in range(31, 46):                                   # 0 -> 60 degrees in 0.5 s
        t = i / FPS; f = tr.update(t, ang(60 * (i - 30) / 15))
    assert f.rapid is True
    for i in range(46, 46 + int(1.5 * FPS)):                  # ends t = 3.0; last trigger was ~2.1
        f = tr.update(i / FPS, ang(60))
    assert f.rapid is True
    for i in range(46 + int(1.5 * FPS), 46 + int(3.0 * FPS)): # ends t = 4.5
        f = tr.update(i / FPS, ang(60))
    assert f.rapid is False

def test_rula_muscle_use_after_ten_minutes_static():
    _, f = run(lambda t: ang(30), 601)
    assert f.rula_muscle_use is True
    _, f = run(lambda t: ang(30), 300)
    assert f.rula_muscle_use is False
```
```python
# tests/test_events.py
from dataclasses import replace
from linesafe.events import EventDetector
from linesafe.reba import RebaScore, Band

FPS = 30
BASE = RebaScore(t=0.0, total=9, band=Band.HIGH, score_a=8, score_b=6, table_c=9, activity=0, parts={}, side="right",
                 drivers=("trunk flexion 64°",), partial=False, missing=(), assumed=())
def sc(total, drivers=("trunk flexion 64°",)):
    return replace(BASE, total=total, drivers=drivers)

def feed(det, seq):
    closed = []
    t = 0.0
    for seconds, total in seq:
        for _ in range(int(round(seconds * FPS))):
            e = det.update(t, None if total is None else sc(total))
            if e: closed.append((t, e))
            t += 1 / FPS
    return closed

def test_short_high_never_opens():
    d = EventDetector("S1")
    assert feed(d, [(2.9, 9), (5, 5)]) == [] and not d.active

def test_event_opens_and_closes_after_three_seconds_below_high():
    d = EventDetector("S1")
    closed = feed(d, [(5, 9), (3.2, 7)])
    assert len(closed) == 1
    t_close, e = closed[0]
    assert e.station == "S1" and e.t_start == 0.0 and abs(e.t_end - (5 - 1 / FPS)) < 1e-6
    assert e.peak == 9 and e.band is Band.HIGH and abs(e.duration_s - e.t_end) < 1e-9
    assert abs(t_close - (5 + 3)) < 0.05

def test_short_dip_keeps_one_event():
    d = EventDetector("S1")
    closed = feed(d, [(4, 9), (1, 5), (4, 11), (4, 3)])
    assert len(closed) == 1 and closed[0][1].peak == 11 and closed[0][1].band is Band.VERY_HIGH

def test_none_counts_as_below():
    d = EventDetector("S1")
    assert len(feed(d, [(4, 9), (3.2, None)])) == 1

def test_flush_closes_active_only():
    d = EventDetector("S1"); feed(d, [(4, 9)])
    assert d.active and d.flush().peak == 9 and not d.active
    d2 = EventDetector("S1"); feed(d2, [(1, 9)])
    assert d2.flush() is None
```
```python
# tests/test_store.py
from linesafe.events import Event
from linesafe.reba import Band
from linesafe.store import EventStore

def ev(t0, peak=9):
    return Event(station="S1", t_start=t0, t_end=t0 + 5, peak=peak, band=Band.HIGH, drivers=("trunk flexion 64°",), duration_s=5.0)

def test_events_newest_first_and_since(tmp_path):
    s = EventStore(tmp_path / "a.db"); s.add_event(ev(100)); s.add_event(ev(200, 10))
    rows = s.events()
    assert [r["t_start"] for r in rows] == [200, 100] and rows[0]["drivers"] == ["trunk flexion 64°"]
    assert [r["t_start"] for r in s.events(since=150)] == [200]

def test_status_upsert_and_persistence(tmp_path):
    p = tmp_path / "a.db"; s = EventStore(p)
    s.set_status("S1", 1.0, 5, "medium", 4, ("x",), False, 27.5)
    s.set_status("S1", 2.0, 9, "high", 6, ("y",), True, 26.0)
    s.close()
    rows = EventStore(p).status()
    assert len(rows) == 1 and rows[0]["reba_total"] == 9 and rows[0]["partial"] in (1, True)

def test_reader_sees_writer(tmp_path):
    p = tmp_path / "a.db"; w = EventStore(p); r = EventStore(p)
    w.add_event(ev(1))
    assert len(r.events()) == 1
```

---

### Task 7: RULA scoring (logic, TDD)

**Files:** Create `src/linesafe/rula.py`, `tests/test_rula.py`.

**Interfaces — Consumes:** `tables.rula_table_a/b/c`; `geometry` types; `config.StationConfig`; `reba.upper_arm_score`.
**Produces:**
```python
class RulaLevel(str, Enum):
    ACCEPTABLE = "acceptable"; INVESTIGATE = "investigate"; CHANGE_SOON = "change soon"; CHANGE_NOW = "change now"
def rula_level(total: int) -> RulaLevel           # 1-2 | 3-4 | 5-6 | 7, else ValueError
def rula_neck_score(flex_deg: float) -> int       # -5<=n<=10 ->1; 10<n<=20 ->2; n>20 ->3; n<-5 ->4
def rula_trunk_score(flex_deg: float, twisted: bool) -> int   # |t|<=5 ->1; 5<t<=20 ->2; 20<t<=60 ->3; t>60 ->4; t<-5 ->2; +1 twisted
def rula_legs_score(bilateral: bool) -> int       # True ->1, False ->2
def rula_lower_arm_score(flex_deg: float) -> int  # 60<=l<=100 ->1 else 2
@dataclass(frozen=True)
class RulaScore:
    t: float
    total: int                 # 1-7
    level: RulaLevel
    score_a: int               # table A + muscle + force
    score_b: int               # table B + muscle + force
    parts: dict[str, int]      # upper_arm, lower_arm, wrist, wrist_twist, neck, trunk, legs, muscle, force
    side: str
    partial: bool
    missing: tuple[str, ...]
def score_rula(angles: Angles, cfg: StationConfig, muscle_use: bool = False) -> RulaScore | None
```
Rules: upper arm = `reba.upper_arm_score(u, cfg.arm_supported)`; lower arm = `rula_lower_arm_score`; wrist =
`cfg.rula_wrist`, twist = `cfg.rula_wrist_twist`; muscle = `int(muscle_use)`; force = `cfg.rula_force`;
`score_a = rula_table_a(upper, lower, wrist, twist) + muscle + force`;
`score_b = rula_table_b(min(neck, 6), min(trunk, 6), legs) + muscle + force`; `total = rula_table_c(score_a, score_b)`.
Side selection: the side with the larger `rula_table_a` value, ties → larger upper-arm flexion → left.
Missing handling mirrors `score_reba`: trunk Missing → None; neck Missing → 1 + partial; legs Missing → 1 + partial;
no Measured upper arm → upper 1, lower 2, side "none", partial; lower arm Missing on the chosen side → 2 + partial;
exactly one upper arm Missing → `"upper arm <side>"` listed, not partial (same as `score_reba` after the Task 4 review);
twist Missing → not twisted, listed, not partial. Worksheet note in the module docstring: muscle use uses
"static > 10 min or repeated ≥ 4/min" (ErgoPlus and IEH agree), computed upstream in `activity.py`.

**Steps:**
- [ ] 1. Tests below → FAIL. - [ ] 2. Implement → PASS, suite green. - [ ] 3. Commit `feat: RULA scoring`.

```python
import pytest
from linesafe.config import StationConfig
from linesafe.geometry import Angles, Measured, Missing, View
from linesafe import rula as U

def angles(trunk=0.0, twisted=False, neck=0.0, ua=(0.0, 0.0), la=(0.0, 0.0), bilateral=True):
    m = lambda v: v if isinstance(v, Missing) else Measured(float(v), 0.9)
    return Angles(t=1.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=twisted, neck_flex=m(neck),
                  upper_arm=(m(ua[0]), m(ua[1])), lower_arm=(m(la[0]), m(la[1])), knee=(m(0), m(0)),
                  legs_bilateral=bilateral)

@pytest.mark.parametrize("n,exp", [(0, 1), (10, 1), (10.01, 2), (20, 2), (20.01, 3), (-5, 1), (-5.01, 4)])
def test_neck(n, exp):
    assert U.rula_neck_score(n) == exp

@pytest.mark.parametrize("t,exp", [(0, 1), (5, 1), (5.01, 2), (20, 2), (20.01, 3), (60, 3), (60.01, 4), (-10, 2)])
def test_trunk(t, exp):
    assert U.rula_trunk_score(t, False) == exp and U.rula_trunk_score(t, True) == exp + 1

@pytest.mark.parametrize("total,level", [(1, U.RulaLevel.ACCEPTABLE), (2, U.RulaLevel.ACCEPTABLE),
    (3, U.RulaLevel.INVESTIGATE), (4, U.RulaLevel.INVESTIGATE), (5, U.RulaLevel.CHANGE_SOON),
    (6, U.RulaLevel.CHANGE_SOON), (7, U.RulaLevel.CHANGE_NOW)])
def test_levels(total, level):
    assert U.rula_level(total) is level

def test_upright_is_acceptable():
    s = U.score_rula(angles(), StationConfig(station_id="S1"))
    # upper1 lower2 wrist1 twist1 -> A 2; neck1 trunk1 legs1 -> B 1; C[2][1] = 2
    assert (s.score_a, s.score_b, s.total, s.level) == (2, 1, 2, U.RulaLevel.ACCEPTABLE)

def test_lifting_box_is_change_now():
    cfg = StationConfig(station_id="S1", rula_wrist=2, rula_force=3)
    s = U.score_rula(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90)), cfg)
    # right arm: upper4 lower1 wrist2 twist1 -> A 4 + 0 + 3 = 7; neck3 trunk4 legs1 -> B 5 + 0 + 3 = 8; C[7][7+] = 7
    assert (s.score_a, s.score_b, s.total, s.level, s.side) == (7, 8, 7, U.RulaLevel.CHANGE_NOW, "right")

def test_muscle_use_adds_to_both_groups():
    s0 = U.score_rula(angles(), StationConfig(station_id="S1"))
    s1 = U.score_rula(angles(), StationConfig(station_id="S1"), muscle_use=True)
    assert (s1.score_a, s1.score_b) == (s0.score_a + 1, s0.score_b + 1)

def test_missing_trunk_is_none():
    assert U.score_rula(angles(trunk=Missing("facing unknown")), StationConfig(station_id="S1")) is None
```

---

### Task 8: Pipeline, overlay and CLI (`run`, `replay`, `extract`) (glue + one end-to-end test)

**Files:** Create `src/linesafe/pipeline.py`, `src/linesafe/overlay.py`, `src/linesafe/cli.py`, `tests/synth.py`,
`tests/test_pipeline.py`, `tests/test_cli.py`, `tests/test_overlay.py`.

**Interfaces — Consumes:** everything above: `PoseBackend`, `ReplayBackend`, `make_backend`, `load_keypoints`,
`save_keypoints`, `extract_keypoints`, `Tracker`, `compute_angles`, `classify_view`, `View`, `score_reba`,
`score_rula`, `ActivityTracker`, `EventDetector`, `EventStore`, `load_station`, `SKELETON`.
**Produces:**
```python
# pipeline.py
@dataclass(frozen=True)
class PipelineParams:
    tracker: TrackerParams = TrackerParams()
    geometry: GeometryParams = GeometryParams()
    activity: ActivityParams = ActivityParams()
    events: EventParams = EventParams()
    wrong_view_s: float = 2.0      # FRONT continuously this long -> wrong_view
    status_every_s: float = 0.5    # store.set_status at most this often
@dataclass(frozen=True)
class FrameResult:
    t: float
    pose: PoseFrame | None
    view: View | None
    wrong_view: bool
    angles: Angles | None
    reba: RebaScore | None
    rula: RulaScore | None
    activity: ActivityFlags
    event_active: bool
    closed_event: Event | None
class Pipeline:
    def __init__(self, backend: PoseBackend, cfg: StationConfig, params: PipelineParams = PipelineParams(),
                 store: EventStore | None = None, wall_offset: float = 0.0) -> None
    # t passed to step() is monotonic (strictly increasing); everything written to the store is t + wall_offset
    def step(self, t: float, frame_bgr: np.ndarray | None, idx: int, fps: float = 0.0) -> FrameResult
    def close(self) -> Event | None     # flushes the event detector (stores the event); does NOT close the backend

# overlay.py
BAND_BGR: dict[Band, tuple[int, int, int]]   # negligible/low (87,139,46), medium (13,147,217), high (47,105,228), very high (27,55,200)
def draw(frame_bgr: np.ndarray, result: FrameResult, cfg: StationConfig, fps: float) -> np.ndarray   # draws in place, returns frame
```
`Pipeline.step`: `dets = backend.infer(frame_bgr, idx)` → `pose = tracker.update(t, dets)`; pose None → view None,
angles None, reba/rula None; else `view = classify_view(pose, params.geometry)`, wrong-view debounce (FRONT since ≥ `wrong_view_s` →
True; any SIDE resets), `angles = compute_angles(pose, params.geometry, view_ok=not wrong_view)` (and
`classify_view(pose, params.geometry)` — both calls use the same params);
`flags = activity.update(t, angles)`; `reba = score_reba(angles, cfg, flags.reba_points) if angles else None`;
`rula = score_rula(angles, cfg, flags.rula_muscle_use) if angles else None`; `closed = events.update(t, reba)`;
store (if any): `add_event(replace(closed, t_start=closed.t_start + wall_offset, t_end=closed.t_end + wall_offset))`
when closed; `set_status(station, t + wall_offset, ...)` when `t - last_status_t >= status_every_s`. `close()` stores a
flushed event the same way.

`overlay.draw` must show: skeleton segments whose both ends have conf ≥ 0.3; angle labels (trunk at hip mid, upper
arm at shoulder, elbow, knee) as integers with "°"; a top-left panel with station id, `REBA <total> <band>` on a
band-coloured chip, `RULA <total> <level>`, first two drivers; a bottom REBA ruler of 15 cells coloured by band
with the current total outlined; a 12 px red frame border while `event_active`; a yellow line
`PARTIAL: <missing joined>` when `reba.partial`; a red banner `WRONG CAMERA VIEW - station expects a side view`
when `wrong_view`; `no person in view` when pose None; `FPS <x.x>` top-right, red when `0 < fps < 15`.

CLI (`typer`, `app = typer.Typer(no_args_is_help=True)`):
- `linesafe run --source TEXT (camera index or video path, default "0") --station PATH --backend ultralytics|hailo
  --device TEXT? --db PATH (default linesafe.db) --record PATH?` — opens `cv2.VideoCapture`, `t = time.monotonic()` per
  frame (activity, events and tracker all reject time that goes backwards; wall-clock can jump on an NTP sync), the
  Pipeline gets `wall_offset = time.time() - time.monotonic()` captured once at start, FPS = exponential moving average of 1/dt (α 0.1), `overlay.draw`, `cv2.imshow("LineSafe", …)`; key `s` saves
  `pilot/<station>_<YYYYmmdd-HHMMSS-fff>.png` (drawn frame) and `.json` (`t`, angles as degrees or reason strings,
  `reba` parts/total/band, `rula` parts/total/level); key `q` or ESC quits; `--record` writes the raw frames to mp4 at the
  capture's fps. On exit: `pipeline.close()`, `backend.close()`, release capture/writer, destroy windows. Exit 1 on
  backend errors, 2 on a bad station file (message on stderr).
- `linesafe replay --keypoints PATH --station PATH [--db PATH] [--out PATH (JSONL)] [--events-out PATH (JSON)]
  [--render-dir DIR --render-every N (default 15)]` — `ReplayBackend`, `t = idx / raw.fps`, frame `None`; JSONL line per
  frame `{"t", "reba", "band", "rula", "level", "event_active", "partial"}`; events JSON list of dicts; render: overlay
  on a black `raw.height × raw.width` canvas every N frames to `DIR/frame_00000.png`. Prints a one-line summary
  `frames=<n> events=<k> max_reba=<m>`.
- `linesafe extract VIDEO --out PATH --backend ultralytics --device?` — `extract_keypoints` + `save_keypoints`.

`tests/synth.py`: `side_pose(trunk_deg, arm_deg=0.0, elbow_deg=0.0, knee_deg=0.0, facing=1, x0=640.0, y_hip=500.0,
torso=200.0) -> tuple[np.ndarray, np.ndarray]` returning `(kpts (17,2), conf (17,))` for a side-view person built so that
`compute_angles` returns those angles (shoulders/hips/ears overlapped ±2 px; nose 30 px ahead of the ear; upper arm
length 0.55·torso, forearm 0.5·torso, thigh 0.9·torso, shin 0.9·torso; ear placed so neck flexion is 0);
`front_pose(x0=640.0, y_hip=500.0, torso=200.0)` (shoulders x0 ± 70, hips x0 ± 40, ears x0 ± 25, nose at x0 — so
`classify_view` is FRONT and facing is None); `session(segments: list[tuple[float, dict]], fps=30.0, ramp_s=0.5)
-> RawDetections` where each segment is `(seconds, kwargs)` for `side_pose`, or `(seconds, {"front": True})` for
`front_pose`; between two side segments the angles are linearly interpolated over the first `ramp_s` of the later
segment (a real body never teleports, and a jump would break the tracker's IoU follow); bbox = min/max of the
points padded 20 px; width 1280, height 720.

**Steps:**
- [ ] 1. `tests/synth.py` + `tests/test_pipeline.py`:
  `test_synth_pose_roundtrip` (for trunk 64, arm 95, elbow 90, knee 40 → `compute_angles` within 1°);
  `test_replay_bending_session_makes_one_high_event` (session: 3 s upright, 6 s trunk 64/arm 95/elbow 90/knee 40,
  5 s upright; station `reba_load=2, reba_coupling=1, reba_wrist=2`; run `Pipeline(ReplayBackend(file))` over all
  frames then `close()` → exactly one closed event, peak ≥ 8, `3.0 <= t_start <= 4.0` (the 0.5 s ramp plus 1€ lag of a few frames put the first High frame near 3.5 s), stored in a tmp `EventStore`);
  `test_front_view_sets_wrong_view_after_two_seconds` (session of a front-facing pose for 3 s → `wrong_view` False at
  1.9 s, True at 2.1 s, `reba is None`). → FAIL first, then implement `pipeline.py` → PASS.
- [ ] 2. `tests/test_overlay.py`: draw on a 720×1280 black frame for (a) a High result with event active and partial,
  (b) pose None, (c) wrong view — returns same shape/dtype, pixels changed. Implement `overlay.py` → PASS.
- [ ] 3. `tests/test_cli.py` with `typer.testing.CliRunner`: `replay` on a synth session writes JSONL with one line per
  frame and an events JSON of length 1; bad station file → exit code 2. Implement `cli.py` → PASS; suite green.
- [ ] 4. → verify (orchestrator will look at the images): `uv run linesafe replay --keypoints <synth file written by a
  short script into /tmp> --station stations/example.toml --render-dir /tmp/linesafe_render --render-every 30` prints the
  summary line and writes PNGs; list them in the report.
- [ ] 5. Commit: `feat: pipeline, overlay and run/replay/extract CLI`.

**Named failure modes owned here:** wrong camera view banner (pipeline test + overlay); no person → no score and
"no person in view" (overlay test b); FPS below 15 shown red (overlay); `run` itself is not exercised by tests
(camera) — held back for the orchestrator's live check.

---

### Task 9: LAN dashboard (glue)

**Files:** Create `src/linesafe/web.py`, `tests/test_web.py` · Modify `src/linesafe/cli.py` (add `web` command).

**Interfaces — Consumes:** `EventStore`.
**Produces:** `create_app(db_path: Path) -> FastAPI` (fastapi imported inside the function or module guarded so
the rest of the package never imports it); routes:
- `GET /api/status` → `store.status()`.
- `GET /api/events?limit=50&since=<float>` → `store.events(limit, since)`.
- `GET /api/summary?days=7` → `{"days": 7, "events": n, "by_band": {"high": n, "very high": n}, "top_drivers": [[driver, count], …≤5],
  "total_high_seconds": s}` over events with `t_start >= now - days*86400` (use `time.time()`).
- `GET /` → one self-contained HTML page (inline CSS/JS, no external URLs), mobile-first, polling `/api/status` and
  `/api/events` every 2 s: per station a card with the REBA total large, the band word on its band colour, RULA total and
  level, first two drivers, "updated N s ago"; a list of today's events (time, peak, band, duration, drivers); a 7-day
  summary block titled "This week" from `/api/summary` with the line "Plain-language weekly summary: generated on the
  UGen300 in Stage II." Title `LineSafe`.
- CLI `linesafe web --db PATH --host 0.0.0.0 --port 8080` runs uvicorn; if fastapi/uvicorn are missing, exit 1 with
  "install the web extra: uv sync --extra web".

**Steps:**
- [ ] 1. `tests/test_web.py` with `fastapi.testclient.TestClient` (skip the module with `pytest.importorskip("fastapi")`):
  seeded store (two events, one status) → `/api/status` 200 with one row; `/api/events?limit=1` one row newest;
  `/api/summary` counts both events; `/` 200, `text/html`, contains `LineSafe` and no `http://` or `https://` substrings.
  → FAIL, implement → PASS; suite green.
- [ ] 2. → verify: `uv run linesafe web --db /tmp/linesafe_demo.db --port 8765 &` after seeding that db with a short
  script; `curl -s localhost:8765/api/summary` returns JSON; stop the server. Paste output in the report.
- [ ] 3. Commit: `feat: LAN dashboard with status, events and weekly summary`.

---

### Task 10 (orchestrator + Wayne; not dispatched): live samples and golden session

Held back because it needs a camera and a person. The orchestrator prepares, Wayne performs.
1. FPS: `uv run linesafe run --source 0 --station stations/example.toml` with the 1080p webcam; note the steady FPS.
2. Scripted session (≈ 2 min, side view, box ≤ 10 kg): upright 10 s → bend to pick up 10 s → hold bent 10 s → upright
   10 s → arms raised overhead 10 s → squat 10 s → one-leg stance 5 s; record with `--record session.mp4`.
3. `uv run linesafe extract session.mp4 --out tests/fixtures/session.keypoints.json`; replay it; commit keypoints +
   `tests/golden/session.results.json` + `tests/test_golden.py` (replay equals golden; no torch).
4. Pilot check: 20 postures held 5 s, `s` pressed each time; Wayne measures trunk/neck/upper-arm/elbow/knee on the saved
   PNGs with a protractor tool and fills the REBA worksheet by hand; orchestrator computes per-angle error and REBA
   exact-match rate into `docs/evidence/2026-10-pilot-check.md`, which feeds deck slide 12.
5. `linesafe web` on the laptop, phone on the same Wi-Fi opens `http://<laptop-ip>:8080` — screenshot for the video.
