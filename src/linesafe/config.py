"""Per-station inputs a camera cannot see, set at calibration.

These are the only defaults for non-visual scoring inputs. ``load_station`` reads
them from the ``[station]`` table of a TOML file; ``id`` maps to ``station_id``
and every other key equals a field name.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path


def _check_int(name: str, value: object, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"StationConfig.{name} must be an int, got {value!r}")
    if not low <= value <= high:
        raise ValueError(f"StationConfig.{name} must be {low}..{high}, got {value}")


def _check_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"StationConfig.{name} must be a bool, got {value!r}")


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    reba_load: int = 0  # 0: < 5 kg, 1: 5-10 kg, 2: > 10 kg
    reba_shock: bool = False  # +1 shock or rapid build-up of force
    reba_coupling: int = 0  # 0 good, 1 fair, 2 poor, 3 unacceptable
    reba_wrist: int = 1  # final REBA wrist score 1-3 (incl. deviation/twist)
    arm_supported: bool = False  # -1 upper arm in REBA and RULA
    rula_wrist: int = 1  # 1-4 (incl. bent-from-midline)
    rula_wrist_twist: int = 1  # 1-2
    rula_force: int = 0  # 0-3
    roi: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2 pixels; None = whole frame

    def __post_init__(self) -> None:
        if not isinstance(self.station_id, str) or not self.station_id.strip():
            raise ValueError(
                f"StationConfig.station_id must be a non-blank str, got {self.station_id!r}"
            )
        _check_int("reba_load", self.reba_load, 0, 2)
        _check_bool("reba_shock", self.reba_shock)
        _check_int("reba_coupling", self.reba_coupling, 0, 3)
        _check_int("reba_wrist", self.reba_wrist, 1, 3)
        _check_bool("arm_supported", self.arm_supported)
        _check_int("rula_wrist", self.rula_wrist, 1, 4)
        _check_int("rula_wrist_twist", self.rula_wrist_twist, 1, 2)
        _check_int("rula_force", self.rula_force, 0, 3)
        if self.roi is not None:
            roi = self.roi
            if (
                not isinstance(roi, tuple)
                or len(roi) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)
            ):
                raise ValueError(
                    f"StationConfig.roi must be a tuple of 4 ints (x1, y1, x2, y2), got {roi!r}"
                )
            x1, y1, x2, y2 = roi
            if not (x1 < x2 and y1 < y2):
                raise ValueError(f"StationConfig.roi must have x1 < x2 and y1 < y2, got {roi!r}")


_FIELDS = {f.name for f in dataclasses.fields(StationConfig)}


def load_station(path: Path | str) -> StationConfig:
    """Read the ``[station]`` table of a TOML file; unknown keys raise ``ValueError``."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    for key in data:
        if key != "station":
            raise ValueError(f"{path}: unknown top-level key {key!r} (keys belong in [station])")
    table = data.get("station")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: missing [station] table")
    kwargs: dict[str, object] = {}
    for key, value in table.items():
        if key == "id":
            kwargs["station_id"] = value
        elif key in _FIELDS and key != "station_id":
            kwargs[key] = tuple(value) if key == "roi" and isinstance(value, list) else value
        else:
            raise ValueError(f"{path}: unknown key {key!r} in [station]")
    if "station_id" not in kwargs:
        raise ValueError(f"{path}: [station] needs id (station_id)")
    return StationConfig(**kwargs)  # type: ignore[arg-type]
