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

# --- Fix round 1: review minors M1, M6, M7 ---

def test_top_level_key_outside_station(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('reba_load = 2\n[station]\nid = "S7"\n')
    with pytest.raises(ValueError, match="reba_load"):
        load_station(p)

def test_blank_station_id():
    with pytest.raises(ValueError, match="station_id"):
        StationConfig(station_id="   ")

@pytest.mark.parametrize("field,value", [("reba_shock", 1), ("arm_supported", "yes"), ("reba_load", True)])
def test_type_checks_name_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        StationConfig(station_id="S1", **{field: value})

def test_literal_station_id_key_rejected(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[station]\nid = "S7"\nstation_id = "S8"\n')
    with pytest.raises(ValueError, match="station_id"):
        load_station(p)
