from src.artifacts import load_joblib, load_json, save_joblib, save_json


def test_json_round_trip(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    save_json({"metric": 0.75}, destination)
    assert load_json(destination) == {"metric": 0.75}


def test_joblib_round_trip(tmp_path):
    destination = tmp_path / "nested" / "model.joblib"
    save_joblib({"value": [1, 2, 3]}, destination)
    assert load_joblib(destination) == {"value": [1, 2, 3]}

