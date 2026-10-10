"""Tests for api/app.py: endpoints, input validation, CORS and one call through the real model."""
import pytest

from api.app import DEFAULT_ORIGINS, HOST, MAP_PATH, Dataset, create_app
from src.models import MODELS_DIR

VALID = {"state": "Karnataka", "factor_group": "weather", "factor_value": "Rainy"}


def fake_explain(row):
    """Stands in for explain_one so that most tests need no model files."""
    fake_explain.last_row = row
    return {
        "prediction": "High",
        "probabilities": {"Low": 0.1, "Medium": 0.2, "High": 0.7},
        "factors": [
            {"feature": f, "label": f, "value": str(row[f]), "contribution": 0.1, "effect": "supports"}
            for f in ("state_harmonised", "region", "factor_value")
        ],
    }


@pytest.fixture(scope="module")
def data():
    return Dataset()


@pytest.fixture(scope="module")
def client(data):
    return create_app(explain=fake_explain, data=data).test_client()


def test_server_binds_to_localhost_only():
    assert HOST == "127.0.0.1"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_options_lists_every_choice(client):
    options = client.get("/api/options").get_json()
    assert "Karnataka" in options["states"] and len(options["states"]) == 35
    assert set(options["regions"]) == {"Central", "East", "North", "North-East", "South", "West"}
    assert len(options["factor_groups"]) == 9
    assert set(options["factor_values"]) == set(options["factor_groups"])
    assert "Rainy" in options["factor_values"]["weather"]
    assert options["years"]["default"] == 2024


def test_predict_returns_prediction_factors_and_history(client):
    response = client.post("/api/predict", json=VALID)
    assert response.status_code == 200
    body = response.get_json()
    assert body["severity_level"] == "High"
    assert set(body["probabilities"]) == {"Low", "Medium", "High"}
    assert len(body["top_factors"]) == 3
    assert [h["year"] for h in body["history"]] == [2019, 2020, 2021, 2022, 2023, 2024]
    assert all({"accidents", "killed", "severity_index", "national_severity"} <= set(h) for h in body["history"])
    # Region, State/UT and the accident count are filled in from the dataset.
    assert body["input"]["region"] == "South" and body["input"]["state_type"] == "State"
    assert body["input"]["year"] == 2024 and body["input"]["accidents_year"] == 2024
    assert body["input"]["accidents"] == body["history"][-1]["accidents"]


def test_history_severity_is_killed_per_100_accidents(client):
    latest = client.post("/api/predict", json=VALID).get_json()["history"][-1]
    assert latest["severity_index"] == pytest.approx(100 * latest["killed"] / latest["accidents"], abs=0.01)


def test_predict_uses_the_requested_year(client):
    body = client.post("/api/predict", json={**VALID, "year": 2021}).get_json()
    assert body["input"]["year"] == 2021 and body["input"]["accidents_year"] == 2021
    assert fake_explain.last_row["year"] == 2021
    future = client.post("/api/predict", json={**VALID, "year": 2026}).get_json()
    assert future["input"]["year"] == 2026 and future["input"]["accidents_year"] == 2024


@pytest.mark.parametrize("payload,expected", [
    ({"factor_group": "weather", "factor_value": "Rainy"}, "Missing required field(s): state"),
    ({**VALID, "state": "Atlantis"}, "Unknown state 'Atlantis'"),
    ({**VALID, "state": ""}, "'state' is required"),
    ({**VALID, "state": 7}, "'state' is required"),
    ({**VALID, "factor_group": "moon_phase"}, "Unknown factor_group 'moon_phase'"),
    ({**VALID, "factor_value": "Rural"}, "'Rural' is not a factor_value of 'weather'"),
    ({**VALID, "year": "2024"}, "'year' must be a whole number"),
    ({**VALID, "year": 2024.5}, "'year' must be a whole number"),
    ({**VALID, "year": True}, "'year' must be a whole number"),
    ({**VALID, "year": 1999}, "'year' must be a whole number"),
    ({**VALID, "year": 2100}, "'year' must be a whole number"),
    (["Karnataka", "weather", "Rainy"], "Send a JSON object"),
])
def test_predict_rejects_bad_input_with_a_clear_message(client, payload, expected):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert expected in response.get_json()["error"]


def test_predict_rejects_non_json_and_broken_json(client):
    plain = client.post("/api/predict", data="state=Karnataka")
    assert plain.status_code == 400 and "JSON" in plain.get_json()["error"]
    broken = client.post("/api/predict", data="{not json", content_type="application/json")
    assert broken.status_code == 400 and "not valid JSON" in broken.get_json()["error"]


def test_predict_rejects_a_combination_with_no_accidents(client, data):
    totals = data.history.groupby(level=[0, 1, 2])["accidents"].sum()
    seen = set(totals[totals > 0].index)
    missing = next(
        ((s, g, v) for s in data.state_info.index for g, values in data.factor_values.items() for v in values
         if (s, g, v) not in seen), None,
    )
    if missing is None:
        pytest.skip("every state and category has accidents")
    response = client.post("/api/predict", json=dict(zip(VALID, missing)))
    assert response.status_code == 400
    assert "no accidents" in response.get_json()["error"]


def test_unknown_route_and_wrong_method_return_json_errors(client):
    assert client.get("/api/nothing").status_code == 404
    assert "error" in client.get("/api/nothing").get_json()
    assert client.get("/api/predict").status_code == 405
    assert "error" in client.get("/api/predict").get_json()


@pytest.mark.skipif(not MAP_PATH.exists(), reason="run python -m src.maps first")
def test_map_returns_geojson_with_severity(client):
    response = client.get("/api/map")
    assert response.status_code == 200
    assert response.mimetype == "application/geo+json"
    features = response.get_json(force=True)["features"]
    assert len(features) == 36
    assert "severity_2024" in features[0]["properties"]


def test_map_reports_a_missing_file(data, tmp_path):
    client = create_app(explain=fake_explain, data=data, map_path=tmp_path / "missing.geojson").test_client()
    response = client.get("/api/map")
    assert response.status_code == 503 and "error" in response.get_json()


def test_cors_allows_only_the_frontend_origin(client):
    allowed = client.get("/api/options", headers={"Origin": DEFAULT_ORIGINS[0]})
    assert allowed.headers.get("Access-Control-Allow-Origin") == DEFAULT_ORIGINS[0]
    other = client.get("/api/options", headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in other.headers


def test_cors_origins_can_be_configured(data, monkeypatch):
    monkeypatch.setenv("CRASHLENS_FRONTEND_ORIGINS", "http://localhost:3000")
    client = create_app(explain=fake_explain, data=data).test_client()
    assert client.get("/health", headers={"Origin": "http://localhost:3000"}).headers.get(
        "Access-Control-Allow-Origin") == "http://localhost:3000"
    assert "Access-Control-Allow-Origin" not in client.get("/health", headers={"Origin": DEFAULT_ORIGINS[0]}).headers


@pytest.mark.skipif(not any(MODELS_DIR.glob("*.joblib")), reason="run python -m src.models first")
def test_predict_with_the_real_model(data):
    """One call through the real pipeline and SHAP explainer."""
    client = create_app(data=data).test_client()
    assert "test double" not in client.get("/health").get_json()["model"]
    body = client.post("/api/predict", json={"state": "Bihar", "factor_group": "weather",
                                             "factor_value": "Foggy & Misty"}).get_json()
    assert body["severity_level"] in {"Low", "Medium", "High"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["severity_level"] == max(body["probabilities"], key=body["probabilities"].get)
    assert len(body["top_factors"]) == 3
    assert all({"feature", "label", "value", "contribution", "effect"} <= set(f) for f in body["top_factors"])
