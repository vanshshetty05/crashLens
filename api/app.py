"""CrashLens API.

Endpoints:

    GET  /health        liveness check and the name of the loaded model
    GET  /api/options   valid states, regions, factor groups and values per group
    POST /api/predict   severity level, probabilities, top three factors and history
    GET  /api/map       state shapes with severity values (GeoJSON)

The model is the one behind ``src.explain.explain_one`` (the pipeline with the
highest tuning macro-F1) and is loaded once at startup.

Run from the project root:

    python -m api.app

The server listens on 127.0.0.1 only. Browsers may call it only from the
frontend origins: http://127.0.0.1:8000 and http://localhost:8000 by default,
or a comma-separated list in the CRASHLENS_FRONTEND_ORIGINS environment variable.
"""
from __future__ import annotations

import os

import pandas as pd
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from src.data import PROJECT_ROOT, SEVERITY_ORDER, load_data

HOST = "127.0.0.1"
PORT = 5000
DEFAULT_ORIGINS = ["http://127.0.0.1:8000", "http://localhost:8000"]
MAP_PATH = PROJECT_ROOT / "frontend" / "data" / "india_states_severity.geojson"

# The model was trained on 2019 to 2022. A few years past the data are allowed for "what if" queries.
MAX_YEAR_AHEAD = 5
REQUIRED_FIELDS = ["state", "factor_group", "factor_value"]


class BadRequest(Exception):
    """A request the client can fix. Returned as HTTP 400 with the message."""


def frontend_origins() -> list[str]:
    configured = os.environ.get("CRASHLENS_FRONTEND_ORIGINS", "")
    return [o.strip() for o in configured.split(",") if o.strip()] or DEFAULT_ORIGINS


class Dataset:
    """What the API needs from the raw CSV: the valid choices and the yearly history.

    All rows are kept, including low-count rows, so the history shows every
    year that was reported.
    """

    def __init__(self, df: pd.DataFrame | None = None):
        df = load_data(drop_low_count=False) if df is None else df
        states = df.drop_duplicates("state_harmonised").set_index("state_harmonised")
        self.state_info = states[["region", "state_type"]].sort_index()
        self.regions = sorted(df["region"].unique())
        self.factor_values = {
            group: sorted(values.unique()) for group, values in df.groupby("factor_group")["factor_value"]
        }
        self.min_year, self.max_year = int(df["year"].min()), int(df["year"].max())
        self.national = df.groupby("year")["national_severity"].first()
        # state_harmonised merges some units (Jammu & Kashmir with Ladakh), so counts are summed
        # per year and the severity index is recomputed from the sums.
        self.history = (
            df.groupby(["state_harmonised", "factor_group", "factor_value", "year"])[["accidents", "killed"]]
            .sum().sort_index()
        )

    def options(self) -> dict:
        return {
            "states": list(self.state_info.index),
            "regions": self.regions,
            "factor_groups": sorted(self.factor_values),
            "factor_values": self.factor_values,
            "years": {"min": self.min_year, "max": self.max_year + MAX_YEAR_AHEAD, "default": self.max_year},
        }

    def history_for(self, state: str, group: str, value: str) -> list[dict]:
        """Accidents, deaths and severity index for each reported year, oldest first."""
        key = (state, group, value)
        if key not in self.history.index:
            return []
        rows = []
        for year, row in self.history.loc[key].iterrows():
            accidents, killed = int(row["accidents"]), int(row["killed"])
            rows.append({
                "year": int(year), "accidents": accidents, "killed": killed,
                "severity_index": round(100 * killed / accidents, 2) if accidents else None,
                "national_severity": round(float(self.national[year]), 2),
            })
        return rows


def _text(body: dict, field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BadRequest(f"'{field}' is required and must be a non-empty string.")
    return value.strip()


def parse_prediction_request(body, data: Dataset) -> dict:
    """Validate the request body and return the row to predict. Raises BadRequest with a clear message."""
    if not isinstance(body, dict):
        raise BadRequest("Send a JSON object with 'state', 'factor_group' and 'factor_value'.")
    missing = [f for f in REQUIRED_FIELDS if f not in body]
    if missing:
        raise BadRequest(f"Missing required field(s): {', '.join(missing)}.")

    state, group, value = (_text(body, f) for f in REQUIRED_FIELDS)
    if state not in data.state_info.index:
        raise BadRequest(f"Unknown state '{state}'. See /api/options for the valid names.")
    if group not in data.factor_values:
        raise BadRequest(f"Unknown factor_group '{group}'. Valid groups: {', '.join(sorted(data.factor_values))}.")
    if value not in data.factor_values[group]:
        raise BadRequest(
            f"'{value}' is not a factor_value of '{group}'. Valid values: {', '.join(data.factor_values[group])}."
        )

    year = body.get("year", data.max_year)
    first, last = data.min_year, data.max_year + MAX_YEAR_AHEAD
    # bool is a subclass of int, so it is ruled out by name.
    if isinstance(year, bool) or not isinstance(year, int) or not first <= year <= last:
        raise BadRequest(f"'year' must be a whole number from {first} to {last}.")

    history = data.history_for(state, group, value)
    reported = [h for h in history if h["accidents"] > 0]
    if not reported:
        raise BadRequest(f"The dataset has no accidents for '{value}' in {state}, so there is nothing to predict from.")
    # The model needs an accident count. Use the requested year if it was reported, else the latest earlier one.
    basis = next((h for h in reversed(reported) if h["year"] <= year), reported[0])

    return {
        "row": {
            "state_harmonised": state, "region": data.state_info.at[state, "region"],
            "state_type": data.state_info.at[state, "state_type"],
            "factor_group": group, "factor_value": value, "year": year, "accidents": basis["accidents"],
        },
        "accidents_year": basis["year"],
        "history": history,
    }


def create_app(explain=None, data: Dataset | None = None, map_path=MAP_PATH) -> Flask:
    """Build the app. The model is loaded here, before the first request.

    ``explain`` replaces ``explain_one`` in tests, which then run without the model files.
    """
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024
    CORS(app, origins=frontend_origins(), methods=["GET", "POST"], allow_headers=["Content-Type"])

    data = data or Dataset()
    if explain is None:
        from src.explain import explain_one as explain, load_model
        model_name = load_model()
    else:
        model_name = "test double"
    map_text = map_path.read_text(encoding="utf-8") if map_path.exists() else None

    @app.errorhandler(BadRequest)
    def bad_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def server_error(error):
        app.logger.exception("Unhandled error")
        return jsonify(error="Internal server error."), 500

    @app.get("/health")
    def health():
        return jsonify(status="ok", model=model_name, map_available=map_text is not None)

    @app.get("/api/options")
    def options():
        return jsonify(data.options())

    @app.post("/api/predict")
    def predict():
        if not request.is_json:
            raise BadRequest("Send the request as JSON with the header Content-Type: application/json.")
        body = request.get_json(silent=True)
        if body is None:
            raise BadRequest("The request body is not valid JSON.")
        parsed = parse_prediction_request(body, data)
        row = parsed["row"]
        result = explain(row)
        return jsonify(
            severity_level=result["prediction"],
            probabilities={level: result["probabilities"][level] for level in SEVERITY_ORDER},
            top_factors=result["factors"],
            history=parsed["history"],
            input={
                "state": row["state_harmonised"], "region": row["region"], "state_type": row["state_type"],
                "factor_group": row["factor_group"], "factor_value": row["factor_value"], "year": row["year"],
                "accidents": row["accidents"], "accidents_year": parsed["accidents_year"],
            },
            model=model_name,
        )

    @app.get("/api/map")
    def state_map():
        if map_text is None:
            return jsonify(error="The map file is missing. Run 'python -m src.maps' to create it."), 503
        return Response(map_text, mimetype="application/geo+json")

    return app


if __name__ == "__main__":
    create_app().run(host=HOST, port=PORT, debug=False)
