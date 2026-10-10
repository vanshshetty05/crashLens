"""Descriptive maps, tables and profiles from the dataset alone. No model is used.

1. State choropleth of severity in 2024 and its change from 2019, and the same
   values written into a GeoJSON for the frontend.
2. Heatmap of severity by region and factor value, for weather and road feature.
3. Ranked table of the deadliest conditions in India, 2019 to 2024 combined.
4. A profile of one state (Karnataka) against the national figures.

Rules, as in notebooks/01_eda.ipynb:

- Severity is always 100 * sum(killed) / sum(accidents). A severity_index is never averaged.
- State, regional and national totals come from one factor group (collision_type),
  because every factor group is a breakdown of the same accidents.
- All rows are used, including low-count rows: their counts still belong in the sums.

Boundaries: data/external/india_states/india_states.geojson (see SOURCE.txt there).

Run from the project root:

    python -m src.maps
"""
from __future__ import annotations

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

from src.data import PROJECT_ROOT, TOTALS_GROUP, load_data, seed_everything
from src.evaluate import AXIS, FIG_DIR, INK, INK2, MUTED, SEQ, TAB_DIR, save_fig, set_style

BOUNDARIES_PATH = PROJECT_ROOT / "data" / "external" / "india_states" / "india_states.geojson"
FRONTEND_GEOJSON = PROJECT_ROOT / "frontend" / "data" / "india_states_severity.geojson"

FIRST_YEAR, LAST_YEAR = 2019, 2024
FOCUS_STATE = "Karnataka"
SMALL = 500        # below this many accidents in a year, a state's ratio is unstable
MIN_CELL = 100     # below this many accidents, a heatmap or profile cell is not coloured
HEATMAP_GROUPS = ["weather", "road_feature"]
PROFILE_GROUPS = ["collision_type", "road_feature", "weather"]

# Boundary-file names that differ from state_harmonised. The dataset reports Jammu & Kashmir
# and Ladakh as one unit, so both shapes carry the combined value.
NAME_TO_HARMONISED = {
    "Andaman & Nicobar": "Andaman & Nicobar Islands",
    "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu",
    "Jammu & Kashmir": "Jammu & Kashmir and Ladakh",
    "Ladakh": "Jammu & Kashmir and Ladakh",
}

# Degrees to move a state label so that neighbours do not overlap.
LABEL_NUDGE = {"Kerala": (-1.3, -0.6), "Tamil Nadu": (0.9, 0.5)}

BLUE = "#2a78d6"
NO_DATA = "#e1e0d9"
# Diverging: blue = severity fell, red = severity rose, neutral grey at no change.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#d03b3b", "#8c1f1f"]
)


def pretty(name: str) -> str:
    return name.replace("_", " ").capitalize()


def severity_table(d: pd.DataFrame, by) -> pd.DataFrame:
    """Summed accidents, summed killed and the resulting severity for each group."""
    t = d.groupby(by, observed=True)[["accidents", "killed"]].sum()
    t["severity"] = 100 * t["killed"] / t["accidents"].where(t["accidents"] > 0)
    return t


# ---- 1. choropleth and GeoJSON ----------------------------------------------

def load_boundaries() -> dict:
    if not BOUNDARIES_PATH.exists():
        raise FileNotFoundError(f"State boundaries not found at {BOUNDARIES_PATH}.")
    return json.loads(BOUNDARIES_PATH.read_text(encoding="utf-8"))


def harmonised_name(feature: dict) -> str:
    name = feature["properties"]["ST_NM"]
    return NAME_TO_HARMONISED.get(name, name)


def state_severity(tot: pd.DataFrame) -> pd.DataFrame:
    """One row per state_harmonised: accidents, killed and severity for every year, and the change."""
    by_year = severity_table(tot, ["state_harmonised", "year"]).unstack("year")
    wide = pd.DataFrame(index=by_year.index)
    for year in sorted(tot["year"].unique()):
        wide[f"accidents_{year}"] = by_year[("accidents", year)].fillna(0).astype(int)
        wide[f"killed_{year}"] = by_year[("killed", year)].fillna(0).astype(int)
        wide[f"severity_{year}"] = by_year[("severity", year)]
    wide["severity_change"] = wide[f"severity_{LAST_YEAR}"] - wide[f"severity_{FIRST_YEAR}"]
    wide["small_count"] = (wide[f"accidents_{FIRST_YEAR}"] < SMALL) | (wide[f"accidents_{LAST_YEAR}"] < SMALL)
    info = tot.drop_duplicates("state_harmonised").set_index("state_harmonised")[["region", "state_type"]]
    return info.join(wide).sort_values(f"severity_{LAST_YEAR}", ascending=False)


def _rings(geometry: dict) -> list[list]:
    """Each polygon of a Polygon or MultiPolygon as its list of rings (outer first, then holes)."""
    return [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]


def _patch(geometry: dict) -> PathPatch:
    vertices, codes = [], []
    for polygon in _rings(geometry):
        for ring in polygon:
            vertices.extend(ring)
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 2) + [MplPath.CLOSEPOLY])
    return PathPatch(MplPath(np.asarray(vertices)[:, :2], codes))


def _label_point(geometry: dict) -> tuple[float, float]:
    """Middle of the largest polygon's bounding box. Good enough to place a state name."""
    outer = max((np.asarray(p[0])[:, :2] for p in _rings(geometry)), key=len)
    return tuple((outer.min(axis=0) + outer.max(axis=0)) / 2)


def _draw_map(ax, boundaries, values: pd.Series, cmap, norm, labels: list[str]) -> None:
    patches, colors = [], []
    for feature in boundaries["features"]:
        value = values.get(harmonised_name(feature), np.nan)
        patches.append(_patch(feature["geometry"]))
        colors.append(NO_DATA if pd.isna(value) else cmap(norm(value)))
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.5))

    for feature in boundaries["features"]:
        name = harmonised_name(feature)
        # A merged unit is labelled once, on the shape that carries its own name.
        if name in labels and feature["properties"]["ST_NM"] in (name, name.split(" and ")[0]):
            x, y = _label_point(feature["geometry"])
            dx, dy = LABEL_NUDGE.get(name, (0, 0))
            x, y = x + dx, y + dy
            ax.text(x, y, f"{feature['properties']['ST_NM']}\n{values[name]:+.1f}" if norm.vmin < 0
                    else f"{feature['properties']['ST_NM']}\n{values[name]:.1f}",
                    ha="center", va="center", fontsize=6.5, color=INK, linespacing=1.1,
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_xlim(67.5, 98)
    ax.set_ylim(6, 37.6)
    ax.set_aspect(1 / np.cos(np.radians(22)))   # equal distances at the latitude of central India
    ax.axis("off")


def plot_choropleth(states: pd.DataFrame, boundaries: dict) -> None:
    now, change = states[f"severity_{LAST_YEAR}"], states["severity_change"]
    stable = states[~states["small_count"]]
    limit = np.ceil(stable["severity_change"].abs().max())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.6))
    panels = [
        (now, SEQ, Normalize(0, np.ceil(now.max() / 10) * 10), f"Severity in {LAST_YEAR}",
         "Persons killed per 100 accidents", "neither",
         list(stable[f"severity_{LAST_YEAR}"].nlargest(5).index) + list(stable[f"severity_{LAST_YEAR}"].nsmallest(3).index)),
        (change, DIVERGING, TwoSlopeNorm(0, -limit, limit), f"Change in severity, {FIRST_YEAR} to {LAST_YEAR}",
         "Change in persons killed per 100 accidents (red = deadlier)", "both",
         list(stable["severity_change"].nlargest(4).index) + list(stable["severity_change"].nsmallest(4).index)),
    ]
    for ax, (values, cmap, norm, title, label, extend, labels) in zip(axes, panels):
        _draw_map(ax, boundaries, values, cmap, norm, labels)
        ax.set_title(title)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal",
                          fraction=0.035, pad=0.02, shrink=0.7, extend=extend)
        cb.set_label(label)
        cb.outline.set_visible(False)
    fig.suptitle(f"Road accident severity by State and UT, India, {LAST_YEAR}",
                 x=0.01, y=0.99, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01,
             f"Labelled states have at least {SMALL:,} accidents in both years. The change scale is set by those states; "
             "smaller units beyond it take the end colour.\nJammu & Kashmir and Ladakh are reported as one unit and "
             "share one value. Grey = no accidents reported.",
             fontsize=7.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save_fig(fig, "fig11_state_severity_map")


def _rounded(coordinates, digits: int = 4):
    """Round coordinates to about 11 m, which keeps the file small for the browser."""
    if isinstance(coordinates[0], (int, float)):
        return [round(coordinates[0], digits), round(coordinates[1], digits)]
    return [_rounded(c, digits) for c in coordinates]


def write_geojson(states: pd.DataFrame, boundaries: dict) -> None:
    """The state shapes with the severity values as properties, for the Leaflet map."""
    features = []
    for feature in boundaries["features"]:
        name = harmonised_name(feature)
        properties = {"state": feature["properties"]["ST_NM"], "state_harmonised": name,
                      "shares_value_with_merged_unit": feature["properties"]["ST_NM"] != name
                      and name == "Jammu & Kashmir and Ladakh"}
        if name in states.index:
            for key, value in states.loc[name].items():
                if isinstance(value, (bool, np.bool_)):
                    properties[key] = bool(value)
                elif isinstance(value, str):
                    properties[key] = value
                elif pd.isna(value):
                    properties[key] = None
                else:
                    properties[key] = int(value) if float(value).is_integer() and "severity" not in key \
                        else round(float(value), 2)
        features.append({"type": "Feature", "properties": properties,
                         "geometry": {"type": feature["geometry"]["type"],
                                      "coordinates": _rounded(feature["geometry"]["coordinates"])}})
    FRONTEND_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    FRONTEND_GEOJSON.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved {FRONTEND_GEOJSON.relative_to(PROJECT_ROOT).as_posix()} "
          f"({FRONTEND_GEOJSON.stat().st_size / 1024:.0f} KB, {len(features)} shapes)")


# ---- 2. region x factor value heatmap ---------------------------------------

def region_factor_table(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw["factor_group"].isin(HEATMAP_GROUPS)]
    by_region = severity_table(d, ["factor_group", "factor_value", "region"])
    national = severity_table(d, ["factor_group", "factor_value"]).assign(region="India")
    national = national.set_index("region", append=True)
    return pd.concat([by_region, national]).sort_index()


def plot_region_heatmap(table: pd.DataFrame, region_order: list[str]) -> None:
    columns = region_order + ["India"]
    grids = {}
    for g in HEATMAP_GROUPS:
        sev = table.loc[g]["severity"].unstack("region")[columns]
        acc = table.loc[g]["accidents"].unstack("region")[columns]
        order = sev["India"].sort_values(ascending=False).index
        grids[g] = (sev.loc[order], acc.loc[order])
    shown = pd.concat([sev.where(acc >= MIN_CELL).stack() for sev, acc in grids.values()])
    vmin, vmax = np.floor(shown.min() / 5) * 5, np.ceil(shown.max() / 5) * 5

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3), gridspec_kw={"wspace": 0.62})
    for ax, g in zip(axes, HEATMAP_GROUPS):
        sev, acc = grids[g]
        cmap = SEQ.copy()
        cmap.set_bad(NO_DATA)
        im = ax.imshow(np.ma.masked_invalid(sev.where(acc >= MIN_CELL).values), cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(sev)), [v.replace("/", "/\n", 1) if len(v) > 24 else v for v in sev.index])
        ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(sev)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.axvline(len(columns) - 1.5, color="white", linewidth=6)    # sets the national column apart
        ax.tick_params(which="both", length=0)
        ax.get_xticklabels()[-1].set_fontweight("bold")
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(sev.shape[0]):
            for j in range(sev.shape[1]):
                v, n = sev.values[i, j], acc.values[i, j]
                if pd.isna(v) or n < MIN_CELL:
                    ax.text(j, i, "few" if n > 0 else "none", ha="center", va="center", fontsize=7, color=MUTED)
                else:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                            color="white" if (v - vmin) / (vmax - vmin) > 0.45 else INK)
        ax.set_title(pretty(g))
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("Killed per 100 accidents")
    cb.outline.set_visible(False)
    fig.suptitle(f"Severity by region and condition, India, {FIRST_YEAR} to {LAST_YEAR} combined",
                 x=0.01, y=1.03, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, -0.04, f"Rows are sorted by national severity, regions by overall severity. "
             f"'few' = fewer than {MIN_CELL} accidents in six years, not coloured.", fontsize=7.5, color=INK2)
    save_fig(fig, "fig12_region_condition_heatmap")


# ---- 3. deadliest conditions ------------------------------------------------

def deadliest_conditions(raw: pd.DataFrame, national_severity: float) -> pd.DataFrame:
    """Every factor value in India, ranked by severity over all six years."""
    t = severity_table(raw, ["factor_group", "factor_value"])
    t["years_reported"] = raw[raw["accidents"] > 0].groupby(["factor_group", "factor_value"])["year"].nunique()
    t["vs_national"] = t["severity"] / national_severity
    t = t.sort_values("severity", ascending=False).reset_index()
    t.insert(0, "rank", t.index + 1)
    return t


def deadliest_latex(table: pd.DataFrame, national_severity: float, top: int = 15) -> str:
    t = table.head(top)
    out = pd.DataFrame({
        "Rank": t["rank"],
        "Condition": t["factor_value"].str.replace("&", r"\&", regex=False),
        "Factor group": t["factor_group"].map(pretty),
        "Accidents": t["accidents"].map("{:,}".format).str.replace(",", "{,}", regex=False),
        "Deaths": t["killed"].map("{:,}".format).str.replace(",", "{,}", regex=False),
        "Severity": t["severity"].map("{:.1f}".format),
    })
    return out.to_latex(
        index=False, escape=False, column_format="rllrrr",
        caption=(f"The {top} deadliest conditions in India, {FIRST_YEAR} to {LAST_YEAR} combined. Severity is persons "
                 f"killed per 100 accidents; the national value is {national_severity:.1f}."),
        label="tab:deadliest_conditions", position="t",
    )


def export_for_frontend(table: pd.DataFrame, national_severity: float) -> None:
    """The ranked conditions as JSON, and the two static figures, for the Insights tab."""
    conditions = table[["rank", "factor_group", "factor_value", "accidents", "killed", "severity"]].copy()
    conditions["severity"] = conditions["severity"].round(2)
    payload = {"first_year": FIRST_YEAR, "last_year": LAST_YEAR, "national_severity": round(national_severity, 2),
               "conditions": conditions.to_dict(orient="records")}
    FRONTEND_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    (FRONTEND_GEOJSON.parent / "deadliest_conditions.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    images = PROJECT_ROOT / "frontend" / "img"
    images.mkdir(parents=True, exist_ok=True)
    slug = FOCUS_STATE.lower().replace(" ", "_")
    for name in ("fig12_region_condition_heatmap", f"fig13_{slug}_profile"):
        shutil.copyfile(FIG_DIR / f"{name}.png", images / f"{name}.png")
    print("saved frontend/data/deadliest_conditions.json and copied two figures to frontend/img")


# ---- 4. state profile -------------------------------------------------------

def state_profile(raw: pd.DataFrame, state: str) -> pd.DataFrame:
    """Each factor value in one state against the same factor value nationally, all years combined."""
    mine = severity_table(raw[raw["state_harmonised"] == state], ["factor_group", "factor_value"])
    india = severity_table(raw, ["factor_group", "factor_value"])
    t = mine.join(india, rsuffix="_india")
    t["difference"] = t["severity"] - t["severity_india"]
    t["ratio"] = t["severity"] / t["severity_india"]
    t["share_of_india_accidents"] = t["accidents"] / t["accidents_india"]
    return t.reset_index()


def plot_state_profile(profile: pd.DataFrame, tot: pd.DataFrame, state: str) -> None:
    by_year = pd.DataFrame({
        state: severity_table(tot[tot["state_harmonised"] == state], "year")["severity"],
        "India": severity_table(tot, "year")["severity"],
    })
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), gridspec_kw={"hspace": 0.42, "wspace": 0.5})

    ax = axes[0, 0]
    for name, color in [("India", INK2), (state, BLUE)]:
        ax.plot(by_year.index, by_year[name], color=color, marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=1.5, zorder=3)
        ax.text(by_year.index[-1] + 0.15, by_year[name].iloc[-1], f"{name} {by_year[name].iloc[-1]:.1f}",
                va="center", fontsize=8.5, color=INK)
        ax.text(by_year.index[0], by_year[name].iloc[0] + 0.9, f"{by_year[name].iloc[0]:.1f}", ha="center",
                fontsize=8, color=INK)
    ax.set_xlim(by_year.index[0] - 0.4, by_year.index[-1] + 1.5)
    ax.set_ylim(0, by_year.max().max() * 1.15)
    ax.set_xticks(by_year.index)
    ax.set_ylabel("Killed per 100 accidents")
    ax.set_title("Severity per year")
    ax.grid(axis="y")
    ax.tick_params(length=0)

    xmax = profile.loc[profile["factor_group"].isin(PROFILE_GROUPS), ["severity", "severity_india"]].max().max() * 1.08
    for ax, g in zip(axes.ravel()[1:], PROFILE_GROUPS):
        t = profile[profile["factor_group"] == g].sort_values("severity_india")
        t = t[t["accidents"] >= MIN_CELL]
        y = np.arange(len(t))
        ax.hlines(y, t["severity"], t["severity_india"], color=AXIS, linewidth=2, zorder=1)
        ax.scatter(t["severity_india"], y, s=46, color=INK2, edgecolor="white", linewidth=1.2, zorder=3, label="India")
        ax.scatter(t["severity"], y, s=46, color=BLUE, edgecolor="white", linewidth=1.2, zorder=4, label=state)
        for yi, k, n in zip(y, t["severity"], t["severity_india"]):
            ax.text(min(k, n) - xmax * 0.02, yi, f"{k:.0f}" if k < n else f"{n:.0f}", ha="right", va="center",
                    fontsize=7.5, color=INK)
            ax.text(max(k, n) + xmax * 0.02, yi, f"{n:.0f}" if k < n else f"{k:.0f}", ha="left", va="center",
                    fontsize=7.5, color=INK)
        ax.set_yticks(y, [v.replace("/", "/\n", 1) if len(v) > 24 else v for v in t["factor_value"]])
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.7, len(t) - 0.3)
        ax.set_xlabel("Killed per 100 accidents")
        ax.set_title(pretty(g))
        ax.grid(axis="x")
        ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    axes[0, 1].legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2, columnspacing=1.0)
    fig.suptitle(f"{state} against the national figures, {FIRST_YEAR} to {LAST_YEAR}",
                 x=0.01, y=0.97, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.02, f"Condition panels combine all six years and are sorted by national severity. "
             f"Conditions with fewer than {MIN_CELL} accidents in {state} are left out.", fontsize=7.5, color=INK2)
    save_fig(fig, f"fig13_{state.lower().replace(' ', '_')}_profile")


def profile_headline(tot: pd.DataFrame, states: pd.DataFrame, state: str) -> pd.DataFrame:
    """A few summary numbers for the state: totals, shares and rank."""
    mine, india = tot[tot["state_harmonised"] == state], tot
    ranked = states.loc[~states["small_count"], f"severity_{LAST_YEAR}"].rank(ascending=False)
    rows = [
        ("Accidents, all years", mine["accidents"].sum(), india["accidents"].sum()),
        ("Persons killed, all years", mine["killed"].sum(), india["killed"].sum()),
        ("Severity, all years", 100 * mine["killed"].sum() / mine["accidents"].sum(),
         100 * india["killed"].sum() / india["accidents"].sum()),
        (f"Severity, {FIRST_YEAR}", states.at[state, f"severity_{FIRST_YEAR}"],
         severity_table(india, "year").at[FIRST_YEAR, "severity"]),
        (f"Severity, {LAST_YEAR}", states.at[state, f"severity_{LAST_YEAR}"],
         severity_table(india, "year").at[LAST_YEAR, "severity"]),
        (f"Severity rank in {LAST_YEAR} (1 = deadliest, states with {SMALL}+ accidents)", ranked[state], len(ranked)),
    ]
    return pd.DataFrame(rows, columns=["measure", state, "India"])


# ---- entry point ------------------------------------------------------------

def main() -> int:
    seed_everything()
    set_style()
    pd.set_option("display.width", 250)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data(drop_low_count=False)
    tot = raw[raw["factor_group"] == TOTALS_GROUP]
    national = 100 * tot["killed"].sum() / tot["accidents"].sum()

    boundaries = load_boundaries()
    states = state_severity(tot)
    unmatched = sorted({harmonised_name(f) for f in boundaries["features"]} ^ set(states.index))
    if unmatched:
        raise ValueError(f"State names do not match between the boundaries and the dataset: {unmatched}")
    states.round(2).to_csv(TAB_DIR / "state_severity_by_year.csv")
    plot_choropleth(states, boundaries)
    write_geojson(states, boundaries)
    cols = [f"accidents_{LAST_YEAR}", f"severity_{FIRST_YEAR}", f"severity_{LAST_YEAR}", "severity_change", "small_count"]
    print(states[cols].round(1).to_string())

    region_order = list(severity_table(tot, "region")["severity"].sort_values(ascending=False).index)
    region_table = region_factor_table(raw)
    region_table.round(2).to_csv(TAB_DIR / "severity_by_region_condition.csv")
    plot_region_heatmap(region_table, region_order)

    deadliest = deadliest_conditions(raw, national)
    deadliest.round(3).to_csv(TAB_DIR / "deadliest_conditions.csv", index=False)
    (TAB_DIR / "deadliest_conditions.tex").write_text(deadliest_latex(deadliest, national), encoding="utf-8")
    print(f"\nDeadliest conditions, {FIRST_YEAR} to {LAST_YEAR} combined (national severity {national:.1f})")
    print(deadliest.head(15).round(2).to_string(index=False))

    profile = state_profile(raw, FOCUS_STATE)
    slug = FOCUS_STATE.lower().replace(" ", "_")
    profile.round(3).to_csv(TAB_DIR / f"{slug}_profile.csv", index=False)
    headline = profile_headline(tot, states, FOCUS_STATE)
    headline.round(2).to_csv(TAB_DIR / f"{slug}_headline.csv", index=False)
    plot_state_profile(profile, tot, FOCUS_STATE)
    print(f"\n{FOCUS_STATE} against India")
    print(headline.round(2).to_string(index=False))
    big = profile[profile["accidents"] >= MIN_CELL].sort_values("ratio")
    print(big[["factor_group", "factor_value", "accidents", "killed", "severity", "severity_india", "ratio"]]
          .round(2).to_string(index=False))

    export_for_frontend(deadliest, national)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
