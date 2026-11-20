import math
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

SEASON = 2026
BASE_URL = "https://api.collegefootballdata.com"

st.set_page_config(page_title="Cover 0", page_icon="🏈", layout="wide")
st.title("🏈 Cover 0")
st.caption("Season-aware projections from completed results, opponent strength, recent form, and home field.")


def normalize_games(df: pd.DataFrame, season=SEASON) -> pd.DataFrame:
    aliases = {
        "season": ["season", "year"], "week": ["week"],
        "home_team": ["home_team", "homeTeam", "home"],
        "away_team": ["away_team", "awayTeam", "away"],
        "home_points": ["home_points", "homePoints", "home_score"],
        "away_points": ["away_points", "awayPoints", "away_score"],
        "completed": ["completed", "is_completed"],
        "neutral_site": ["neutral_site", "neutralSite", "neutral"],
    }
    out = pd.DataFrame(index=df.index)
    for target, opts in aliases.items():
        found = next((c for c in opts if c in df.columns), None)
        default = True if target == "completed" else (False if target == "neutral_site" else np.nan)
        out[target] = df[found] if found else default
    out["season"] = pd.to_numeric(out["season"], errors="coerce").fillna(SEASON)
    out["week"] = pd.to_numeric(out["week"], errors="coerce").fillna(0)
    out["home_points"] = pd.to_numeric(out["home_points"], errors="coerce")
    out["away_points"] = pd.to_numeric(out["away_points"], errors="coerce")
    out["home_team"] = out["home_team"].astype("string").str.strip()
    out["away_team"] = out["away_team"].astype("string").str.strip()
    if out["completed"].dtype != bool:
        out["completed"] = out["completed"].astype(str).str.lower().isin(["true", "1", "yes", "completed"])
    if out["neutral_site"].dtype != bool:
        out["neutral_site"] = out["neutral_site"].astype(str).str.lower().isin(["true", "1", "yes"])
    if season is not None:
        out = out[out.season == season]
    out = out[out.completed & out.home_points.notna() & out.away_points.notna()]
    out = out[(out.home_team.notna()) & (out.away_team.notna())]
    out["home_margin"] = out.home_points - out.away_points
    return out.reset_index(drop=True)


def fetch_games():
    key = os.getenv("CFBD_API_KEY", "").strip()
    # Streamlit Community Cloud secrets are exposed through st.secrets.
    # Keep the environment variable fallback for local and other-host deployments.
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        return None, "CFBD_API_KEY is not set. Upload a season CSV or configure a key to fetch results."
    try:
        teams_response = requests.get(
            f"{BASE_URL}/teams/fbs", params={"year": SEASON},
            headers={"Authorization": f"Bearer {key}"}, timeout=25,
        )
        teams_response.raise_for_status()
        fbs_teams = {str(team["school"]).strip() for team in teams_response.json() if team.get("school")}
        response = requests.get(
            f"{BASE_URL}/games",
            params={"year": SEASON, "seasonType": "regular", "classification": "fbs"},
            headers={"Authorization": f"Bearer {key}"}, timeout=25,
        )
        response.raise_for_status()
        games = normalize_games(pd.DataFrame(response.json()))
        # Keep only FBS-vs-FBS games for this FBS-only model. This excludes lower-division
        # schedules and opponents from the ratings instead of silently mixing subdivisions.
        games = games[games.home_team.isin(fbs_teams) & games.away_team.isin(fbs_teams)].reset_index(drop=True)
        return games, None
    except Exception as exc:
        return None, f"Could not fetch and verify FBS season data: {exc}"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_fbs_teams(year=SEASON):
    key = os.getenv("CFBD_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        return None
    response = requests.get(
        f"{BASE_URL}/teams/fbs", params={"year": year},
        headers={"Authorization": f"Bearer {key}"}, timeout=25,
    )
    response.raise_for_status()
    return {str(team["school"]).strip() for team in response.json() if team.get("school")}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_upcoming_games(year=SEASON):
    """Fetch future, uncompleted regular-season FBS-vs-FBS matchups from CFBD."""
    key = os.getenv("CFBD_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        return None, "CFBD_API_KEY is not set in this app's Secrets."
    try:
        response = requests.get(
            f"{BASE_URL}/games",
            params={"year": year, "seasonType": "regular", "classification": "fbs"},
            headers={"Authorization": f"Bearer {key}"},
            timeout=25,
        )
        response.raise_for_status()
        raw = pd.DataFrame(response.json())
        if raw.empty:
            return pd.DataFrame(), None
        def column(names, default=None):
            name = next((candidate for candidate in names if candidate in raw.columns), None)
            return raw[name] if name else pd.Series(default, index=raw.index)

        games = pd.DataFrame({
            "season": pd.to_numeric(column(["season", "year"], year), errors="coerce"),
            "week": pd.to_numeric(column(["week"], 0), errors="coerce").fillna(0),
            "season_type": column(["seasonType", "season_type"], "regular").astype(str).str.lower(),
            "kickoff": pd.to_datetime(column(["startDate", "start_date"]), errors="coerce", utc=True),
            "completed": column(["completed", "is_completed"], False),
            "neutral_site": column(["neutralSite", "neutral_site"], False),
            "home_team": column(["homeTeam", "home_team", "home"], "").astype("string").str.strip(),
            "away_team": column(["awayTeam", "away_team", "away"], "").astype("string").str.strip(),
            "home_classification": column(["homeClassification", "home_classification"], "").astype(str).str.lower(),
            "away_classification": column(["awayClassification", "away_classification"], "").astype(str).str.lower(),
        })
        if games["completed"].dtype != bool:
            games["completed"] = games["completed"].astype(str).str.lower().isin(["true", "1", "yes", "completed"])
        if games["neutral_site"].dtype != bool:
            games["neutral_site"] = games["neutral_site"].astype(str).str.lower().isin(["true", "1", "yes"])
        now = pd.Timestamp.now(tz="UTC")
        future = games[
            (games.season == year)
            & games.season_type.eq("regular")
            & ~games.completed
            & games.kickoff.notna()
            & (games.kickoff >= now)
            & games.home_team.notna()
            & games.away_team.notna()
            & games.home_team.ne("")
            & games.away_team.ne("")
        ].copy()
        # CFBD normally includes classifications on schedule rows; use the FBS roster
        # as a fallback when those fields are blank so FCS opponents never enter here.
        has_classifications = games.home_classification.eq("fbs").any() and games.away_classification.eq("fbs").any()
        if has_classifications:
            future = future[
                future.home_classification.eq("fbs") & future.away_classification.eq("fbs")
            ]
        else:
            fbs_teams = fetch_fbs_teams(year)
            if fbs_teams is None:
                return None, "Could not verify the FBS team roster for scheduled games."
            future = future[future.home_team.isin(fbs_teams) & future.away_team.isin(fbs_teams)]
        future["kickoff_utc"] = future.kickoff.dt.strftime("%b %d, %Y %I:%M %p UTC")
        return future[["season", "week", "kickoff", "kickoff_utc", "home_team", "away_team", "neutral_site"]].sort_values(
            ["kickoff", "home_team"]
        ).reset_index(drop=True), None
    except Exception as exc:
        return None, f"Could not fetch upcoming FBS games: {exc}"


def project_upcoming_games(upcoming, season_r, recent_r, home_field, form_weight, margin_sd):
    """Project scheduled matchups using only the currently loaded completed results."""
    if upcoming is None or upcoming.empty:
        return pd.DataFrame()
    rows = []
    for game in upcoming.itertuples():
        home = game.home_team
        away = game.away_team
        home_rating = (1 - form_weight) * season_r.get(home, 0.0) + form_weight * recent_r.get(home, 0.0)
        away_rating = (1 - form_weight) * season_r.get(away, 0.0) + form_weight * recent_r.get(away, 0.0)
        home_margin = home_rating - away_rating + (0.0 if game.neutral_site else home_field)
        home_probability = 1 / (1 + math.exp(-1.702 * home_margin / margin_sd))
        rows.append({
            "season": int(game.season), "week": int(game.week),
            "kickoff_utc": game.kickoff.isoformat(),
            "away_team": away, "home_team": home,
            "neutral_site": bool(game.neutral_site),
            "projected_spread": f"{home} {-home_margin:+.1f}",
            "predicted_home_margin": home_margin,
            "home_win_probability": home_probability,
            "away_win_probability": 1 - home_probability,
        })
    return pd.DataFrame(rows)


def grade_saved_forecasts(uploaded_files, completed_games):
    """Match saved pregame forecast CSVs against completed scores in the loaded season."""
    snapshots = []
    for uploaded_file in uploaded_files:
        frame = pd.read_csv(uploaded_file)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        # Legacy downloads had display labels, including "Kickoff (UTC)" and "Home team".
        legacy = {
            "kickoff_utc": "kickoff (utc)", "away_team": "away team", "home_team": "home team",
            "projected_spread": "projected spread", "home_win_probability": "home win probability",
            "away_win_probability": "away win probability",
        }
        frame = frame.rename(columns={old: new for new, old in legacy.items() if new not in frame.columns and old in frame.columns})
        if not {"home_team", "away_team", "home_win_probability"}.issubset(frame.columns):
            continue
        if "season" not in frame:
            frame["season"] = SEASON
        if "week" not in frame:
            frame["week"] = 0
        frame["season"] = pd.to_numeric(frame.season, errors="coerce").fillna(SEASON).astype(int)
        frame["week"] = pd.to_numeric(frame.week, errors="coerce").fillna(0).astype(int)
        frame["home_team"] = frame.home_team.astype("string").str.strip()
        frame["away_team"] = frame.away_team.astype("string").str.strip()
        if "kickoff_utc" in frame:
            frame["kickoff_utc"] = pd.to_datetime(frame.kickoff_utc, errors="coerce", utc=True)
        else:
            frame["kickoff_utc"] = pd.NaT
        if "forecast_generated_utc" in frame:
            frame["forecast_generated_utc"] = pd.to_datetime(frame.forecast_generated_utc, errors="coerce", utc=True)
        else:
            frame["forecast_generated_utc"] = pd.NaT
        for probability_col in ("home_win_probability", "away_win_probability"):
            frame[probability_col] = pd.to_numeric(
                frame[probability_col].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )
            if frame[probability_col].dropna().gt(1).any():
                frame[probability_col] = frame[probability_col] / 100
        if "predicted_home_margin" not in frame:
            if "projected_spread" not in frame:
                continue
            spread_values = pd.to_numeric(
                frame.projected_spread.astype(str).str.extract(r"([+-]?\d+(?:\.\d+)?)\s*$")[0], errors="coerce"
            )
            starts_with_home = frame.apply(
                lambda row: str(row.projected_spread).strip().lower().startswith(str(row.home_team).lower()), axis=1
            )
            frame["predicted_home_margin"] = np.where(starts_with_home, -spread_values, np.nan)
        frame["predicted_home_margin"] = pd.to_numeric(frame.predicted_home_margin, errors="coerce")
        snapshots.append(frame)
    if not snapshots:
        return pd.DataFrame(), "Upload a Cover 0 upcoming-predictions CSV with teams and win probabilities.", 0

    forecasts = pd.concat(snapshots, ignore_index=True)
    before_kickoff = forecasts.forecast_generated_utc.isna() | forecasts.kickoff_utc.isna() | (
        forecasts.forecast_generated_utc < forecasts.kickoff_utc
    )
    excluded_late = int((~before_kickoff).sum())
    forecasts = forecasts[before_kickoff].copy()
    keys = ["season", "week", "home_team", "away_team"]
    forecasts = forecasts.sort_values("forecast_generated_utc").drop_duplicates(keys, keep="last")
    actuals = completed_games[keys + ["home_margin", "home_points", "away_points"]].copy()
    actuals["actual_home_win"] = (actuals.home_margin > 0).astype(int)
    scored = forecasts.merge(actuals, on=keys, how="inner")
    if scored.empty:
        return scored, None, excluded_late
    scored["actual_home_margin"] = scored.home_margin
    scored["absolute_margin_error"] = (scored.predicted_home_margin - scored.home_margin).abs()
    scored["correct_winner"] = ((scored.home_win_probability >= 0.5) == (scored.actual_home_win == 1)).astype(int)
    scored["brier_component"] = (scored.home_win_probability - scored.actual_home_win) ** 2
    return scored, None, excluded_late


def restrict_to_fbs_matchups(games, fbs_teams):
    if fbs_teams is None:
        return games
    return games[games.home_team.isin(fbs_teams) & games.away_team.isin(fbs_teams)].reset_index(drop=True)


def fetch_historical_games(start_year, end_year):
    key = os.getenv("CFBD_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        return None, "CFBD_API_KEY is not set in this app's Secrets."
    frames = []
    try:
        for year in range(start_year, end_year + 1):
            teams_response = requests.get(
                f"{BASE_URL}/teams/fbs", params={"year": year},
                headers={"Authorization": f"Bearer {key}"}, timeout=25,
            )
            teams_response.raise_for_status()
            fbs_teams = {str(team["school"]).strip() for team in teams_response.json() if team.get("school")}
            response = requests.get(
                f"{BASE_URL}/games",
                params={"year": year, "seasonType": "regular", "classification": "fbs"},
                headers={"Authorization": f"Bearer {key}"}, timeout=25,
            )
            response.raise_for_status()
            frames.append(restrict_to_fbs_matchups(normalize_games(pd.DataFrame(response.json()), season=year), fbs_teams))
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return raw, None
    except Exception as exc:
        return None, f"Could not load historical games: {exc}"


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_advanced_game_stats(year):
    """Fetch per-game PPA, success-rate, and explosiveness inputs for one season."""
    key = os.getenv("CFBD_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set in this app's Secrets.")
    response = requests.get(
        f"{BASE_URL}/stats/game/advanced",
        params={"year": year, "seasonType": "regular", "excludeGarbageTime": True},
        headers={"Authorization": f"Bearer {key}"},
        timeout=40,
    )
    response.raise_for_status()
    return response.json()


def fetch_historical_advanced_games(start_year, end_year, historical_games):
    """Load advanced game stats, restricted to the FBS-vs-FBS games in the backtest."""
    records = []
    teams_by_year = {}
    for year, games_for_year in historical_games.groupby("season"):
        teams_by_year[int(year)] = set(games_for_year.home_team) | set(games_for_year.away_team)
    try:
        for year in range(start_year, end_year + 1):
            for game in fetch_advanced_game_stats(year):
                if str(game.get("seasonType", "regular")).lower() != "regular":
                    continue
                team = str(game.get("team") or "").strip()
                opponent = str(game.get("opponent") or "").strip()
                if team not in teams_by_year.get(year, set()) or opponent not in teams_by_year.get(year, set()):
                    continue
                offense = game.get("offense") or {}
                defense = game.get("defense") or {}
                row = {
                    "season": int(game.get("season", year)),
                    "week": int(game.get("week", 0) or 0),
                    "team": team,
                    "opponent": opponent,
                }
                for side, stats in (("off", offense), ("def", defense)):
                    for api_name, col in (("ppa", "ppa"), ("successRate", "success_rate"), ("explosiveness", "explosiveness")):
                        row[f"{side}_{col}"] = pd.to_numeric(stats.get(api_name), errors="coerce")
                records.append(row)
        advanced = pd.DataFrame(records)
        if advanced.empty:
            return None, "CollegeFootballData returned no usable advanced FBS game stats for those seasons."
        return advanced, None
    except Exception as exc:
        return None, f"Could not load advanced game stats: {exc}"
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_team_box_scores(year, week):
    """Fetch weekly FBS team box scores from CollegeFootballData."""
    key = os.getenv("CFBD_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        raise RuntimeError("CFBD_API_KEY is not set in this app's Secrets.")
    response = requests.get(
        f"{BASE_URL}/games/teams",
        params={"year": year, "week": week, "seasonType": "regular", "classification": "fbs"},
        headers={"Authorization": f"Bearer {key}"}, timeout=40,
    )
    response.raise_for_status()
    return response.json()


def fetch_historical_turnovers(start_year, end_year, historical_games):
    """Load turnovers for historical FBS games in the selected seasons."""
    rows = []
    allowed = {(int(g.season), int(g.week), str(g.home_team), str(g.away_team)) for g in historical_games.itertuples()}
    try:
        for year, season_games in historical_games.groupby("season"):
            for week in sorted(season_games.week.unique()):
                for game in fetch_team_box_scores(int(year), int(week)):
                    sides = {}
                    for team in game.get("teams") or []:
                        side = str(team.get("homeAway") or "").lower()
                        name = str(team.get("team") or "").strip()
                        if side not in {"home", "away"} or not name:
                            continue
                        turnovers = np.nan
                        for stat in team.get("stats") or []:
                            category = re.sub(r"[^a-z]", "", str(stat.get("category", "")).lower())
                            if category in {"turnover", "turnovers", "turnoverslost"}:
                                turnovers = pd.to_numeric(stat.get("stat"), errors="coerce")
                                break
                        sides[side] = (name, turnovers)
                    if "home" not in sides or "away" not in sides:
                        continue
                    home, home_to = sides["home"]
                    away, away_to = sides["away"]
                    if (int(year), int(week), home, away) not in allowed or pd.isna(home_to) or pd.isna(away_to):
                        continue
                    rows.append({"season": int(year), "week": int(week), "home_team": home, "away_team": away, "home_turnover_margin": float(away_to - home_to)})
        data = pd.DataFrame(rows)
        if data.empty:
            return None, "CFBD returned no usable turnover box scores for these seasons."
        return data, None
    except Exception as exc:
        return None, f"Could not load historical turnovers: {exc}"





@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_closing_lines(start_year, end_year):
    """Fetch CFBD closing spreads and convert each to an implied home margin."""
    key = os.getenv("CFBD_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("CFBD_API_KEY", "")).strip()
        except Exception:
            key = ""
    if not key:
        return None, "CFBD_API_KEY is not set in this app's Secrets."
    headers = {"Authorization": f"Bearer {key}"}
    records = []
    try:
        for year in range(start_year, end_year + 1):
            response = requests.get(
                f"{BASE_URL}/lines",
                params={"year": year, "seasonType": "regular"},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            for game in response.json():
                # CFBD supplies one game with a nested list of provider lines.
                lines = game.get("lines") or []
                if isinstance(lines, dict):
                    lines = [lines]
                usable = []
                for line in lines:
                    spread = pd.to_numeric(line.get("spread"), errors="coerce")
                    if pd.isna(spread):
                        continue
                    formatted = str(line.get("formattedSpread") or "")
                    home = str(game.get("homeTeam") or "")
                    away = str(game.get("awayTeam") or "")
                    # formattedSpread names the favorite (e.g. "Arizona -10.5").
                    # Convert that favorite's points into an implied home-team margin.
                    if home and home.lower() in formatted.lower():
                        home_line = abs(float(spread))
                    elif away and away.lower() in formatted.lower():
                        home_line = -abs(float(spread))
                    else:
                        # Without a formatted team, fall back to CFBD's raw home-relative sign.
                        home_line = -float(spread)
                    usable.append((str(line.get("provider") or ""), home_line))
                if not usable:
                    continue
                consensus = [value for provider, value in usable if provider.strip().lower() == "consensus"]
                # Prefer CFBD's consensus line; when absent, use the median provider line.
                home_line = float(np.median(consensus if consensus else [value for _, value in usable]))
                records.append({
                    "season": int(game.get("season", year)),
                    "week": int(game.get("week", 0) or 0),
                    "home_team": str(game.get("homeTeam") or "").strip(),
                    "away_team": str(game.get("awayTeam") or "").strip(),
                    "market_home_margin": home_line,
                })
        lines = pd.DataFrame(records)
        if not lines.empty:
            lines = lines.drop_duplicates(["season", "week", "home_team", "away_team"])
        return lines, None
    except Exception as exc:
        return None, f"Could not load historical closing lines: {exc}"


def compare_to_market(forecasts: pd.DataFrame, closing_lines: pd.DataFrame):
    """Compare model and closing-line margin errors and spread-cover picks."""
    keys = ["season", "week", "home_team", "away_team"]
    joined = forecasts.merge(closing_lines, on=keys, how="inner")
    if joined.empty:
        return joined
    joined["model_margin_error"] = (joined.predicted_home_margin - joined.actual_home_margin).abs()
    joined["market_margin_error"] = (joined.market_home_margin - joined.actual_home_margin).abs()
    joined["model_cover_result"] = np.sign(joined.actual_home_margin - joined.market_home_margin) * np.sign(
        joined.predicted_home_margin - joined.market_home_margin
    )
    # result is +1 for a cover, 0 for a push, and -1 for a loss.
    joined["market_cover_result"] = np.sign(joined.actual_home_margin - joined.market_home_margin)
    joined["model_market_gap"] = (joined.predicted_home_margin - joined.market_home_margin).abs()
    joined["model_market_gap_band"] = pd.cut(
        joined.model_market_gap,
        bins=[0, 1, 3, 5, np.inf],
        labels=["Under 1 pt", "1–<3 pts", "3–<5 pts", "5+ pts"],
        right=False,
    )
    return joined


def cover_record(results):
    decided = results[results != 0]
    wins = int((decided > 0).sum())
    losses = int((decided < 0).sum())
    pushes = int((results == 0).sum())
    pct = wins / len(decided) if len(decided) else float("nan")
    return wins, losses, pushes, pct


def calculate_ratings(games: pd.DataFrame, home_field=2.0):
    teams = sorted(set(games.home_team) | set(games.away_team))
    if not teams:
        return {}, {}, {}
    idx = {team: i for i, team in enumerate(teams)}
    # Solve team strength differences against results. Ridge shrinkage stabilizes small samples.
    X = np.zeros((len(games), len(teams)), dtype=float)
    # Convert observed home margins to neutral-field margins before fitting team ratings.
    # Neutral-site games receive no adjustment.
    # Older Streamlit sessions can retain a pre-neutral-site DataFrame after deploy.
    neutral_values = games["neutral_site"] if "neutral_site" in games.columns else pd.Series(False, index=games.index)
    neutral = neutral_values.fillna(False).to_numpy(dtype=bool)
    y = games.home_margin.to_numpy(dtype=float) - np.where(neutral, 0.0, home_field)
    weeks = games.week.to_numpy(dtype=float)
    max_week = max(weeks.max(initial=0), 1)
    for row, game in enumerate(games.itertuples()):
        X[row, idx[game.home_team]] = 1.0
        X[row, idx[game.away_team]] = -1.0
    # More recent results count more, with a floor so early form doesn't disappear.
    weights = np.maximum(0.35, np.exp(-0.09 * (max_week - weeks)))
    lhs = X.T @ (weights[:, None] * X) + np.eye(len(teams)) * 1.8
    rhs = X.T @ (weights * y)
    raw = np.linalg.solve(lhs, rhs)
    raw -= raw.mean()
    season = dict(zip(teams, raw))

    # Recent performance: recency-weighted raw margin with simple opponent-strength adjustment.
    # Aggregate both teams in one pass instead of rescanning every game for
    # every team. This is algebraically identical and keeps multi-season
    # walk-forward tuning responsive.
    recent_numerator = {team: 0.0 for team in teams}
    recent_denominator = {team: 0.0 for team in teams}
    counts = {team: 0 for team in teams}
    for g in games.itertuples():
        neutral_game = bool(getattr(g, "neutral_site", False))
        home_margin = g.home_margin - (0.0 if neutral_game else home_field)
        weight = max(0.45, math.exp(-0.18 * (max_week - g.week)))
        recent_numerator[g.home_team] += weight * (home_margin + 0.35 * season[g.away_team])
        recent_denominator[g.home_team] += weight
        recent_numerator[g.away_team] += weight * (-home_margin + 0.35 * season[g.home_team])
        recent_denominator[g.away_team] += weight
        counts[g.home_team] += 1
        counts[g.away_team] += 1
    recent = {
        team: recent_numerator[team] / recent_denominator[team] if recent_denominator[team] else 0.0
        for team in teams
    }
    return season, recent, counts


def run_walk_forward_backtest(all_games: pd.DataFrame, seasons, home_field=2.0, form_weight=0.25, margin_sd=14.0):
    """Predict each week's games using only completed results from earlier weeks."""
    forecasts = []
    for year in seasons:
        season_games = all_games[all_games.season == year]
        for week in sorted(season_games.week.unique()):
            prior = season_games[season_games.week < week]
            target = season_games[season_games.week == week]
            season_r, recent_r, _ = calculate_ratings(prior, home_field=home_field)
            for game in target.itertuples():
                home_rating = (1 - form_weight) * season_r.get(game.home_team, 0.0) + form_weight * recent_r.get(game.home_team, 0.0)
                away_rating = (1 - form_weight) * season_r.get(game.away_team, 0.0) + form_weight * recent_r.get(game.away_team, 0.0)
                predicted_margin = home_rating - away_rating + (0.0 if bool(game.neutral_site) else home_field)
                p_home = 1 / (1 + math.exp(-1.702 * predicted_margin / margin_sd))
                actual_home_win = int(game.home_points > game.away_points)
                actual_margin = float(game.home_points - game.away_points)
                forecasts.append({
                    "season": int(year), "week": int(week),
                    "home_team": game.home_team, "away_team": game.away_team,
                    "predicted_home_margin": predicted_margin,
                    "actual_home_margin": actual_margin,
                    "home_win_probability": p_home,
                    "actual_home_win": actual_home_win,
                    "absolute_margin_error": abs(predicted_margin - actual_margin),
                    "squared_margin_error": (predicted_margin - actual_margin) ** 2,
                    "brier_component": (p_home - actual_home_win) ** 2,
                    "correct_winner": int((p_home >= 0.5) == bool(actual_home_win)),
                })
    return pd.DataFrame(forecasts)


def run_walk_forward_grid(all_games: pd.DataFrame, seasons, home_fields, form_weights, margin_sd=14.0):
    """Evaluate a parameter grid while fitting ratings only once per home-field setting/week."""
    forecasts = {(home_field, form_weight): [] for home_field in home_fields for form_weight in form_weights}
    for home_field in home_fields:
        for year in seasons:
            season_games = all_games[all_games.season == year]
            for week in sorted(season_games.week.unique()):
                prior = season_games[season_games.week < week]
                target = season_games[season_games.week == week]
                season_r, recent_r, _ = calculate_ratings(prior, home_field=home_field)
                for game in target.itertuples():
                    neutral_site = bool(getattr(game, "neutral_site", False))
                    actual_home_win = int(game.home_points > game.away_points)
                    actual_margin = float(game.home_points - game.away_points)
                    for form_weight in form_weights:
                        home_rating = (1 - form_weight) * season_r.get(game.home_team, 0.0) + form_weight * recent_r.get(game.home_team, 0.0)
                        away_rating = (1 - form_weight) * season_r.get(game.away_team, 0.0) + form_weight * recent_r.get(game.away_team, 0.0)
                        predicted_margin = home_rating - away_rating + (0.0 if neutral_site else home_field)
                        p_home = 1 / (1 + math.exp(-1.702 * predicted_margin / margin_sd))
                        forecasts[(home_field, form_weight)].append({
                            "season": int(year), "week": int(week),
                            "home_team": game.home_team, "away_team": game.away_team,
                            "predicted_home_margin": predicted_margin,
                            "actual_home_margin": actual_margin,
                            "home_win_probability": p_home,
                            "actual_home_win": actual_home_win,
                            "absolute_margin_error": abs(predicted_margin - actual_margin),
                            "squared_margin_error": (predicted_margin - actual_margin) ** 2,
                            "brier_component": (p_home - actual_home_win) ** 2,
                            "correct_winner": int((p_home >= 0.5) == bool(actual_home_win)),
                        })
    return {setting: pd.DataFrame(rows) for setting, rows in forecasts.items()}


def run_holdout_tuning(grid_forecasts, tuning_seasons, holdout_seasons):
    """Select settings on earlier seasons and evaluate them once on later seasons."""
    candidates = []
    for home_field in (1.5, 2.0, 2.5):
        for form_weight in (0.15, 0.30):
            tuning_forecasts = grid_forecasts[(home_field, form_weight)]
            tuning_forecasts = tuning_forecasts[tuning_forecasts.season.isin(tuning_seasons)]
            tuning_mae = float(summarize_forecasts(tuning_forecasts).iloc[0].Spread_MAE)
            candidates.append({
                "Home_Field": home_field,
                "Recent_Form_Weight": form_weight,
                "Tuning_Spread_MAE": tuning_mae,
            })
    candidates = pd.DataFrame(candidates).sort_values("Tuning_Spread_MAE").reset_index(drop=True)
    best = candidates.iloc[0]
    baseline = grid_forecasts[(2.0, 0.25)]
    baseline = baseline[baseline.season.isin(holdout_seasons)].copy()
    tuned = grid_forecasts[(float(best.Home_Field), float(best.Recent_Form_Weight))]
    tuned = tuned[tuned.season.isin(holdout_seasons)].copy()
    return candidates, best, baseline, tuned


def attach_efficiency_signal(forecasts, advanced_games):
    """Attach a pregame efficiency differential using only earlier weeks."""
    metric_cols = ["off_ppa", "def_ppa", "off_success_rate", "def_success_rate", "off_explosiveness", "def_explosiveness"]
    result = forecasts.copy()
    result["efficiency_signal"] = 0.0
    for year in sorted(result.season.unique()):
        year_advanced = advanced_games[advanced_games.season == year]
        year_rows = result[result.season == year]
        for week in sorted(year_rows.week.unique()):
            prior = year_advanced[year_advanced.week < week]
            if prior.empty:
                continue
            team_stats = prior.groupby("team")[metric_cols].mean()
            z = pd.DataFrame(index=team_stats.index)
            for col in metric_cols:
                spread = team_stats[col].std(ddof=0)
                if pd.isna(spread) or spread <= 1e-9:
                    z[col] = 0.0
                else:
                    z[col] = (team_stats[col] - team_stats[col].mean()) / spread
            # Higher offensive efficiency and lower efficiency allowed are favorable.
            team_stats["efficiency_strength"] = (
                z.off_ppa - z.def_ppa
                + z.off_success_rate - z.def_success_rate
                + z.off_explosiveness - z.def_explosiveness
            ) / 3.0
            team_strength = team_stats.efficiency_strength.to_dict()
            target_idx = year_rows.index[year_rows.week == week]
            result.loc[target_idx, "efficiency_signal"] = [
                team_strength.get(home, 0.0) - team_strength.get(away, 0.0)
                for home, away in zip(result.loc[target_idx, "home_team"], result.loc[target_idx, "away_team"])
            ]
    return result


def tune_efficiency_overlay(forecasts, tuning_seasons, holdout_seasons, margin_sd=14.0):
    """Tune one efficiency-overlay weight on earlier seasons and report later holdout."""
    choices = np.arange(0.0, 8.01, 0.25)
    tuning_rows = forecasts[forecasts.season.isin(tuning_seasons)]
    scored_tuning = []
    for weight in choices:
        predicted = tuning_rows.predicted_home_margin + weight * tuning_rows.efficiency_signal
        scored_tuning.append({"weight": weight, "mae": float((predicted - tuning_rows.actual_home_margin).abs().mean())})
    best = min(scored_tuning, key=lambda row: row["mae"])

    def score(seasons, weight):
        scored = forecasts[forecasts.season.isin(seasons)].copy()
        scored["predicted_home_margin"] = scored.predicted_home_margin + weight * scored.efficiency_signal
        scored["absolute_margin_error"] = (scored.predicted_home_margin - scored.actual_home_margin).abs()
        scored["squared_margin_error"] = (scored.predicted_home_margin - scored.actual_home_margin) ** 2
        scored["home_win_probability"] = 1 / (1 + np.exp(-1.702 * scored.predicted_home_margin / margin_sd))
        scored["brier_component"] = (scored.home_win_probability - scored.actual_home_win) ** 2
        scored["correct_winner"] = ((scored.home_win_probability >= 0.5) == (scored.actual_home_win == 1)).astype(int)
        return summarize_forecasts(scored).iloc[0]

    baseline = score(holdout_seasons, 0.0)
    efficiency = score(holdout_seasons, best["weight"])
    return best, baseline, efficiency


def attach_turnover_signal(forecasts, turnover_games):
    """Use only turnover results from weeks before each forecast."""
    result = forecasts.copy()
    result["turnover_signal"] = 0.0
    for year in sorted(result.season.unique()):
        year_turnovers = turnover_games[turnover_games.season == year]
        year_rows = result[result.season == year]
        for week in sorted(year_rows.week.unique()):
            prior = year_turnovers[year_turnovers.week < week]
            if prior.empty:
                continue
            team_rows = pd.concat([
                prior[["home_team", "home_turnover_margin"]].rename(columns={"home_team": "team", "home_turnover_margin": "edge"}),
                prior[["away_team", "home_turnover_margin"]].assign(edge=lambda d: -d.home_turnover_margin).rename(columns={"away_team": "team"})[["team", "edge"]],
            ], ignore_index=True)
            strengths = team_rows.groupby("team").edge.mean()
            scale = strengths.std(ddof=0)
            if pd.isna(scale) or scale <= 1e-9:
                strengths = strengths * 0.0
            else:
                strengths = (strengths - strengths.mean()) / scale
            idx = year_rows.index[year_rows.week == week]
            result.loc[idx, "turnover_signal"] = [strengths.get(home, 0.0) - strengths.get(away, 0.0) for home, away in zip(result.loc[idx, "home_team"], result.loc[idx, "away_team"])]
    return result


def tune_turnover_overlay(forecasts, tuning_seasons, holdout_seasons, margin_sd=14.0):
    """Tune on 2018–2022 and evaluate once on later seasons."""
    training = forecasts[forecasts.season.isin(tuning_seasons)]
    best = min(({"weight": float(w), "mae": float((training.predicted_home_margin + w * training.turnover_signal - training.actual_home_margin).abs().mean())} for w in np.arange(0.0, 2.01, 0.05)), key=lambda row: row["mae"])
    def score(seasons, weight):
        data = forecasts[forecasts.season.isin(seasons)].copy()
        data["predicted_home_margin"] += weight * data.turnover_signal
        data["absolute_margin_error"] = (data.predicted_home_margin - data.actual_home_margin).abs()
        data["squared_margin_error"] = (data.predicted_home_margin - data.actual_home_margin) ** 2
        data["home_win_probability"] = 1 / (1 + np.exp(-1.702 * data.predicted_home_margin / margin_sd))
        data["brier_component"] = (data.home_win_probability - data.actual_home_win) ** 2
        data["correct_winner"] = ((data.home_win_probability >= 0.5) == (data.actual_home_win == 1)).astype(int)
        return summarize_forecasts(data).iloc[0]
    return best, score(holdout_seasons, 0.0), score(holdout_seasons, best["weight"])





def summarize_forecasts(forecasts: pd.DataFrame, group=None):
    by = group or []
    if by:
        result = forecasts.groupby(by, as_index=False).agg(
            Games=("actual_home_win", "size"),
            Spread_MAE=("absolute_margin_error", "mean"),
            Margin_RMSE=("squared_margin_error", lambda x: float(np.sqrt(x.mean()))),
            Winner_Accuracy=("correct_winner", "mean"),
            Brier_Score=("brier_component", "mean"),
            Avg_Home_Win_Prob=("home_win_probability", "mean"),
            Actual_Home_Win_Rate=("actual_home_win", "mean"),
        )
    else:
        result = pd.DataFrame([{
            "Games": len(forecasts),
            "Spread_MAE": forecasts.absolute_margin_error.mean(),
            "Margin_RMSE": np.sqrt(forecasts.squared_margin_error.mean()),
            "Winner_Accuracy": forecasts.correct_winner.mean(),
            "Brier_Score": forecasts.brier_component.mean(),
            "Avg_Home_Win_Prob": forecasts.home_win_probability.mean(),
            "Actual_Home_Win_Rate": forecasts.actual_home_win.mean(),
        }])
    return result


with st.sidebar:
    st.header("Season data")
    if "games" not in st.session_state:
        fetched, fetch_error = fetch_games()
        st.session_state.games = fetched if fetched is not None else pd.DataFrame()
        st.session_state.data_note = fetch_error
        if fetched is not None:
            st.session_state.data_loaded_at = datetime.now(timezone.utc)
            st.session_state.data_source = "CollegeFootballData API · FBS-vs-FBS games"
    if "upcoming_games" not in st.session_state:
        upcoming, upcoming_error = fetch_upcoming_games()
        st.session_state.upcoming_games = upcoming if upcoming is not None else pd.DataFrame()
        st.session_state.upcoming_error = upcoming_error
        if upcoming is not None:
            st.session_state.upcoming_loaded_at = datetime.now(timezone.utc)
    if st.button("Refresh 2026 results"):
        fetched, fetch_error = fetch_games()
        if fetched is not None:
            st.session_state.games = fetched
            st.session_state.data_note = None
            st.session_state.data_loaded_at = datetime.now(timezone.utc)
            st.session_state.data_source = "CollegeFootballData API · FBS-vs-FBS games"
            st.success(f"Loaded {len(fetched)} completed games.")
            upcoming, upcoming_error = fetch_upcoming_games()
            st.session_state.upcoming_games = upcoming if upcoming is not None else pd.DataFrame()
            st.session_state.upcoming_error = upcoming_error
            if upcoming is not None:
                st.session_state.upcoming_loaded_at = datetime.now(timezone.utc)
                st.info(f"Found {len(upcoming)} upcoming FBS games.")
            else:
                st.warning(upcoming_error)
        else:
            st.session_state.data_note = fetch_error
            st.error(fetch_error)
    upload = st.file_uploader("Or upload season CSV", type=["csv"])
    if upload is not None:
        try:
            uploaded = normalize_games(pd.read_csv(upload))
            fbs_teams = fetch_fbs_teams()
            if fbs_teams is not None:
                uploaded = restrict_to_fbs_matchups(uploaded, fbs_teams)
                st.session_state.data_source = "Uploaded CSV · filtered to FBS-vs-FBS games using CFBD season roster"
            else:
                st.session_state.data_source = "Uploaded CSV · FBS scope not verified"
            st.session_state.games = uploaded
            st.session_state.data_note = None
            st.session_state.data_loaded_at = datetime.now(timezone.utc)
            st.success(f"Loaded {len(uploaded)} completed 2026 games.")
        except Exception as exc:
            st.error(f"CSV could not be read: {exc}")
    st.caption("API access uses CollegeFootballData. Set CFBD_API_KEY in your environment to fetch automatically.")

games = st.session_state.games
if games.empty:
    st.info(st.session_state.get("data_note") or "No completed 2026 results loaded yet. You can upload a CSV or add CFBD_API_KEY and refresh.")
    st.markdown("CSV fields: `season, week, home_team, away_team, home_points, away_points` and optional `completed`.")
    st.stop()

teams = sorted(set(games.home_team) | set(games.away_team))
latest_week = int(games.week.max())
loaded_at = st.session_state.get("data_loaded_at")
loaded_label = loaded_at.strftime("%b %d, %Y %H:%M UTC") if loaded_at else "unknown"
st.caption(f"Using {len(games)} completed games through week {latest_week}. Data loaded {loaded_label}. Source: {st.session_state.get('data_source', 'unknown')}.")
if st.session_state.get("data_source", "").endswith("FBS scope not verified"):
    st.warning("This uploaded CSV could include non-FBS teams. Add a CFBD API key and refresh, or upload an FBS-only CSV, before relying on these projections.")

col1, col2, col3 = st.columns(3)
with col1:
    home = st.selectbox("Home team", teams, index=0)
with col2:
    away = st.selectbox("Away team", teams, index=min(1, len(teams)-1))
with col3:
    neutral = st.checkbox("Neutral site", value=False)

with st.expander("Model inputs", expanded=True):
    a, b, c = st.columns(3)
    with a:
        home_field = st.slider("Home-field advantage (points)", 0.0, 4.0, 2.0, 0.25, disabled=neutral)
    with b:
        form_weight = st.slider("Recent form weight", 0.0, 0.6, 0.25, 0.05)
    with c:
        margin_sd = st.slider("Scoring margin uncertainty", 8.0, 20.0, 14.0, 0.5)

season_r, recent_r, counts = calculate_ratings(games, home_field=home_field)

if home == away:
    st.warning("Choose two different teams to project a matchup.")
    st.stop()

home_rating = (1 - form_weight) * season_r[home] + form_weight * recent_r[home]
away_rating = (1 - form_weight) * season_r[away] + form_weight * recent_r[away]
home_adv = 0.0 if neutral else home_field
spread = home_rating - away_rating + home_adv
home_win = 1 / (1 + math.exp(-1.702 * spread / margin_sd))

st.subheader(f"{home} vs. {away}")
m1, m2, m3 = st.columns(3)
m1.metric("Projected spread", f"{home} {(-spread):+.1f}", help="Standard spread format: a negative number means the home team is favored.")
m2.metric(f"{home} win probability", f"{home_win:.1%}")
m3.metric(f"{away} win probability", f"{1-home_win:.1%}")

st.progress(float(home_win), text=f"{home} {home_win:.1%}  ·  {away} {1-home_win:.1%}")
st.markdown("#### Team form and season strength")
comparison = pd.DataFrame({
    "Team": [home, away],
    "Adjusted season rating": [season_r[home], season_r[away]],
    "Recent form rating": [recent_r[home], recent_r[away]],
    "Completed games": [counts[home], counts[away]],
})
st.dataframe(comparison.style.format({"Adjusted season rating": "+.1f", "Recent form rating": "+.1f"}), hide_index=True, use_container_width=True)
st.caption("Ratings are points above/below the field on a neutral site. Spread is a model projection before any market adjustment; probability is a normal-margin approximation. Results are opponent adjusted and shrunk toward zero for small samples.")

with st.expander("Loaded season results"):
    st.dataframe(games[["week", "away_team", "away_points", "home_team", "home_points"]].sort_values(["week", "home_team"], ascending=[False, True]), hide_index=True, use_container_width=True)

st.markdown("---")
st.subheader("Upcoming FBS game predictions")
upcoming = st.session_state.get("upcoming_games", pd.DataFrame())
if st.session_state.get("upcoming_error"):
    st.warning(f"Upcoming schedule unavailable: {st.session_state.upcoming_error}")
elif upcoming.empty:
    st.info("CollegeFootballData did not return any future regular-season FBS-vs-FBS games. Use **Refresh 2026 results** in the sidebar to check again.")
else:
    upcoming_predictions = project_upcoming_games(
        upcoming, season_r, recent_r, home_field, form_weight, margin_sd
    )
    as_of = datetime.now(timezone.utc).strftime("%b %d, %Y %I:%M %p UTC")
    st.caption(f"{len(upcoming_predictions)} scheduled matchups · projections generated {as_of}. Ratings use the {len(games)} completed games currently loaded above. Kickoff times are UTC. Download a dated CSV to preserve this snapshot before results change.")
    display_predictions = upcoming_predictions.rename(columns={
        "week": "Week", "kickoff_utc": "Kickoff (UTC)", "away_team": "Away team",
        "home_team": "Home team", "projected_spread": "Projected spread",
        "home_win_probability": "Home win probability", "away_win_probability": "Away win probability",
    })[["Week", "Kickoff (UTC)", "Away team", "Home team", "Projected spread", "Home win probability", "Away win probability"]]
    st.dataframe(display_predictions.style.format({
        "Home win probability": "{:.1%}", "Away win probability": "{:.1%}"
    }), hide_index=True, use_container_width=True)
    snapshot = upcoming_predictions.copy()
    snapshot.insert(0, "forecast_generated_utc", datetime.now(timezone.utc).isoformat())
    st.download_button(
        "Download upcoming predictions CSV",
        data=snapshot.to_csv(index=False).encode("utf-8"),
        file_name=f"cover0_upcoming_predictions_{SEASON}.csv",
        mime="text/csv",
    )

st.markdown("#### Grade saved forecasts")
st.caption("Upload one or more Cover 0 prediction CSVs after games finish. Cover 0 will match them to completed scores and grade only forecasts saved before kickoff when timestamps are available.")
forecast_files = st.file_uploader(
    "Upload saved prediction CSVs to grade",
    type=["csv"],
    accept_multiple_files=True,
    key="forecast_grading_upload",
)
if forecast_files:
    scored_forecasts, grading_error, excluded_late = grade_saved_forecasts(forecast_files, games)
    if grading_error:
        st.error(grading_error)
    elif scored_forecasts.empty:
        st.info("No saved forecasts matched completed games yet. Keep the CSV and upload it again after those games finish.")
    else:
        if excluded_late:
            st.warning(f"Excluded {excluded_late} forecast rows timestamped after kickoff.")
        summary = {
            "Games": len(scored_forecasts),
            "Spread_MAE": scored_forecasts.absolute_margin_error.mean(),
            "Winner_Accuracy": scored_forecasts.correct_winner.mean(),
            "Brier_Score": scored_forecasts.brier_component.mean(),
        }
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Completed games graded", f"{summary['Games']:,}")
        s2.metric("Spread MAE", f"{summary['Spread_MAE']:.2f} pts")
        s3.metric("Winner accuracy", f"{summary['Winner_Accuracy']:.1%}")
        s4.metric("Win-probability Brier", f"{summary['Brier_Score']:.3f}")
        scored_display = scored_forecasts.assign(
            Actual_Result=np.where(scored_forecasts.actual_home_win == 1, "Home win", "Away win")
        )[["week", "away_team", "home_team", "predicted_home_margin", "home_margin",
           "home_win_probability", "actual_home_win", "Actual_Result"]].rename(columns={
               "week": "Week", "away_team": "Away team", "home_team": "Home team",
               "predicted_home_margin": "Predicted home margin", "home_margin": "Actual home margin",
               "home_win_probability": "Predicted home win probability",
               "actual_home_win": "Actual home win",
           })
        st.dataframe(scored_display.style.format({
            "Predicted home margin": "+.1f", "Actual home margin": "+.0f",
            "Predicted home win probability": "{:.1%}", "Actual home win": "{:.0f}",
        }), hide_index=True, use_container_width=True)
        with st.spinner("Checking closing spreads for graded games…"):
            grading_lines, grading_lines_error = fetch_historical_closing_lines(SEASON, SEASON)
        if grading_lines is not None and not grading_lines.empty:
            market_grade = compare_to_market(scored_forecasts, grading_lines)
            if not market_grade.empty:
                wins, losses, pushes, pct = cover_record(market_grade.model_cover_result)
                st.write(f"**ATS record vs. closing spreads:** {wins}-{losses}-{pushes} ({pct:.1%}, pushes excluded). Matched lines: {len(market_grade):,}.")
            else:
                st.info("No closing spreads matched the completed games yet.")
        elif grading_lines_error:
            st.caption(f"ATS grading is not available yet: {grading_lines_error}")
        st.caption("These results describe this saved forecast sample; they do not guarantee future performance or profit.")

st.markdown("---")
st.subheader("Historical walk-forward backtest")
st.caption("For each week, the model uses only results from earlier weeks in that season. Historical seasons are fetched from CollegeFootballData.")
st.caption("The full 2018–2025 run also compares a small PPA, success-rate, and explosiveness overlay. Each forecast uses only advanced stats from earlier weeks; the overlay weight is selected on 2018–2022 and evaluated on 2023–2025.")
start_year, end_year = st.select_slider(
    "Seasons to evaluate",
    options=list(range(2016, 2026)),
    value=(2018, 2025),
)
if st.button("Run historical backtest"):
    with st.spinner(f"Loading seasons {start_year}–{end_year} and evaluating forecasts…"):
        historical, history_error = fetch_historical_games(start_year, end_year)
        if history_error:
            st.session_state.backtest_error = history_error
            st.session_state.backtest_forecasts = None
        elif historical.empty:
            st.session_state.backtest_error = "No completed games were returned for those seasons."
            st.session_state.backtest_forecasts = None
        else:
            st.session_state.backtest_error = None
            if start_year <= 2018 and end_year >= 2025:
                grid_forecasts = run_walk_forward_grid(
                    historical,
                    range(start_year, end_year + 1),
                    home_fields=(1.5, 2.0, 2.5),
                    form_weights=(0.15, 0.25, 0.30),
                )
                st.session_state.backtest_forecasts = grid_forecasts[(2.0, 0.25)]
            else:
                grid_forecasts = None
                st.session_state.backtest_forecasts = run_walk_forward_backtest(historical, range(start_year, end_year + 1))
            if start_year <= 2018 and end_year >= 2025:
                advanced_games, advanced_error = fetch_historical_advanced_games(start_year, end_year, historical)
                st.session_state.efficiency_error = advanced_error
                if advanced_games is not None:
                    efficiency_forecasts = attach_efficiency_signal(
                        grid_forecasts[(2.0, 0.25)], advanced_games
                    )
                    best_efficiency, baseline_efficiency, enhanced_efficiency = tune_efficiency_overlay(
                        efficiency_forecasts,
                        tuning_seasons=range(2018, 2023),
                        holdout_seasons=range(2023, 2026),
                    )
                    st.session_state.efficiency_experiment = {
                        "weight": best_efficiency["weight"],
                        "training_mae": best_efficiency["mae"],
                        "baseline": baseline_efficiency.to_dict(),
                        "enhanced": enhanced_efficiency.to_dict(),
                    }
                else:
                    st.session_state.efficiency_experiment = None
            else:
                st.session_state.efficiency_error = None
                st.session_state.efficiency_experiment = None
            if start_year <= 2018 and end_year >= 2025:
                turnover_games, turnover_error = fetch_historical_turnovers(start_year, end_year, historical)
                st.session_state.turnover_error = turnover_error
                if turnover_games is not None:
                    turnover_forecasts = attach_turnover_signal(grid_forecasts[(2.0, 0.25)], turnover_games)
                    best_turnover, baseline_turnover, enhanced_turnover = tune_turnover_overlay(turnover_forecasts, range(2018, 2023), range(2023, 2026))
                    st.session_state.turnover_experiment = {"weight": best_turnover["weight"], "training_mae": best_turnover["mae"], "baseline": baseline_turnover.to_dict(), "enhanced": enhanced_turnover.to_dict()}
                else:
                    st.session_state.turnover_experiment = None
            else:
                st.session_state.turnover_error = None
                st.session_state.turnover_experiment = None

            closing_lines, lines_error = fetch_historical_closing_lines(start_year, end_year)
            st.session_state.backtest_lines_error = lines_error
            st.session_state.holdout_lines = closing_lines
            st.session_state.backtest_market = (
                compare_to_market(st.session_state.backtest_forecasts, closing_lines)
                if closing_lines is not None else None
            )
            if start_year <= 2018 and end_year >= 2025:
                tuning_seasons = list(range(2018, 2023))
                holdout_seasons = list(range(2023, 2026))
                candidates, best, baseline, tuned = run_holdout_tuning(
                    grid_forecasts, tuning_seasons, holdout_seasons
                )
                st.session_state.holdout_candidates = candidates
                st.session_state.holdout_best = best.to_dict()
                st.session_state.holdout_baseline = baseline
                st.session_state.holdout_tuned = tuned
                st.session_state.holdout_message = None
            else:
                st.session_state.holdout_candidates = None
                st.session_state.holdout_best = None
                st.session_state.holdout_baseline = None
                st.session_state.holdout_tuned = None
                st.session_state.holdout_message = "Select the full 2018–2025 range to run this chronological holdout."

if st.session_state.get("backtest_error"):
    st.error(st.session_state.backtest_error)

forecasts = st.session_state.get("backtest_forecasts")
if forecasts is not None and not forecasts.empty:
    overall = summarize_forecasts(forecasts).iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Games evaluated", f"{int(overall.Games):,}")
    k2.metric("Spread error (MAE)", f"{overall.Spread_MAE:.1f} pts", help="Average absolute difference between predicted and actual home-team margin. Lower is better.")
    k3.metric("Winner accuracy", f"{overall.Winner_Accuracy:.1%}")
    k4.metric("Win-probability Brier score", f"{overall.Brier_Score:.3f}", help="Lower is better; evaluates the quality of win probabilities, not just picked winners.")

    st.markdown("#### Results by season")
    by_season = summarize_forecasts(forecasts, ["season"])
    st.dataframe(by_season.style.format({
        "Spread_MAE": "{:.2f}", "Margin_RMSE": "{:.2f}",
        "Winner_Accuracy": "{:.1%}", "Brier_Score": "{:.3f}",
        "Avg_Home_Win_Prob": "{:.1%}", "Actual_Home_Win_Rate": "{:.1%}",
    }), hide_index=True, use_container_width=True)

    st.markdown("#### Chronological holdout: tune 2018–2022, evaluate 2023–2025")
    st.caption("Six settings are ranked by spread MAE on 2018–2022. The selected home-field and recent-form weights are then evaluated on 2023–2025 and compared with the default settings. The later seasons do not affect setting selection.")
    if st.session_state.get("holdout_message"):
        st.info(st.session_state.holdout_message)
    elif st.session_state.get("holdout_candidates") is not None:
        candidates = st.session_state.holdout_candidates
        best = st.session_state.holdout_best
        st.write(f"Selected settings: **{best['Home_Field']:.1f} point home field** and **{best['Recent_Form_Weight']:.0%} recent-form weight** (training spread MAE **{best['Tuning_Spread_MAE']:.2f} points**).")
        st.dataframe(candidates.style.format({
            "Home_Field": "{:.1f}", "Recent_Form_Weight": "{:.0%}", "Tuning_Spread_MAE": "{:.2f}"
        }), hide_index=True, use_container_width=True)
        baseline = st.session_state.holdout_baseline
        tuned = st.session_state.holdout_tuned
        baseline_summary = summarize_forecasts(baseline).iloc[0]
        tuned_summary = summarize_forecasts(tuned).iloc[0]
        holdout_comparison = pd.DataFrame([
            {"Settings": "Current defaults", "Games": int(baseline_summary.Games), "Spread_MAE": baseline_summary.Spread_MAE,
             "Winner_Accuracy": baseline_summary.Winner_Accuracy, "Brier_Score": baseline_summary.Brier_Score},
            {"Settings": "Selected on 2018–2022", "Games": int(tuned_summary.Games), "Spread_MAE": tuned_summary.Spread_MAE,
             "Winner_Accuracy": tuned_summary.Winner_Accuracy, "Brier_Score": tuned_summary.Brier_Score},
        ])
        holdout_lines = st.session_state.get("holdout_lines")
        if holdout_lines is not None and not holdout_lines.empty:
            selected_lines = holdout_lines[holdout_lines.season.isin([2023, 2024, 2025])]
            for label, data in (("Current defaults", baseline), ("Selected on 2018–2022", tuned)):
                matched = compare_to_market(data, selected_lines)
                record = cover_record(matched.model_cover_result) if not matched.empty else (0, 0, 0, float("nan"))
                holdout_comparison.loc[holdout_comparison.Settings == label, "ATS_Record"] = f"{record[0]}-{record[1]}-{record[2]}"
                holdout_comparison.loc[holdout_comparison.Settings == label, "ATS_Win_Rate"] = record[3]
        st.dataframe(holdout_comparison.style.format({
            "Spread_MAE": "{:.2f}", "Winner_Accuracy": "{:.1%}", "Brier_Score": "{:.3f}", "ATS_Win_Rate": "{:.1%}"
        }), hide_index=True, use_container_width=True)
        efficiency_result = st.session_state.get("efficiency_experiment")
        if st.session_state.get("efficiency_error"):
            st.warning(f"The base backtest ran, but the efficiency-input comparison was unavailable: {st.session_state.efficiency_error}")
        elif efficiency_result:
            st.markdown("#### Efficiency-input experiment (holdout: 2023–2025)")
            st.caption("The feature is the difference in teams’ pregame PPA, success-rate, and explosiveness profile, based only on earlier weeks. A weight of zero is the current-score model; the selected weight was tuned on 2018–2022.")
            efficiency_comparison = pd.DataFrame([
                {"Model": "Current score-based model", "Weight (points per efficiency unit)": 0.0, **efficiency_result["baseline"]},
                {"Model": "With efficiency overlay", "Weight (points per efficiency unit)": efficiency_result["weight"], **efficiency_result["enhanced"]},
            ])
            st.write(f"Selected overlay weight: **{efficiency_result['weight']:.2f} points per efficiency unit** (2018–2022 training MAE: **{efficiency_result['training_mae']:.2f} points**).")
            st.dataframe(efficiency_comparison.style.format({
                "Weight (points per efficiency unit)": "{:.2f}", "Spread_MAE": "{:.2f}",
                "Margin_RMSE": "{:.2f}", "Winner_Accuracy": "{:.1%}", "Brier_Score": "{:.3f}",
                "Avg_Home_Win_Prob": "{:.1%}", "Actual_Home_Win_Rate": "{:.1%}",
            }), hide_index=True, use_container_width=True)
            if efficiency_result["enhanced"]["Spread_MAE"] < efficiency_result["baseline"]["Spread_MAE"]:
                st.success("The efficiency overlay improved holdout spread error in this historical comparison. Keep it experimental until it also holds up on future saved forecasts.")
            else:
                st.info("The efficiency overlay did not improve holdout spread error, so this comparison does not support adding it to live predictions.")
        turnover_result = st.session_state.get("turnover_experiment")
        if st.session_state.get("turnover_error"):
            st.warning(f"Turnover comparison unavailable: {st.session_state.turnover_error}")
        elif turnover_result:
            st.markdown("#### Turnover experiment (holdout: 2023–2025)")
            st.caption("Each forecast uses turnovers from earlier weeks only. The input is tuned on 2018–2022; live predictions remain unchanged.")
            turnover_table = pd.DataFrame([
                {"Model": "Current score-based model", "Weight": 0.0, **turnover_result["baseline"]},
                {"Model": "With turnover input", "Weight": turnover_result["weight"], **turnover_result["enhanced"]},
            ])
            st.write(f"Selected weight: **{turnover_result['weight']:.2f}** (training spread MAE: **{turnover_result['training_mae']:.2f}** points).")
            st.dataframe(turnover_table.style.format({"Weight": "{:.2f}", "Spread_MAE": "{:.2f}", "Winner_Accuracy": "{:.1%}", "Brier_Score": "{:.3f}"}), hide_index=True, use_container_width=True)
            if turnover_result["enhanced"]["Spread_MAE"] < turnover_result["baseline"]["Spread_MAE"]:
                st.success("Turnover improved holdout spread error in this historical test; keep it experimental pending future forecasts.")
            else:
                st.info("Turnover did not improve holdout spread error in this test, so it is not added to live predictions.")

        if holdout_lines is None or holdout_lines.empty:
            st.caption("ATS results are unavailable because no matching historical closing lines were loaded.")
        st.caption("This is a historical model comparison, not a guarantee of future performance or betting profit.")

    market = st.session_state.get("backtest_market")
    if st.session_state.get("backtest_lines_error"):
        st.warning(f"Model backtest ran, but closing-line comparison was unavailable: {st.session_state.backtest_lines_error}")
    elif market is not None:
        if market.empty:
            st.info("CFBD returned no matching closing lines for these games.")
        else:
            st.markdown("#### Cover 0 vs. historical closing spreads")
            st.caption("This uses CFBD closing spreads, preferring its consensus provider and otherwise taking the median available provider line. Games without a matched line are excluded. ATS records exclude pushes; this is not a profit or ROI estimate.")
            model_w, model_l, model_p, model_pct = cover_record(market.model_cover_result)
            market_w, market_l, market_p, market_pct = cover_record(market.market_cover_result)
            matched = len(market)
            a, b, c, d = st.columns(4)
            a.metric("Games with matched lines", f"{matched:,}")
            b.metric("Cover 0 spread MAE", f"{market.model_margin_error.mean():.2f} pts")
            c.metric("Closing-line spread MAE", f"{market.market_margin_error.mean():.2f} pts")
            d.metric("Cover 0 ATS record", f"{model_w}-{model_l}-{model_p} ({model_pct:.1%})")
            st.write(f"Closing line home-side record: **{market_w}-{market_l}-{market_p} ({market_pct:.1%})**. ATS percentages exclude pushes.")
            market_seasons = market.groupby("season").agg(
                Games=("actual_home_margin", "size"),
                Cover_0_Spread_MAE=("model_margin_error", "mean"),
                Closing_Line_Spread_MAE=("market_margin_error", "mean"),
            ).reset_index()
            season_records = []
            for year, season_market in market.groupby("season"):
                wins, losses, pushes, pct = cover_record(season_market.model_cover_result)
                line_wins, line_losses, line_pushes, line_pct = cover_record(season_market.market_cover_result)
                season_records.append({
                    "season": year,
                    "Cover_0_ATS": f"{wins}-{losses}-{pushes}",
                    "Cover_0_ATS_Pct": pct,
                    "Home_Line_Record": f"{line_wins}-{line_losses}-{line_pushes}",
                    "Home_Line_ATS_Pct": line_pct,
                })
            market_seasons = market_seasons.merge(pd.DataFrame(season_records), on="season", how="left")
            st.markdown("#### ATS results by season")
            st.caption("Records show wins-losses-pushes. ATS rates exclude pushes; small season samples can vary widely.")
            st.dataframe(market_seasons.style.format({
                "Cover_0_Spread_MAE": "{:.2f}",
                "Closing_Line_Spread_MAE": "{:.2f}",
                "Cover_0_ATS_Pct": "{:.1%}",
                "Home_Line_ATS_Pct": "{:.1%}",
            }), hide_index=True, use_container_width=True)

            # Existing browser sessions may retain a pre-gap comparison table after deploy.
            if "model_market_gap_band" not in market.columns:
                market["model_market_gap"] = (market.predicted_home_margin - market.market_home_margin).abs()
                market["model_market_gap_band"] = pd.cut(
                    market.model_market_gap,
                    bins=[0, 1, 3, 5, np.inf],
                    labels=["Under 1 pt", "1–<3 pts", "3–<5 pts", "5+ pts"],
                    right=False,
                )
            edge_rows = []
            for band, band_games in market.groupby("model_market_gap_band", observed=True):
                wins, losses, pushes, pct = cover_record(band_games.model_cover_result)
                edge_rows.append({
                    "Model_vs_Line_Gap": str(band),
                    "Games": len(band_games),
                    "Cover_0_ATS": f"{wins}-{losses}-{pushes}",
                    "ATS_Win_Rate": pct,
                })
            st.markdown("#### ATS by model-to-line difference")
            st.caption("Diagnostic buckets show how often Cover 0's side covered as its spread moved away from the closing line. These historical buckets are descriptive, not tuned betting cutoffs.")
            st.dataframe(pd.DataFrame(edge_rows).style.format({
                "ATS_Win_Rate": "{:.1%}",
            }), hide_index=True, use_container_width=True)

    st.markdown("#### Probability calibration")
    calibration = forecasts.copy()
    calibration["favorite_probability"] = np.maximum(calibration.home_win_probability, 1 - calibration.home_win_probability)
    calibration["favorite_won"] = np.where(
        calibration.home_win_probability >= 0.5,
        calibration.actual_home_win,
        1 - calibration.actual_home_win,
    )
    calibration["favorite_probability_band"] = pd.cut(
        calibration.favorite_probability,
        bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.000001],
        labels=["50–60%", "60–70%", "70–80%", "80–90%", "90–100%"],
        include_lowest=True,
    )
    calibration_summary = calibration.groupby("favorite_probability_band", observed=True).agg(
        Games=("favorite_won", "size"),
        Predicted_Favorite_Win_Rate=("favorite_probability", "mean"),
        Actual_Favorite_Win_Rate=("favorite_won", "mean"),
    ).reset_index()
    st.dataframe(calibration_summary.style.format({
        "Predicted_Favorite_Win_Rate": "{:.1%}",
        "Actual_Favorite_Win_Rate": "{:.1%}",
    }), hide_index=True, use_container_width=True)
    st.download_button(
        "Download backtest forecasts",
        forecasts.to_csv(index=False).encode("utf-8"),
        file_name=f"cfb_walk_forward_{start_year}_{end_year}.csv",
        mime="text/csv",
    )
    st.caption("These metrics evaluate the current formula against prior game outcomes. They do not establish that the model can beat sportsbook odds or earn a profit. Compare model projections with closing spreads and odds before making betting decisions.")
