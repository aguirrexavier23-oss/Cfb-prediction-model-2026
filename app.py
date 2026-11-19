import math
import os
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
    if st.button("Refresh 2026 results"):
        fetched, fetch_error = fetch_games()
        if fetched is not None:
            st.session_state.games = fetched
            st.session_state.data_note = None
            st.session_state.data_loaded_at = datetime.now(timezone.utc)
            st.session_state.data_source = "CollegeFootballData API · FBS-vs-FBS games"
            st.success(f"Loaded {len(fetched)} completed games.")
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
st.subheader("Historical walk-forward backtest")
st.caption("For each week, the model uses only results from earlier weeks in that season. Historical seasons are fetched from CollegeFootballData.")
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
