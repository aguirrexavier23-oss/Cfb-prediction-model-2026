import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

SEASON = 2026
BASE_URL = "https://api.collegefootballdata.com"

st.set_page_config(page_title="2026 CFB Matchup Model", page_icon="🏈", layout="wide")
st.title("🏈 2026 College Football Matchup Model")
st.caption("Season-aware projections from completed results, opponent strength, recent form, and home field.")


def normalize_games(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "season": ["season", "year"], "week": ["week"],
        "home_team": ["home_team", "homeTeam", "home"],
        "away_team": ["away_team", "awayTeam", "away"],
        "home_points": ["home_points", "homePoints", "home_score"],
        "away_points": ["away_points", "awayPoints", "away_score"],
        "completed": ["completed", "is_completed"],
    }
    out = pd.DataFrame(index=df.index)
    for target, opts in aliases.items():
        found = next((c for c in opts if c in df.columns), None)
        out[target] = df[found] if found else (np.nan if target not in ("completed",) else True)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").fillna(SEASON)
    out["week"] = pd.to_numeric(out["week"], errors="coerce").fillna(0)
    out["home_points"] = pd.to_numeric(out["home_points"], errors="coerce")
    out["away_points"] = pd.to_numeric(out["away_points"], errors="coerce")
    out["home_team"] = out["home_team"].astype("string").str.strip()
    out["away_team"] = out["away_team"].astype("string").str.strip()
    if out["completed"].dtype != bool:
        out["completed"] = out["completed"].astype(str).str.lower().isin(["true", "1", "yes", "completed"])
    out = out[(out.season == SEASON) & out.completed & out.home_points.notna() & out.away_points.notna()]
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
        response = requests.get(
            f"{BASE_URL}/games",
            params={"year": SEASON, "seasonType": "regular", "division": "fbs"},
            headers={"Authorization": f"Bearer {key}"}, timeout=25,
        )
        response.raise_for_status()
        return normalize_games(pd.DataFrame(response.json())), None
    except Exception as exc:
        return None, f"Could not fetch season data: {exc}"


def calculate_ratings(games: pd.DataFrame):
    teams = sorted(set(games.home_team) | set(games.away_team))
    if not teams:
        return {}, {}, {}
    idx = {team: i for i, team in enumerate(teams)}
    # Solve team strength differences against results. Ridge shrinkage stabilizes small samples.
    X = np.zeros((len(games), len(teams)), dtype=float)
    y = games.home_margin.to_numpy(dtype=float)
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
    recent = {}
    counts = {t: 0 for t in teams}
    for team in teams:
        vals, ws = [], []
        for g in games.itertuples():
            if g.home_team == team:
                opponent = g.away_team
                margin = g.home_margin
            elif g.away_team == team:
                opponent = g.home_team
                margin = -g.home_margin
            else:
                continue
            w = max(0.45, math.exp(-0.18 * (max_week - g.week)))
            vals.append(margin + 0.35 * season[opponent])
            ws.append(w)
            counts[team] += 1
        recent[team] = float(np.average(vals, weights=ws)) if vals else 0.0
    return season, recent, counts


with st.sidebar:
    st.header("Season data")
    if "games" not in st.session_state:
        fetched, fetch_error = fetch_games()
        st.session_state.games = fetched if fetched is not None else pd.DataFrame()
        st.session_state.data_note = fetch_error
    if st.button("Refresh 2026 results"):
        fetched, fetch_error = fetch_games()
        if fetched is not None:
            st.session_state.games = fetched
            st.session_state.data_note = None
            st.success(f"Loaded {len(fetched)} completed games.")
        else:
            st.session_state.data_note = fetch_error
            st.error(fetch_error)
    upload = st.file_uploader("Or upload season CSV", type=["csv"])
    if upload is not None:
        try:
            uploaded = normalize_games(pd.read_csv(upload))
            st.session_state.games = uploaded
            st.session_state.data_note = None
            st.success(f"Loaded {len(uploaded)} completed 2026 games.")
        except Exception as exc:
            st.error(f"CSV could not be read: {exc}")
    st.caption("API access uses CollegeFootballData. Set CFBD_API_KEY in your environment to fetch automatically.")

games = st.session_state.games
if games.empty:
    st.info(st.session_state.get("data_note") or "No completed 2026 results loaded yet. You can upload a CSV or add CFBD_API_KEY and refresh.")
    st.markdown("CSV fields: `season, week, home_team, away_team, home_points, away_points` and optional `completed`.")
    st.stop()

season_r, recent_r, counts = calculate_ratings(games)
teams = sorted(season_r)
latest_week = int(games.week.max())
asof = datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")
st.caption(f"Using {len(games)} completed games through week {latest_week}. Loaded {asof}.")

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
m1.metric("Projected spread", f"{home} {(-spread):+.1f}" if spread < 0 else f"{home} {spread:+.1f}", help="Shown as the home team's projected margin; positive means home favored.")
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
