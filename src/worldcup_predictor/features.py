from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from math import isnan

import numpy as np
import pandas as pd

from .config import DEFAULT_ELO, LABEL_TO_ID, RECENT_WINDOWS
from .elo import EloRatings


COMPETITIVE_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa America",
    "African Cup of Nations",
    "AFC Asian Cup",
    "CONCACAF Championship",
    "CONCACAF Gold Cup",
    "Oceania Nations Cup",
}


@dataclass
class TeamHistory:
    results: deque[tuple[str, int]] = field(default_factory=lambda: deque(maxlen=20))
    last_played: pd.Timestamp | None = None


@dataclass
class FeatureState:
    elo: EloRatings = field(default_factory=EloRatings)
    histories: defaultdict[str, TeamHistory] = field(
        default_factory=lambda: defaultdict(TeamHistory)
    )
    h2h: defaultdict[tuple[str, str], list[tuple[str, int]]] = field(
        default_factory=lambda: defaultdict(list)
    )


def label_from_scores(
    team_score: int,
    opponent_score: int,
    team: str,
    opponent: str,
    shootout_winner: str | float | None,
) -> str:
    if team_score > opponent_score:
        return "win"
    if team_score < opponent_score:
        return "loss"
    if isinstance(shootout_winner, str) and shootout_winner:
        if shootout_winner == team:
            return "win"
        if shootout_winner == opponent:
            return "loss"
    return "draw"


def _score_for_history(label: str) -> float:
    if label == "win":
        return 1.0
    if label == "draw":
        return 0.5
    return 0.0


def _recent_stats(history: TeamHistory, window: int) -> dict[str, float]:
    matches = list(history.results)[-window:]
    if not matches:
        return {
            f"form_points_{window}": 0.0,
            f"form_win_rate_{window}": 0.0,
            f"form_draw_rate_{window}": 0.0,
            f"goal_diff_avg_{window}": 0.0,
            f"matches_played_{window}": 0.0,
        }
    labels = [label for label, _ in matches]
    goal_diffs = [goal_diff for _, goal_diff in matches]
    return {
        f"form_points_{window}": float(np.mean([_score_for_history(label) for label in labels])),
        f"form_win_rate_{window}": float(sum(label == "win" for label in labels) / len(labels)),
        f"form_draw_rate_{window}": float(sum(label == "draw" for label in labels) / len(labels)),
        f"goal_diff_avg_{window}": float(np.mean(goal_diffs)),
        f"matches_played_{window}": float(len(matches)),
    }


def _rest_days(history: TeamHistory, match_date: pd.Timestamp) -> float:
    if history.last_played is None:
        return 30.0
    return float(max((match_date - history.last_played).days, 0))


def _h2h_stats(state: FeatureState, team: str, opponent: str) -> dict[str, float]:
    key = tuple(sorted((team, opponent)))
    records = state.h2h[key]
    if not records:
        return {
            "h2h_matches": 0.0,
            "h2h_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_goal_diff_avg": 0.0,
        }
    team_records = [(label, gd) for record_team, label, gd in records if record_team == team]
    if not team_records:
        return {
            "h2h_matches": 0.0,
            "h2h_win_rate": 0.0,
            "h2h_draw_rate": 0.0,
            "h2h_goal_diff_avg": 0.0,
        }
    labels = [label for label, _ in team_records]
    goal_diffs = [gd for _, gd in team_records]
    return {
        "h2h_matches": float(len(team_records)),
        "h2h_win_rate": float(sum(label == "win" for label in labels) / len(labels)),
        "h2h_draw_rate": float(sum(label == "draw" for label in labels) / len(labels)),
        "h2h_goal_diff_avg": float(np.mean(goal_diffs)),
    }


def _venue_features(is_home: bool, is_away: bool, neutral: bool) -> dict[str, float]:
    return {
        "is_home": float(is_home and not neutral),
        "is_away": float(is_away and not neutral),
        "is_neutral": float(neutral),
    }


def _tournament_features(tournament: str) -> dict[str, float]:
    return {
        "is_world_cup": float(tournament == "FIFA World Cup"),
        "is_friendly": float(tournament == "Friendly"),
        "is_competitive": float(tournament in COMPETITIVE_TOURNAMENTS),
    }


def build_feature_row(
    state: FeatureState,
    *,
    team: str,
    opponent: str,
    match_date: pd.Timestamp,
    tournament: str,
    neutral: bool,
    is_home: bool,
    is_away: bool,
) -> dict[str, float]:
    team_elo = state.elo.get(team)
    opponent_elo = state.elo.get(opponent)
    team_history = state.histories[team]
    opponent_history = state.histories[opponent]

    features: dict[str, float] = {
        "team_elo": team_elo,
        "opponent_elo": opponent_elo,
        "elo_diff": team_elo - opponent_elo,
        "rest_days": _rest_days(team_history, match_date),
        "opponent_rest_days": _rest_days(opponent_history, match_date),
    }
    for window in RECENT_WINDOWS:
        for key, value in _recent_stats(team_history, window).items():
            features[f"team_{key}"] = value
        for key, value in _recent_stats(opponent_history, window).items():
            features[f"opponent_{key}"] = value
    features.update(_h2h_stats(state, team, opponent))
    features.update(_venue_features(is_home, is_away, neutral))
    features.update(_tournament_features(tournament))
    return features


def _update_state_for_match(
    state: FeatureState,
    row: pd.Series,
    home_label: str,
    away_label: str,
) -> None:
    home_team = row["home_team"]
    away_team = row["away_team"]
    home_goal_diff = int(row["home_score"] - row["away_score"])
    away_goal_diff = -home_goal_diff

    state.elo.update(home_team, away_team, home_label, home_goal_diff)
    state.histories[home_team].results.append((home_label, home_goal_diff))
    state.histories[away_team].results.append((away_label, away_goal_diff))
    state.histories[home_team].last_played = row["date"]
    state.histories[away_team].last_played = row["date"]

    key = tuple(sorted((home_team, away_team)))
    state.h2h[key].append((home_team, home_label, home_goal_diff))
    state.h2h[key].append((away_team, away_label, away_goal_diff))


def build_training_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], FeatureState]:
    state = FeatureState()
    rows: list[dict[str, float]] = []
    targets: list[int] = []

    sorted_matches = matches.sort_values("date").reset_index(drop=True)
    for _, match in sorted_matches.iterrows():
        shootout_winner = match.get("shootout_winner")
        home_label = label_from_scores(
            int(match["home_score"]),
            int(match["away_score"]),
            str(match["home_team"]),
            str(match["away_team"]),
            shootout_winner,
        )
        away_label = label_from_scores(
            int(match["away_score"]),
            int(match["home_score"]),
            str(match["away_team"]),
            str(match["home_team"]),
            shootout_winner,
        )
        common = {
            "match_date": match["date"],
            "tournament": str(match["tournament"]),
            "neutral": bool(match["neutral"]),
        }
        rows.append(
            build_feature_row(
                state,
                team=str(match["home_team"]),
                opponent=str(match["away_team"]),
                is_home=True,
                is_away=False,
                **common,
            )
        )
        targets.append(LABEL_TO_ID[home_label])
        rows.append(
            build_feature_row(
                state,
                team=str(match["away_team"]),
                opponent=str(match["home_team"]),
                is_home=False,
                is_away=True,
                **common,
            )
        )
        targets.append(LABEL_TO_ID[away_label])
        _update_state_for_match(state, match, home_label, away_label)

    frame = pd.DataFrame(rows).fillna(0.0)
    feature_names = list(frame.columns)
    return frame, pd.Series(targets, name="target"), feature_names, state


def build_prediction_features(
    state: FeatureState,
    feature_names: list[str],
    team_a: str,
    team_b: str,
    *,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    match_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if match_date is None:
        match_date = pd.Timestamp(datetime.utcnow().date())
    row = build_feature_row(
        state,
        team=team_a,
        opponent=team_b,
        match_date=match_date,
        tournament=tournament,
        neutral=neutral,
        is_home=False,
        is_away=False,
    )
    return pd.DataFrame([{name: row.get(name, 0.0) for name in feature_names}])
