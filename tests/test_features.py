import pandas as pd

from worldcup_predictor.config import LABEL_TO_ID
from worldcup_predictor.data import filter_missing_scores
from worldcup_predictor.features import build_training_frame, label_from_scores


def test_label_uses_score_before_shootout():
    assert label_from_scores(2, 1, "A", "B", "B") == "win"
    assert label_from_scores(1, 2, "A", "B", "A") == "loss"


def test_label_uses_shootout_winner_for_drawn_match():
    assert label_from_scores(1, 1, "A", "B", "A") == "win"
    assert label_from_scores(1, 1, "A", "B", "B") == "loss"
    assert label_from_scores(1, 1, "A", "B", None) == "draw"


def test_training_frame_builds_mirrored_rows_and_targets():
    matches = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-01"),
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "city": "City",
                "country": "Country",
                "neutral": True,
                "shootout_winner": "A",
            },
            {
                "date": pd.Timestamp("2020-02-01"),
                "home_team": "B",
                "away_team": "A",
                "home_score": 2,
                "away_score": 0,
                "tournament": "Friendly",
                "city": "City",
                "country": "Country",
                "neutral": False,
                "shootout_winner": None,
            },
        ]
    )

    features, target, feature_names, state = build_training_frame(matches)

    assert len(features) == 4
    assert len(target) == 4
    assert "elo_diff" in feature_names
    assert target.iloc[0] == LABEL_TO_ID["win"]
    assert target.iloc[1] == LABEL_TO_ID["loss"]
    assert state.elo.get("A") != 1500


def test_filter_missing_scores_removes_rows_before_feature_generation():
    results = pd.DataFrame(
        [
            {"home_team": "A", "away_team": "B", "home_score": 1.0, "away_score": 0.0},
            {"home_team": "C", "away_team": "D", "home_score": None, "away_score": 2.0},
            {"home_team": "E", "away_team": "F", "home_score": 3.0, "away_score": None},
        ]
    )

    filtered, removed = filter_missing_scores(results)

    assert removed == 2
    assert len(filtered) == 1
    assert filtered.iloc[0]["home_team"] == "A"
