from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT_ELO, ELO_K


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opponent_rating - rating) / 400.0))


def result_score(label: str) -> float:
    if label == "win":
        return 1.0
    if label == "draw":
        return 0.5
    if label == "loss":
        return 0.0
    raise ValueError(f"Unknown result label: {label}")


def margin_multiplier(goal_diff: int) -> float:
    diff = abs(goal_diff)
    if diff <= 1:
        return 1.0
    return 1.0 + min(diff - 1, 4) * 0.25


@dataclass
class EloRatings:
    default_rating: float = DEFAULT_ELO
    k_factor: float = ELO_K
    ratings: dict[str, float] = field(default_factory=dict)

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def update(self, team: str, opponent: str, label: str, goal_diff: int = 0) -> None:
        team_rating = self.get(team)
        opponent_rating = self.get(opponent)
        expected = expected_score(team_rating, opponent_rating)
        actual = result_score(label)
        change = self.k_factor * margin_multiplier(goal_diff) * (actual - expected)
        self.ratings[team] = team_rating + change
        self.ratings[opponent] = opponent_rating - change
