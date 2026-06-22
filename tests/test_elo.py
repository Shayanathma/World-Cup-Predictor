from worldcup_predictor.elo import EloRatings, expected_score


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == 0.5


def test_elo_update_win_adds_points_to_winner():
    ratings = EloRatings()
    ratings.update("Argentina", "France", "win", goal_diff=1)

    assert ratings.get("Argentina") > 1500
    assert ratings.get("France") < 1500
