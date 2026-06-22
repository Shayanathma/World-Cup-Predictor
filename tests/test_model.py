import pytest

from worldcup_predictor.model import validate_team


def test_validate_team_accepts_case_insensitive_match():
    assert validate_team("argentina", ["Argentina", "France"]) == "Argentina"


def test_validate_team_rejects_unknown_with_message():
    with pytest.raises(ValueError, match="Unknown team"):
        validate_team("Atlantis", ["Argentina", "France"])
