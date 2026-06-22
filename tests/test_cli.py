from worldcup_predictor.cli import build_parser, build_train_parser


def test_cli_parses_root_prediction_teams():
    args = build_parser().parse_args(["Argentina", "Austria"])

    assert args.team_a == "Argentina"
    assert args.team_b == "Austria"


def test_cli_parses_quoted_multi_word_team_as_single_argument():
    args = build_parser().parse_args(["Saudi Arabia", "Spain"])

    assert args.team_a == "Saudi Arabia"
    assert args.team_b == "Spain"


def test_cli_parses_train_force_download():
    args = build_train_parser().parse_args(["--force-download"])

    assert args.force_download is True
