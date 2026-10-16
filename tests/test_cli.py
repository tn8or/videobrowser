from videobrowser.cli import build_parser


def test_no_landscape_flag():
    args = build_parser().parse_args(["scan", "in", "-o", "out", "--no-landscape"])
    assert args.no_landscape and not args.no_vertical
    args = build_parser().parse_args(["scan", "in", "-o", "out"])
    assert not args.no_landscape
