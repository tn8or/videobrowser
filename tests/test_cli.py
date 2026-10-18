from videobrowser.cli import build_parser


def test_no_landscape_flag():
    args = build_parser().parse_args(["scan", "in", "-o", "out", "--no-landscape"])
    assert args.no_landscape and not args.no_vertical
    args = build_parser().parse_args(["scan", "in", "-o", "out"])
    assert not args.no_landscape


def test_no_overlay_flag_and_racebox_alias():
    parse = build_parser().parse_args
    assert parse(["scan", "in", "-o", "out", "--no-overlay"]).no_overlay
    assert parse(["scan", "in", "-o", "out", "--no-racebox"]).no_overlay
    assert not parse(["scan", "in", "-o", "out"]).no_overlay
