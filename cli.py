"""Shared --states / --out arguments. No data is bundled with this package."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_COLUMNS = [
    "pair",
    "window_id",
    "t_ns",
    "split",
    "phase",
    "T_left",
    "level_ok",
    "pB",
    "bB",
    "aB",
    "spreadB",
    "dB_open",
    "rB5",
    "imbB",
]


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--states",
        default=os.environ.get("S09B_STATES"),
        help="Parquet of as-of pair states. Or set S09B_STATES. See README.md for the schema.",
    )
    p.add_argument(
        "--out",
        default=os.environ.get("S09B_OUT", "outputs"),
        help="Directory for scores, trades, and plots (default: ./outputs, or S09B_OUT).",
    )
    return p


def resolve(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.states:
        sys.exit("pass --states path/to/states.parquet, or set S09B_STATES. See README.md.")
    src = Path(args.states).expanduser()
    if not src.is_file():
        sys.exit(f"states file not found: {src}")
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    return src, out


def require_columns(df) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"states parquet is missing columns: {missing}. See README.md.")
    splits = set(df["split"].dropna().unique())
    if "discovery" not in splits:
        sys.exit("states parquet has no rows with split='discovery' (the training split).")
