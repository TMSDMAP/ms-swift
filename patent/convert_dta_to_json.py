#!/usr/bin/env python3
"""Batch convert province-level .dta patent files to JSON with robust text sanitization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import warnings
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import pyreadstat

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
MOJIBAKE_HINT_RE = re.compile(r"[鍙鎶鐢绗鏍鎽棣瀹鑷澶杞]")


def clean_text(text: str) -> str:
    """Replace malformed or non-printable chars with whitespace."""
    text = text.replace("\ufffd", " ")
    text = CONTROL_CHAR_RE.sub(" ", text)
    return text


def sanitize_value(value: Any) -> Any:
    """Convert values to JSON-safe types while preserving rows."""
    if value is None:
        return None

    if isinstance(value, np.generic):
        value = value.item()

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, bytes):
        return clean_text(value.decode("utf-8", errors="replace"))

    if isinstance(value, str):
        return clean_text(value)

    if isinstance(value, (pd.Timestamp, dt.datetime, dt.date)):
        return value.isoformat()

    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None

    return value


def make_unique_columns(columns: List[str]) -> List[str]:
    """Ensure column names are unique so dict serialization never drops fields."""
    seen: dict[str, int] = {}
    unique: List[str] = []
    for raw in columns:
        col = str(raw)
        n = seen.get(col, 0)
        if n == 0:
            unique.append(col)
        else:
            unique.append(f"{col}__dup{n}")
        seen[col] = n + 1
    return unique


def score_columns(columns: List[str]) -> int:
    """Heuristic score: prefer readable Chinese/ASCII headers, penalize mojibake-like patterns."""
    score = 0
    for col in columns:
        name = str(col)
        if not name:
            continue
        # Reward common patent header keywords.
        if any(k in name for k in ("申请", "专利", "发明", "摘要", "权利", "公开号", "IPC", "标题")):
            score += 4
        # Reward mostly printable names.
        printable = sum(1 for ch in name if ch.isprintable())
        if printable == len(name):
            score += 1
        # Penalize frequent mojibake hints.
        score -= len(MOJIBAKE_HINT_RE.findall(name))
    return score


def _read_with_pyreadstat(file_path: Path, encoding: str | None) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"column '.+' is duplicated, renamed to '.+'",
            category=UserWarning,
        )
        df, _meta = pyreadstat.read_dta(
            str(file_path),
            apply_value_formats=False,
            encoding=encoding,
        )
    return df


def read_dta_robust(file_path: Path) -> pd.DataFrame:
    """Read .dta with pandas first, fallback to pyreadstat encoding probes."""
    try:
        df = pd.read_stata(file_path, convert_categoricals=False)
        df.columns = make_unique_columns([str(c) for c in df.columns])
        return df
    except Exception as first_exc:
        last_exc: Exception = first_exc

        # First try pyreadstat auto-detected encoding.
        try:
            df = _read_with_pyreadstat(file_path, encoding=None)
            df.columns = make_unique_columns([str(c) for c in df.columns])
            return df
        except Exception as exc:
            last_exc = exc

        # Then probe explicit encodings and pick the best header readability score.
        best_df: pd.DataFrame | None = None
        best_score: int | None = None
        for enc in ("gb18030", "gbk", "utf-8", "latin1"):
            try:
                df = _read_with_pyreadstat(file_path, encoding=enc)
                cols = [str(c) for c in df.columns]
                s = score_columns(cols)
                if best_df is None or (best_score is not None and s > best_score) or best_score is None:
                    best_df = df
                    best_score = s
            except Exception as exc:
                last_exc = exc

        if best_df is not None:
            best_df.columns = make_unique_columns([str(c) for c in best_df.columns])
            return best_df
        raise last_exc


def write_json_rows(df: pd.DataFrame, output_path: Path) -> None:
    """Write one JSON array file while sanitizing every field."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(df.columns)
    with output_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        for row in df.itertuples(index=False, name=None):
            item = {columns[i]: sanitize_value(row[i]) for i in range(len(columns))}
            if not first:
                fout.write(",\n")
            json.dump(item, fout, ensure_ascii=False)
            first = False
        fout.write("\n]\n")


def list_dta_files(source_root: Path) -> List[Tuple[str, Path]]:
    """Return (province_name, dta_path) for one-level province folders."""
    pairs: List[Tuple[str, Path]] = []
    for province_dir in sorted(source_root.iterdir(), key=lambda p: p.name):
        if not province_dir.is_dir():
            continue
        for dta_file in sorted(province_dir.glob("*.dta"), key=lambda p: p.name):
            pairs.append((province_dir.name, dta_file))
    return pairs


def convert_all(source_root: Path, output_root: Path, log_every: int) -> None:
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    tasks = list_dta_files(source_root)
    if not tasks:
        raise RuntimeError(f"No .dta files found under: {source_root}")

    success = 0
    failed = 0
    failures: List[str] = []

    total = len(tasks)
    print(f"Found {total} dta files. Start conversion...")

    for idx, (province, dta_path) in enumerate(tasks, start=1):
        out_dir = output_root / province
        out_file = out_dir / f"{dta_path.stem}.json"

        try:
            df = read_dta_robust(dta_path)
            write_json_rows(df, out_file)
            success += 1
        except Exception as exc:
            failed += 1
            failures.append(f"{dta_path}\t{type(exc).__name__}: {exc}")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file.write_text("[]\n", encoding="utf-8")

        if log_every > 0 and idx % log_every == 0:
            print(f"Progress: {idx}/{total}, success={success}, failed={failed}")

    log_path = output_root / "conversion_errors.log"
    if failures:
        log_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
    elif log_path.exists():
        log_path.unlink()

    print("===== Conversion Done =====")
    print(f"Source: {source_root}")
    print(f"Output: {output_root}")
    print(f"Success files: {success}")
    print(f"Failed files: {failed}")
    if failures:
        print(f"Error log: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DTA files to JSON with text sanitization.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/ljh/data1/patent/原始数据dta省份汇总2003年之后的专利"),
        help="Root folder containing province subfolders with .dta files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ljh/data1/patent/dta_to_json"),
        help="Root output folder to write JSON files by province.",
    )
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_all(source_root=args.source, output_root=args.output, log_every=args.log_every)
