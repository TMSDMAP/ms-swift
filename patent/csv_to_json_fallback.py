#!/usr/bin/env python3
"""Fallback converter: use existing CSV files to recover failed DTA->JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ENCODINGS = ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin1"]


def clean_text(text: str) -> str:
    text = text.replace("\ufffd", " ")
    text = CONTROL_CHAR_RE.sub(" ", text)
    return text


def sanitize_value(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return clean_text(str(value))


def make_unique_columns(cols: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        name = str(c)
        n = seen.get(name, 0)
        if n == 0:
            out.append(name)
        else:
            out.append(f"{name}__dup{n}")
        seen[name] = n + 1
    return out


def parse_error_line(line: str) -> Path:
    line = line.strip()
    if not line:
        raise ValueError("empty error line")
    marker = ".dta"
    idx = line.find(marker)
    if idx < 0:
        raise ValueError(f"invalid error line: {line}")
    return Path(line[: idx + len(marker)])


def get_failed_dta_paths(error_log: Path) -> List[Path]:
    paths: List[Path] = []
    for raw in error_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            paths.append(parse_error_line(raw))
        except Exception:
            continue
    return paths


def detect_csv_dialect(csv_path: Path) -> Tuple[str, str, str]:
    for enc in ENCODINGS:
        try:
            with csv_path.open("r", encoding=enc, errors="replace", newline="") as f:
                sample = f.read(30000)
                if not sample:
                    return enc, ",", '"'
                try:
                    d = csv.Sniffer().sniff(sample, delimiters=",\t;|")
                    delim = d.delimiter
                    quote = getattr(d, "quotechar", '"') or '"'
                except Exception:
                    delim = ","
                    quote = '"'
                return enc, delim, quote
        except Exception:
            continue
    return "utf-8-sig", ",", '"'


def stream_json_open(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fout = path.open("w", encoding="utf-8")
    fout.write("[\n")
    return fout


def stream_json_write_record(fout, record: dict, is_first: bool) -> bool:
    if not is_first:
        fout.write(",\n")
    json.dump(record, fout, ensure_ascii=False)
    return False


def stream_json_close(fout):
    fout.write("\n]\n")
    fout.close()


def convert_with_pandas(csv_path: Path, out_json: Path, chunksize: int) -> int:
    enc, delim, quote = detect_csv_dialect(csv_path)
    rows_written = 0
    is_first = True
    fout = stream_json_open(out_json)
    try:
        for chunk in pd.read_csv(
            csv_path,
            encoding=enc,
            sep=delim,
            quotechar=quote,
            engine="python",
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            on_bad_lines="error",
        ):
            chunk.columns = make_unique_columns(chunk.columns)
            for rec in chunk.to_dict(orient="records"):
                item = {k: sanitize_value(v) for k, v in rec.items()}
                is_first = stream_json_write_record(fout, item, is_first)
                rows_written += 1
        stream_json_close(fout)
        return rows_written
    except Exception:
        fout.close()
        raise


def convert_with_csv_module(csv_path: Path, out_json: Path) -> int:
    enc, delim, quote = detect_csv_dialect(csv_path)
    rows_written = 0
    is_first = True
    fout = stream_json_open(out_json)
    try:
        with csv_path.open("r", encoding=enc, errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delim, quotechar=quote)
            raw_header = next(reader)
            header = make_unique_columns(raw_header)
            base_len = len(header)
            for row in reader:
                item = {}
                fixed = list(row[:base_len])
                if len(fixed) < base_len:
                    fixed.extend([""] * (base_len - len(fixed)))
                for i, key in enumerate(header):
                    item[key] = sanitize_value(fixed[i])
                if len(row) > base_len:
                    for j, extra_val in enumerate(row[base_len:], start=1):
                        item[f"__extra_{j}"] = sanitize_value(extra_val)
                is_first = stream_json_write_record(fout, item, is_first)
                rows_written += 1
        stream_json_close(fout)
        return rows_written
    except Exception:
        fout.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover failed DTA outputs from existing CSV files.")
    parser.add_argument(
        "--error-log",
        type=Path,
        default=Path("/home/ljh/data1/patent/dta_to_json/conversion_errors.log"),
    )
    parser.add_argument(
        "--csv-root",
        type=Path,
        default=Path("/home/ljh/data1/patent"),
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("/home/ljh/data1/patent/dta_to_json"),
    )
    parser.add_argument("--chunksize", type=int, default=120000)
    args = parser.parse_args()

    failed = get_failed_dta_paths(args.error_log)
    csv_by_stem = {p.stem: p for p in args.csv_root.glob("*.csv")}

    matched = []
    missing = []
    for dta in failed:
        stem = dta.stem
        csv_path = csv_by_stem.get(stem)
        if csv_path is None:
            missing.append(dta)
        else:
            matched.append((dta, csv_path))

    print(f"failed_dta_total={len(failed)}")
    print(f"matched_csv={len(matched)}")
    print(f"missing_csv={len(missing)}")

    recovered = 0
    recover_fail = 0
    fallback_used = 0
    recover_fail_lines: List[str] = []

    for idx, (dta_path, csv_path) in enumerate(matched, start=1):
        province = dta_path.parent.name.strip()
        out_json = args.json_root / province / f"{dta_path.stem}.json"
        try:
            rows = convert_with_pandas(csv_path, out_json, chunksize=args.chunksize)
            recovered += 1
            print(f"[{idx}/{len(matched)}] ok pandas: {csv_path.name} -> {out_json} rows={rows}")
        except Exception as e1:
            try:
                rows = convert_with_csv_module(csv_path, out_json)
                recovered += 1
                fallback_used += 1
                print(f"[{idx}/{len(matched)}] ok csv-fallback: {csv_path.name} -> {out_json} rows={rows}")
            except Exception as e2:
                recover_fail += 1
                recover_fail_lines.append(f"{csv_path}\t{type(e2).__name__}: {e2} (after pandas error: {type(e1).__name__}: {e1})")
                print(f"[{idx}/{len(matched)}] FAIL: {csv_path.name}")

    missing_log = args.json_root / "csv_missing_for_failed_dta.log"
    missing_log.write_text("\n".join(str(p) for p in missing) + ("\n" if missing else ""), encoding="utf-8")

    recover_fail_log = args.json_root / "csv_recovery_errors.log"
    if recover_fail_lines:
        recover_fail_log.write_text("\n".join(recover_fail_lines) + "\n", encoding="utf-8")
    elif recover_fail_log.exists():
        recover_fail_log.unlink()

    print("===== CSV Recovery Done =====")
    print(f"recovered_ok={recovered}")
    print(f"recovered_fail={recover_fail}")
    print(f"fallback_used={fallback_used}")
    print(f"missing_log={missing_log}")
    if recover_fail_lines:
        print(f"recover_fail_log={recover_fail_log}")


if __name__ == "__main__":
    main()
