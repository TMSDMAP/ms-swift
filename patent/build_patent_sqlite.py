#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sqlite3
import traceback
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

INVALID_TOKENS = {
    "",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "<na>",
    "-",
    "--",
}

REQUIRED_FIELDS = [
    "公开公告号",
    "摘要",
    "首项权利要求",
    "标题",
    "申请日",
    "IPC主分类",
    "IPC分类",
    "第一申请人",
]

FIELD_CANDIDATES: Dict[str, List[str]] = {
    "公开公告号": ["公开公告号", "公开号", "首次公开号", "授权公告号"],
    "摘要": ["摘要", "摘要中文", "摘要翻译", "摘要英文", "摘要小语种原文"],
    "首项权利要求": ["首项权利要求", "独立权利要求"],
    "标题": ["标题", "标题中文", "标题翻译", "标题英文", "标题小语种原文"],
    "申请日": ["申请日"],
    "IPC主分类": ["IPC主分类", "IPC主分类号"],
    "IPC分类": ["IPC", "IPC分类", "国际专利分类号"],
    "第一申请人": ["第一申请人", "第一申请人名称", "申请人"],
    "自引信息": ["自引信息", "被自引信息", "自引次数", "被自引次数"],
    "他引信息": ["他引信息", "被他引信息", "他引次数", "被他引次数"],
    "引证专利": ["引证专利", "被引证专利", "引证科技文献"],
    "申请人": ["申请人", "标准化申请人", "申请人翻译", "申请人其他", "当前权利人"],
}

DATE_PAT = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
LINE_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "gbk", "latin1")


def now_text() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_key(key: str) -> str:
    s = str(key)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", "", s)
    return s


def normalize_text(value: object) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False)
    else:
        s = str(value)

    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None

    if s.lower() in INVALID_TOKENS:
        return None

    return s


def normalize_date(value: object) -> Optional[str]:
    s = normalize_text(value)
    if s is None:
        return None

    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    s = s.replace("/", "-").replace(".", "-")

    if "T" in s:
        s = s.split("T", 1)[0]
    if " " in s:
        s = s.split(" ", 1)[0]

    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    if re.fullmatch(r"\d{8}", s):
        s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"

    m = DATE_PAT.fullmatch(s)
    if not m:
        return None

    y, mth, d = map(int, m.groups())
    try:
        return dt.date(y, mth, d).isoformat()
    except ValueError:
        return None


def split_first_applicant(applicants: Optional[str]) -> Optional[str]:
    if not applicants:
        return None
    parts = re.split(r"[;；,，|、]+", applicants)
    for p in parts:
        c = normalize_text(p)
        if c:
            return c
    return None


def decode_line(raw: bytes) -> str:
    for enc in LINE_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def iter_json_objects(path: Path) -> Iterator[dict]:
    with path.open("rb") as f:
        for raw in f:
            line = decode_line(raw)
            s = line.strip()
            if not s or s in {"[", "]"}:
                continue
            if s.endswith(","):
                s = s[:-1]
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def choose_key(keys: List[str], candidates: List[str], *, exclude: Optional[List[str]] = None) -> Optional[str]:
    excludes = exclude or []

    for c in candidates:
        if c in keys and not any(ex in c for ex in excludes):
            return c

    normalized = {normalize_key(k): k for k in keys}
    for c in candidates:
        nk = normalize_key(c)
        k = normalized.get(nk)
        if k is not None and not any(ex in k for ex in excludes):
            return k

    for c in candidates:
        for k in keys:
            if c in k and not any(ex in k for ex in excludes):
                return k

    return None


def resolve_field_keys(keys: Iterable[str]) -> Dict[str, Optional[str]]:
    key_list = list(keys)

    mapping: Dict[str, Optional[str]] = {}
    mapping["公开公告号"] = choose_key(key_list, FIELD_CANDIDATES["公开公告号"])
    mapping["摘要"] = choose_key(key_list, FIELD_CANDIDATES["摘要"])
    mapping["首项权利要求"] = choose_key(key_list, FIELD_CANDIDATES["首项权利要求"])
    mapping["标题"] = choose_key(key_list, FIELD_CANDIDATES["标题"])
    mapping["申请日"] = choose_key(key_list, FIELD_CANDIDATES["申请日"])
    mapping["IPC主分类"] = choose_key(key_list, FIELD_CANDIDATES["IPC主分类"])
    mapping["IPC分类"] = choose_key(key_list, FIELD_CANDIDATES["IPC分类"], exclude=["主分类"])
    mapping["第一申请人"] = choose_key(key_list, FIELD_CANDIDATES["第一申请人"])
    mapping["自引信息"] = choose_key(key_list, FIELD_CANDIDATES["自引信息"])
    mapping["他引信息"] = choose_key(key_list, FIELD_CANDIDATES["他引信息"])
    mapping["引证专利"] = choose_key(key_list, FIELD_CANDIDATES["引证专利"])
    mapping["申请人"] = choose_key(key_list, FIELD_CANDIDATES["申请人"])

    return mapping


def get_value(obj: dict, key: Optional[str]) -> object:
    if key is None:
        return None
    return obj.get(key)


def build_row(
    obj: dict,
    key_map: Dict[str, Optional[str]],
    source_province: str,
    source_file: str,
) -> Tuple[Optional[Tuple[object, ...]], Optional[str]]:
    public_no = normalize_text(get_value(obj, key_map["公开公告号"]))
    abstract = normalize_text(get_value(obj, key_map["摘要"]))
    first_claim = normalize_text(get_value(obj, key_map["首项权利要求"]))
    title = normalize_text(get_value(obj, key_map["标题"]))
    app_date = normalize_date(get_value(obj, key_map["申请日"]))
    ipc_main = normalize_text(get_value(obj, key_map["IPC主分类"]))
    ipc_class = normalize_text(get_value(obj, key_map["IPC分类"]))

    applicants = normalize_text(get_value(obj, key_map["申请人"]))
    first_applicant = normalize_text(get_value(obj, key_map["第一申请人"]))
    if first_applicant is None:
        first_applicant = split_first_applicant(applicants)

    self_cite = normalize_text(get_value(obj, key_map["自引信息"]))
    other_cite = normalize_text(get_value(obj, key_map["他引信息"]))
    cited_patents = normalize_text(get_value(obj, key_map["引证专利"]))

    required_values = {
        "公开公告号": public_no,
        "摘要": abstract,
        "首项权利要求": first_claim,
        "标题": title,
        "申请日": app_date,
        "IPC主分类": ipc_main,
        "IPC分类": ipc_class,
        "第一申请人": first_applicant,
    }
    for k in REQUIRED_FIELDS:
        if required_values[k] is None:
            return None, f"missing_{k}"

    row = (
        public_no,
        abstract,
        first_claim,
        title,
        app_date,
        ipc_main,
        ipc_class,
        first_applicant,
        self_cite,
        other_cite,
        cited_patents,
        applicants,
        source_province,
        source_file,
    )
    return row, None


def valid_table_name(name: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None


def init_db(conn: sqlite3.Connection, table: str, rebuild: bool) -> None:
    if rebuild:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')

    conn.execute(
        f'''
        CREATE TABLE IF NOT EXISTS "{table}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            "公开公告号" TEXT NOT NULL CHECK(length(trim("公开公告号")) > 0),
            "摘要" TEXT NOT NULL CHECK(length(trim("摘要")) > 0),
            "首项权利要求" TEXT NOT NULL CHECK(length(trim("首项权利要求")) > 0),
            "标题" TEXT NOT NULL CHECK(length(trim("标题")) > 0),
            "申请日" TEXT NOT NULL CHECK(length(trim("申请日")) = 10),
            "IPC主分类" TEXT NOT NULL CHECK(length(trim("IPC主分类")) > 0),
            "IPC分类" TEXT NOT NULL CHECK(length(trim("IPC分类")) > 0),
            "第一申请人" TEXT NOT NULL CHECK(length(trim("第一申请人")) > 0),
            "自引信息" TEXT,
            "他引信息" TEXT,
            "引证专利" TEXT,
            "申请人" TEXT,
            "来源省份" TEXT NOT NULL,
            "来源文件" TEXT NOT NULL
        )
        '''
    )

    conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "idx_{table}_pubno" ON "{table}"("公开公告号")')
    conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_app_date" ON "{table}"("申请日")')
    conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table}_ipc_main" ON "{table}"("IPC主分类")')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SQLite database from patent JSON files")
    p.add_argument(
        "--json-root",
        type=Path,
        default=Path("/home/ljh/data1/patent/dta_to_json"),
        help="Root directory that contains province JSON files",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path("/home/ljh/data1/patent/patent.sqlite"),
        help="Output sqlite database path",
    )
    p.add_argument("--table", default="patents", help="SQLite table name")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--progress-files", type=int, default=20)
    p.add_argument("--limit-files", type=int, default=0, help="0 means all files")
    p.add_argument("--rebuild", action="store_true", help="Drop and rebuild table before import")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.json_root.exists():
        print(f"[FATAL] json root not found: {args.json_root}")
        return 2
    if not valid_table_name(args.table):
        print(f"[FATAL] invalid table name: {args.table}")
        return 2

    files = sorted(args.json_root.rglob("*.json"))
    if args.limit_files > 0:
        files = files[: args.limit_files]

    print(f"[{now_text()}] json_root={args.json_root}")
    print(f"[{now_text()}] db_path={args.db_path}")
    print(f"[{now_text()}] table={args.table}")
    print(f"[{now_text()}] json_files={len(files)}")

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")

    init_db(conn, args.table, rebuild=args.rebuild)

    insert_sql = (
        f'INSERT OR IGNORE INTO "{args.table}" '
        '("公开公告号","摘要","首项权利要求","标题","申请日","IPC主分类","IPC分类","第一申请人",'
        '"自引信息","他引信息","引证专利","申请人","来源省份","来源文件") '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
    )

    stats: Dict[str, int] = {
        "rows_seen": 0,
        "inserted": 0,
        "duplicate_pubno": 0,
        "parse_or_non_dict_skipped": 0,
        "files_processed": 0,
        "files_failed": 0,
    }

    missing_counter: Dict[str, int] = {}
    batch: List[Tuple[object, ...]] = []

    try:
        for i, fp in enumerate(files, start=1):
            rel = fp.relative_to(args.json_root)
            province = rel.parts[0] if len(rel.parts) >= 2 else fp.parent.name

            key_map: Optional[Dict[str, Optional[str]]] = None
            file_rows = 0
            file_added = 0
            file_dup = 0

            try:
                for obj in iter_json_objects(fp):
                    stats["rows_seen"] += 1
                    file_rows += 1

                    if key_map is None:
                        key_map = resolve_field_keys(obj.keys())

                    row, reason = build_row(obj, key_map, province, str(rel))
                    if row is None:
                        missing_counter[reason or "unknown"] = missing_counter.get(reason or "unknown", 0) + 1
                        continue

                    batch.append(row)
                    if len(batch) >= args.batch_size:
                        before = conn.total_changes
                        conn.executemany(insert_sql, batch)
                        delta = conn.total_changes - before
                        file_added += delta
                        file_dup += (len(batch) - delta)
                        batch.clear()

                if batch:
                    before = conn.total_changes
                    conn.executemany(insert_sql, batch)
                    delta = conn.total_changes - before
                    file_added += delta
                    file_dup += (len(batch) - delta)
                    batch.clear()

                conn.commit()
                stats["files_processed"] += 1
                stats["inserted"] += file_added
                stats["duplicate_pubno"] += file_dup

                if args.progress_files > 0 and i % args.progress_files == 0:
                    print(
                        f"[{now_text()}] progress files={i}/{len(files)} "
                        f"rows_seen={stats['rows_seen']} inserted={stats['inserted']} dup={stats['duplicate_pubno']}"
                    )

            except Exception:
                conn.rollback()
                stats["files_failed"] += 1
                print(f"[ERROR] file={fp}")
                print(traceback.format_exc())

        conn.commit()

    finally:
        conn.close()

    print("\n[SUMMARY]")
    print(f"files_total={len(files)}")
    print(f"files_processed={stats['files_processed']}")
    print(f"files_failed={stats['files_failed']}")
    print(f"rows_seen={stats['rows_seen']}")
    print(f"inserted={stats['inserted']}")
    print(f"duplicate_pubno={stats['duplicate_pubno']}")

    if missing_counter:
        print("missing_required_top=")
        for k, v in sorted(missing_counter.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {k}: {v}")

    if stats["files_failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
