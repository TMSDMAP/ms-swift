#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import bm25s
import jieba
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PUBNO_SPLIT_RE = re.compile(r"[;；,，|\s]+")
IPC4_RE = re.compile(r"([A-HY]\d{2}[A-Z])")
CATEGORY_LETTER_RE = re.compile(r"[A-Z]")


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def clean_text(x: Optional[str]) -> str:
    return (x or "").replace("\x00", " ").strip()


def normalize_pubno(x: Optional[str]) -> str:
    return re.sub(r"\s+", "", clean_text(x).upper())


def normalize_ipc4(raw: str) -> str:
    s = clean_text(raw).upper()
    if not s:
        return ""
    first = re.split(r"[;；,，|]", s, maxsplit=1)[0]
    first = re.sub(r"\s+", "", first)
    m = IPC4_RE.search(first)
    if m:
        return m.group(1)
    return first[:4]


def parse_pubnos(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for t in PUBNO_SPLIT_RE.split(clean_text(text)):
        x = t.strip().upper()
        if len(x) < 6:
            continue
        if not re.match(r"^[A-Z0-9]+$", x):
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def quote_sql_identifier(name: str) -> str:
    n = clean_text(name)
    if not n:
        raise ValueError("empty SQL identifier")
    if '"' in n or "\x00" in n:
        raise ValueError(f"unsafe SQL identifier: {name!r}")
    return f'"{n}"'


def get_table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    sql = f"PRAGMA table_info({quote_sql_identifier(table)})"
    rows = conn.execute(sql).fetchall()
    return {clean_text(r[1]) for r in rows}


def split_citation_tokens(text: str) -> List[str]:
    return [t.strip() for t in PUBNO_SPLIT_RE.split(clean_text(text)) if t.strip()]


def parse_pubnos_by_category(
    pubno_text: str,
    category_text: str,
    allowed_categories: Set[str],
) -> Tuple[List[str], int, int]:
    pub_tokens = split_citation_tokens(pubno_text)
    cat_tokens = split_citation_tokens(category_text)
    pair_n = min(len(pub_tokens), len(cat_tokens))

    out: List[str] = []
    seen: Set[str] = set()
    for i in range(pair_n):
        pub = normalize_pubno(pub_tokens[i])
        if not pub:
            continue
        cat_letters = set(CATEGORY_LETTER_RE.findall(clean_text(cat_tokens[i]).upper()))
        if not cat_letters.intersection(allowed_categories):
            continue
        if pub in seen:
            continue
        seen.add(pub)
        out.append(pub)

    return out, len(pub_tokens), len(cat_tokens)


def make_input_text(title: str, abstract: str, first_claim: str) -> str:
    return (
        f"标题：{clean_text(title)}\n"
        f"摘要：{clean_text(abstract)}\n"
        f"首项权利要求：{clean_text(first_claim)}"
    )


def tokenize_for_bm25(text: str) -> List[str]:
    toks = [x.strip() for x in jieba.cut(text) if x.strip()]
    if not toks and text:
        toks = [text]
    return toks


def load_excluded_pubnos(path: str) -> Set[str]:
    if not path:
        return set()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"exclude file not found: {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")
    out: Set[str] = set()

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            for item in obj:
                out.add(normalize_pubno(str(item)))
        elif isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    for item in value:
                        out.add(normalize_pubno(str(item)))
                else:
                    out.add(normalize_pubno(str(value)))
    except json.JSONDecodeError:
        for line in text.splitlines():
            out.add(normalize_pubno(line))

    out.discard("")
    return out


@dataclass
class PatentText:
    pubno: str
    title: str
    abstract: str
    first_claim: str
    applicant: str
    year: int
    ipc4: str

    def as_input_text(self) -> str:
        return make_input_text(self.title, self.abstract, self.first_claim)


@dataclass
class QueryRaw:
    query_pubno: str
    query_year: int
    query_ipc4: str
    title: str
    abstract: str
    first_claim: str
    self_cites_raw: List[str]
    other_cites_raw: List[str]

    def as_input_text(self) -> str:
        return make_input_text(self.title, self.abstract, self.first_claim)


@dataclass
class QueryFinal:
    query_pubno: str
    query_year: int
    query_ipc4: str
    title: str
    abstract: str
    first_claim: str
    positive_pubnos: List[str]
    positive_from_self: int
    positive_from_other: int
    bm25_hard_neg_pubnos: List[str]
    ipc_semantic_neg_pubnos: List[str]
    ipc_random_neg_pubnos: List[str]
    cross_ipc_hard_neg_pubnos: List[str]

    def as_input_text(self) -> str:
        return make_input_text(self.title, self.abstract, self.first_claim)


class HFEmbedder:
    def __init__(self, model_path: str, device: str, max_length: int, batch_size: int) -> None:
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        log(f"Loading tokenizer/model from: {model_path}")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        log(f"Model ready on device: {self.device}")

    @torch.inference_mode()
    def encode_batch(self, batch: Sequence[str]) -> np.ndarray:
        if not batch:
            return np.zeros((0, 1), dtype=np.float32)

        tokens = self.tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.model(**tokens, return_dict=True)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            denom = torch.clamp(mask.sum(dim=1), min=1)
            emb = (hidden * mask).sum(dim=1) / denom

        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        log(f"Encoding {len(texts)} texts (batch_size={self.batch_size}, max_length={self.max_length})")
        out: List[np.ndarray] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        report_step = max(1, total_batches // 50)  # up to ~50 progress lines per encode call
        next_report_at = 1

        for batch_idx, start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            out.append(self.encode_batch(batch))

            if batch_idx == next_report_at or batch_idx == total_batches:
                pct = 100.0 * batch_idx / max(1, total_batches)
                log(f"Encoding progress: {batch_idx}/{total_batches} batches ({pct:.1f}%)")
                next_report_at = batch_idx + report_step

        return np.concatenate(out, axis=0)


def encode_with_resume_cache(
    embedder: HFEmbedder,
    texts: Sequence[str],
    cache_dir: Optional[Path],
    cache_key: str,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if cache_dir is None:
        return embedder.encode(texts)

    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{cache_key}.meta.json"
    data_path = cache_dir / f"{cache_key}.f32memmap"

    total_texts = len(texts)
    batch_size = max(1, embedder.batch_size)
    total_batches = (total_texts + batch_size - 1) // batch_size

    start_batch = 0
    emb_dim = 0
    meta_ok = False

    if meta_path.exists() and data_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_ok = (
                int(meta.get("text_count", -1)) == total_texts
                and int(meta.get("batch_size", -1)) == batch_size
                and int(meta.get("max_length", -1)) == int(embedder.max_length)
            )
            if meta_ok:
                start_batch = int(meta.get("completed_batches", 0))
                start_batch = max(0, min(start_batch, total_batches))
                emb_dim = int(meta.get("embedding_dim", 0))
        except Exception:
            meta_ok = False

    if not meta_ok:
        start_batch = 0
        emb_dim = 0
        if meta_path.exists():
            meta_path.unlink(missing_ok=True)
        if data_path.exists():
            data_path.unlink(missing_ok=True)

    mem: Optional[np.memmap] = None
    if emb_dim > 0 and data_path.exists():
        try:
            mode = "r" if start_batch >= total_batches else "r+"
            mem = np.memmap(data_path, dtype=np.float32, mode=mode, shape=(total_texts, emb_dim))
        except Exception:
            start_batch = 0
            emb_dim = 0
            mem = None
            if meta_path.exists():
                meta_path.unlink(missing_ok=True)
            if data_path.exists():
                data_path.unlink(missing_ok=True)

    if start_batch >= total_batches and mem is not None:
        log(
            f"Resume mode: loading cached {cache_key} embeddings from {data_path} "
            f"(texts={total_texts}, batches={total_batches})"
        )
        return np.asarray(mem)

    log(f"Encoding {total_texts} texts (batch_size={batch_size}, max_length={embedder.max_length})")
    if start_batch > 0:
        log(f"Resume mode: continuing {cache_key} encoding from batch {start_batch + 1}/{total_batches}")

    report_step = max(1, total_batches // 50)
    next_report_at = start_batch + 1

    for batch_no in range(start_batch + 1, total_batches + 1):
        start = (batch_no - 1) * batch_size
        batch = texts[start : start + batch_size]
        emb = embedder.encode_batch(batch)

        if emb_dim <= 0:
            emb_dim = int(emb.shape[1])
            mem = np.memmap(data_path, dtype=np.float32, mode="w+", shape=(total_texts, emb_dim))
        elif emb.shape[1] != emb_dim:
            raise RuntimeError(
                f"embedding dim changed during {cache_key} encoding: {emb.shape[1]} vs {emb_dim}"
            )
        elif mem is None:
            mem = np.memmap(data_path, dtype=np.float32, mode="r+", shape=(total_texts, emb_dim))

        mem[start : start + len(batch)] = emb
        mem.flush()

        meta_obj = {
            "cache_key": cache_key,
            "text_count": total_texts,
            "batch_size": batch_size,
            "max_length": int(embedder.max_length),
            "embedding_dim": emb_dim,
            "completed_batches": batch_no,
            "total_batches": total_batches,
            "complete": bool(batch_no >= total_batches),
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        if batch_no >= next_report_at or batch_no == total_batches:
            pct = 100.0 * batch_no / max(1, total_batches)
            log(f"Encoding progress: {batch_no}/{total_batches} batches ({pct:.1f}%)")
            next_report_at = batch_no + report_step

    assert mem is not None
    return np.asarray(mem)


def valid_text_filter_sql() -> str:
    return (
        'ifnull("公开公告号","")<>"" '
        'AND ifnull("标题","")<>"" '
        'AND ifnull("摘要","")<>"" '
        'AND ifnull("首项权利要求","")<>"" '
        'AND substr("申请日",1,4) GLOB "[0-9][0-9][0-9][0-9]" '
    )


def row_to_patent_text(row: sqlite3.Row) -> PatentText:
    return PatentText(
        pubno=normalize_pubno(row["公开公告号"]),
        title=clean_text(row["标题"]),
        abstract=clean_text(row["摘要"]),
        first_claim=clean_text(row["首项权利要求"]),
        applicant=clean_text(row["第一申请人"]),
        year=int(clean_text(row["申请日"])[:4]),
        ipc4=normalize_ipc4(clean_text(row["IPC主分类"])),
    )


def reservoir_add(res: List[PatentText], item: PatentText, seen: int, cap: int, rng: random.Random) -> int:
    seen += 1
    if cap <= 0:
        return seen
    if len(res) < cap:
        res.append(item)
    else:
        j = rng.randrange(seen)
        if j < cap:
            res[j] = item
    return seen


def pick_random(rng: random.Random, seq: Sequence[str], k: int) -> List[str]:
    if k <= 0:
        return []
    if k >= len(seq):
        return list(seq)
    idx = rng.sample(range(len(seq)), k)
    return [seq[i] for i in idx]


def build_query_raw_pool(
    conn: sqlite3.Connection,
    table: str,
    query_year: int,
    query_size: int,
    query_pool_multiplier: int,
    max_query_scan: int,
    excluded_pubnos: Set[str],
    sql_fetch_batch: int,
    xy_citation_only: bool,
    citation_pubno_field: str,
    citation_category_field: str,
    xy_allowed_categories: Set[str],
) -> List[QueryRaw]:
    if xy_citation_only:
        table_cols = get_table_columns(conn, table)
        missing_cols = [
            c
            for c in [citation_pubno_field, citation_category_field]
            if c not in table_cols
        ]
        if missing_cols:
            preview_cols = sorted(list(table_cols))[:30]
            raise ValueError(
                "XY mode requested citation columns that are not present in table "
                f"{table!r}: missing={missing_cols}. "
                f"Available columns (first 30)={preview_cols}"
            )

        pub_col = quote_sql_identifier(citation_pubno_field)
        cat_col = quote_sql_identifier(citation_category_field)
        table_ident = quote_sql_identifier(table)

        sql = (
            f'SELECT id, "公开公告号", "标题", "摘要", "首项权利要求", "申请日", "IPC主分类", '
            f'{pub_col} AS "__citation_pubnos__", {cat_col} AS "__citation_cats__" '
            f'FROM {table_ident} '
            f'WHERE {valid_text_filter_sql()} '
            'AND substr("申请日",1,4)=? '
            'ORDER BY id'
        )
    else:
        table_ident = quote_sql_identifier(table)
        sql = (
            f'SELECT id, "公开公告号", "标题", "摘要", "首项权利要求", "申请日", "IPC主分类", "自引信息", "他引信息" '
            f'FROM {table_ident} '
            f'WHERE {valid_text_filter_sql()} '
            'AND substr("申请日",1,4)=? '
            'ORDER BY id'
        )

    target_pool = max(query_size, query_size * max(1, query_pool_multiplier))
    out: List[QueryRaw] = []
    seen_query: Set[str] = set()
    xy_mismatch_rows = 0
    xy_total_pub_tokens = 0
    xy_total_cat_tokens = 0
    xy_kept_rows = 0
    xy_rows_with_allowed = 0

    scanned = 0
    cur = conn.execute(sql, (str(query_year),))
    while True:
        rows = cur.fetchmany(sql_fetch_batch)
        if not rows:
            break
        for row in rows:
            scanned += 1
            if max_query_scan > 0 and scanned > max_query_scan:
                break

            qpub = normalize_pubno(row["公开公告号"])
            if not qpub or qpub in excluded_pubnos or qpub in seen_query:
                continue

            if xy_citation_only:
                xy_raw, pub_tok_n, cat_tok_n = parse_pubnos_by_category(
                    pubno_text=clean_text(row["__citation_pubnos__"]),
                    category_text=clean_text(row["__citation_cats__"]),
                    allowed_categories=xy_allowed_categories,
                )
                xy_total_pub_tokens += pub_tok_n
                xy_total_cat_tokens += cat_tok_n
                if pub_tok_n != cat_tok_n:
                    xy_mismatch_rows += 1
                self_raw = [x for x in xy_raw if x and x != qpub and x not in excluded_pubnos]
                other_raw: List[str] = []
                if xy_raw:
                    xy_rows_with_allowed += 1
                if self_raw:
                    xy_kept_rows += 1
            else:
                self_raw = [normalize_pubno(x) for x in parse_pubnos(clean_text(row["自引信息"]))]
                other_raw = [normalize_pubno(x) for x in parse_pubnos(clean_text(row["他引信息"]))]

                self_raw = [x for x in self_raw if x and x != qpub and x not in excluded_pubnos]
                other_raw = [x for x in other_raw if x and x != qpub and x not in excluded_pubnos]
            if not self_raw and not other_raw:
                continue

            q = QueryRaw(
                query_pubno=qpub,
                query_year=query_year,
                query_ipc4=normalize_ipc4(clean_text(row["IPC主分类"])),
                title=clean_text(row["标题"]),
                abstract=clean_text(row["摘要"]),
                first_claim=clean_text(row["首项权利要求"]),
                self_cites_raw=self_raw,
                other_cites_raw=other_raw,
            )
            out.append(q)
            seen_query.add(qpub)

            if len(out) >= target_pool:
                break

        if (max_query_scan > 0 and scanned > max_query_scan) or len(out) >= target_pool:
            break

    if xy_citation_only:
        log(
            f"Query raw pool (XY mode) stats: mismatch_rows={xy_mismatch_rows}, "
            f"total_pub_tokens={xy_total_pub_tokens}, total_cat_tokens={xy_total_cat_tokens}, "
            f"kept_with_xy={xy_kept_rows}, allowed_categories={sorted(xy_allowed_categories)}"
        )
        if xy_rows_with_allowed == 0:
            raise RuntimeError(
                "XY mode found zero citations matching allowed categories in scanned query rows. "
                "Please verify citation category data quality/column mapping or provide another data source."
            )

    log(f"Query raw pool done: scanned={scanned}, pool_size={len(out)}, target={target_pool}")
    return out


def fetch_query_row_by_pubno(conn: sqlite3.Connection, table: str, pubno: str) -> Optional[sqlite3.Row]:
    sql = (
        f'SELECT "公开公告号", "标题", "摘要", "首项权利要求", "申请日", "IPC主分类" '
        f'FROM "{table}" '
        f'WHERE {valid_text_filter_sql()} '
        'AND "公开公告号"=? '
        'LIMIT 1'
    )
    return conn.execute(sql, (pubno,)).fetchone()


def build_query_raw_pool_from_citation_csv(
    conn: sqlite3.Connection,
    table: str,
    citation_csv: str,
    query_size: int,
    query_pool_multiplier: int,
    excluded_pubnos: Set[str],
    rng: random.Random,
) -> List[QueryRaw]:
    p = Path(citation_csv)
    if not p.exists():
        raise FileNotFoundError(f"citation csv not found: {p}")

    target_pool = max(query_size, query_size * max(1, query_pool_multiplier))

    anchor_self: Dict[str, Set[str]] = {}
    anchor_other: Dict[str, Set[str]] = {}

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = normalize_pubno(row.get("anchor_pubno", ""))
            s = normalize_pubno(row.get("self_pubno", ""))
            o = normalize_pubno(row.get("other_pubno", ""))

            if not a or a in excluded_pubnos:
                continue

            anchor_self.setdefault(a, set())
            anchor_other.setdefault(a, set())

            if s and s != a and s not in excluded_pubnos:
                anchor_self[a].add(s)
            if o and o != a and o not in excluded_pubnos:
                anchor_other[a].add(o)

    anchors = [a for a in anchor_self.keys() if anchor_self[a] or anchor_other[a]]
    if len(anchors) > target_pool:
        anchors = rng.sample(anchors, target_pool)

    out: List[QueryRaw] = []
    missing_anchor = 0

    for a in anchors:
        row = fetch_query_row_by_pubno(conn, table, a)
        if row is None:
            missing_anchor += 1
            continue

        year_txt = clean_text(row["申请日"])[:4]
        if not year_txt.isdigit():
            continue

        q = QueryRaw(
            query_pubno=normalize_pubno(row["公开公告号"]),
            query_year=int(year_txt),
            query_ipc4=normalize_ipc4(clean_text(row["IPC主分类"])),
            title=clean_text(row["标题"]),
            abstract=clean_text(row["摘要"]),
            first_claim=clean_text(row["首项权利要求"]),
            self_cites_raw=sorted(anchor_self.get(a, set())),
            other_cites_raw=sorted(anchor_other.get(a, set())),
        )

        if not q.self_cites_raw and not q.other_cites_raw:
            continue

        out.append(q)

    log(
        f"Query raw pool (citation csv) done: anchors_in_csv={len(anchor_self)}, kept={len(out)}, "
        f"target={target_pool}, missing_anchor_in_db={missing_anchor}"
    )
    return out


def scan_window_collect(
    conn: sqlite3.Connection,
    table: str,
    start_year: int,
    end_year: int,
    needed_positive_pubnos: Set[str],
    target_ipc_set: Set[str],
    excluded_pubnos: Set[str],
    bm25_pool_per_ipc: int,
    global_bm25_pool_size: int,
    rng: random.Random,
    sql_fetch_batch: int,
) -> Tuple[Dict[str, PatentText], Dict[str, List[PatentText]], List[PatentText]]:
    sql = (
        f'SELECT "公开公告号", "标题", "摘要", "首项权利要求", "第一申请人", "申请日", "IPC主分类" '
        f'FROM "{table}" '
        f'WHERE {valid_text_filter_sql()} '
        'AND substr("申请日",1,4) BETWEEN ? AND ? '
    )

    positive_map: Dict[str, PatentText] = {}
    ipc_reservoir: Dict[str, List[PatentText]] = {ipc: [] for ipc in target_ipc_set}
    ipc_seen_count: Dict[str, int] = {ipc: 0 for ipc in target_ipc_set}
    global_reservoir: List[PatentText] = []
    global_seen = 0

    scanned = 0
    cur = conn.execute(sql, (str(start_year), str(end_year)))
    while True:
        rows = cur.fetchmany(sql_fetch_batch)
        if not rows:
            break
        for row in rows:
            scanned += 1
            rec = row_to_patent_text(row)
            if not rec.pubno:
                continue
            if rec.pubno in excluded_pubnos:
                continue

            if rec.pubno in needed_positive_pubnos and rec.pubno not in positive_map:
                positive_map[rec.pubno] = rec

            ipc = rec.ipc4
            if ipc in target_ipc_set:
                seen = ipc_seen_count[ipc]
                seen = reservoir_add(ipc_reservoir[ipc], rec, seen, bm25_pool_per_ipc, rng)
                ipc_seen_count[ipc] = seen

            if global_bm25_pool_size > 0:
                global_seen = reservoir_add(global_reservoir, rec, global_seen, global_bm25_pool_size, rng)

            if scanned % 500000 == 0:
                found = len(positive_map)
                log(f"[window scan] scanned={scanned}, positive_found={found}")

    log(
        f"Window scan done: scanned={scanned}, positive_found={len(positive_map)}, "
        f"ipc_pools={len(ipc_reservoir)}, global_pool={len(global_reservoir)}"
    )
    return positive_map, ipc_reservoir, global_reservoir


def bm25_rank_indices(bm25, query_tokens: List[str], top_k: int) -> List[int]:
    try:
        results, _ = bm25.retrieve([query_tokens], k=top_k, show_progress=False)
        raw = results[0]
    except Exception:
        return []

    out: List[int] = []
    for item in raw:
        try:
            if isinstance(item, dict) and "id" in item:
                idx = int(item["id"])
            else:
                idx = int(item)
            out.append(idx)
        except Exception:
            continue
    return out


def unique_docs_by_pubno(docs: Sequence[PatentText]) -> List[PatentText]:
    out: List[PatentText] = []
    seen: Set[str] = set()
    for d in docs:
        if not d.pubno or d.pubno in seen:
            continue
        seen.add(d.pubno)
        out.append(d)
    return out


def build_final_queries(
    query_raw_pool: Sequence[QueryRaw],
    positive_map: Dict[str, PatentText],
    ipc_reservoir: Dict[str, List[PatentText]],
    global_docs: Sequence[PatentText],
    query_size: int,
    hard_neg_per_query: int,
    ipc_semantic_per_query: int,
    ipc_random_per_query: int,
    cross_ipc_hard_per_query: int,
    bm25_top_k: int,
    cross_ipc_bm25_top_k: int,
    rng: random.Random,
    allow_partial_final_queries: bool,
    min_final_queries: int,
) -> List[QueryFinal]:
    # Build BM25 index for each IPC pool once.
    ipc_bm25_state: Dict[str, Dict[str, object]] = {}
    for ipc, docs in ipc_reservoir.items():
        if not docs:
            continue
        corpus = [d.as_input_text() for d in docs]
        tokenized = [tokenize_for_bm25(x) for x in corpus]
        bm25 = bm25s.BM25()
        try:
            bm25.index(tokenized, show_progress=False)
        except TypeError:
            bm25.index(tokenized)
        ipc_bm25_state[ipc] = {
            "docs": docs,
            "bm25": bm25,
            "tokens": tokenized,
        }

    global_bm25_state: Optional[Dict[str, object]] = None
    if cross_ipc_hard_per_query > 0:
        global_docs_u = unique_docs_by_pubno(global_docs)
        if not global_docs_u:
            raise RuntimeError(
                "cross_ipc_hard_per_query > 0 but global_docs is empty; "
                "increase global_bm25_pool_size"
            )
        global_corpus = [d.as_input_text() for d in global_docs_u]
        global_tokens = [tokenize_for_bm25(x) for x in global_corpus]
        global_bm25 = bm25s.BM25()
        try:
            global_bm25.index(global_tokens, show_progress=False)
        except TypeError:
            global_bm25.index(global_tokens)
        global_bm25_state = {
            "docs": global_docs_u,
            "bm25": global_bm25,
            "tokens": global_tokens,
        }
        log(f"Global BM25 ready for cross-IPC negatives: docs={len(global_docs_u)}")

    final: List[QueryFinal] = []

    # Shuffle once for unbiased picking.
    pool = list(query_raw_pool)
    rng.shuffle(pool)

    for q in pool:
        if len(final) >= query_size:
            break

        ipc_state = ipc_bm25_state.get(q.query_ipc4)
        if ipc_state is None:
            continue

        positive_set: Set[str] = set()
        pos_self = 0
        pos_other = 0

        for p in q.self_cites_raw:
            if p in positive_map:
                if p not in positive_set:
                    positive_set.add(p)
                    pos_self += 1

        for p in q.other_cites_raw:
            if p in positive_map:
                if p not in positive_set:
                    positive_set.add(p)
                    pos_other += 1

        if not positive_set:
            continue

        docs: List[PatentText] = ipc_state["docs"]  # type: ignore[assignment]
        bm25 = ipc_state["bm25"]

        forbid = set(positive_set)
        forbid.add(q.query_pubno)

        q_tokens = tokenize_for_bm25(q.as_input_text())
        ranked_idx = bm25_rank_indices(bm25, q_tokens, max(hard_neg_per_query * 12, bm25_top_k))

        hard_pub: List[str] = []
        hard_seen: Set[str] = set()
        for idx in ranked_idx:
            if idx < 0 or idx >= len(docs):
                continue
            p = docs[idx].pubno
            if (not p) or (p in forbid) or (p in hard_seen):
                continue
            hard_seen.add(p)
            hard_pub.append(p)
            if len(hard_pub) >= hard_neg_per_query:
                break

        if len(hard_pub) < hard_neg_per_query:
            fallback_pool = [d.pubno for d in docs if d.pubno and d.pubno not in forbid and d.pubno not in hard_seen]
            need = hard_neg_per_query - len(hard_pub)
            extra = pick_random(rng, fallback_pool, min(need, len(fallback_pool)))
            hard_pub.extend(extra)
            hard_seen.update(extra)

        if len(hard_pub) < hard_neg_per_query:
            continue

        semantic_pub: List[str] = []
        semantic_seen: Set[str] = set()
        if ipc_semantic_per_query > 0:
            for idx in ranked_idx:
                if idx < 0 or idx >= len(docs):
                    continue
                p = docs[idx].pubno
                if (not p) or (p in forbid) or (p in hard_seen) or (p in semantic_seen):
                    continue
                semantic_seen.add(p)
                semantic_pub.append(p)
                if len(semantic_pub) >= ipc_semantic_per_query:
                    break

            if len(semantic_pub) < ipc_semantic_per_query:
                semantic_pool = [
                    d.pubno
                    for d in docs
                    if d.pubno and d.pubno not in forbid and d.pubno not in hard_seen and d.pubno not in semantic_seen
                ]
                need = ipc_semantic_per_query - len(semantic_pub)
                extra = pick_random(rng, semantic_pool, min(need, len(semantic_pool)))
                semantic_pub.extend(extra)
                semantic_seen.update(extra)

            if len(semantic_pub) < ipc_semantic_per_query:
                continue

        rand_pool = [
            d.pubno
            for d in docs
            if d.pubno and d.pubno not in forbid and d.pubno not in hard_seen and d.pubno not in semantic_seen
        ]
        rand_pub = pick_random(rng, rand_pool, ipc_random_per_query)
        if len(rand_pub) < ipc_random_per_query:
            continue
        rand_seen = set(rand_pub)

        cross_pub: List[str] = []
        cross_seen: Set[str] = set()
        if cross_ipc_hard_per_query > 0:
            if global_bm25_state is None:
                raise RuntimeError("global_bm25_state is None while cross_ipc_hard_per_query > 0")

            global_docs_u: List[PatentText] = global_bm25_state["docs"]  # type: ignore[assignment]
            global_bm25 = global_bm25_state["bm25"]
            ranked_global_idx = bm25_rank_indices(
                global_bm25,
                q_tokens,
                max(cross_ipc_hard_per_query * 24, cross_ipc_bm25_top_k),
            )

            for idx in ranked_global_idx:
                if idx < 0 or idx >= len(global_docs_u):
                    continue
                rec = global_docs_u[idx]
                p = rec.pubno
                if (
                    (not p)
                    or (p in forbid)
                    or (p in hard_seen)
                    or (p in semantic_seen)
                    or (p in rand_seen)
                    or (p in cross_seen)
                ):
                    continue
                if rec.ipc4 == q.query_ipc4:
                    continue
                cross_seen.add(p)
                cross_pub.append(p)
                if len(cross_pub) >= cross_ipc_hard_per_query:
                    break

            if len(cross_pub) < cross_ipc_hard_per_query:
                cross_pool = [
                    d.pubno
                    for d in global_docs_u
                    if d.pubno
                    and d.ipc4 != q.query_ipc4
                    and d.pubno not in forbid
                    and d.pubno not in hard_seen
                    and d.pubno not in semantic_seen
                    and d.pubno not in rand_seen
                    and d.pubno not in cross_seen
                ]
                need = cross_ipc_hard_per_query - len(cross_pub)
                extra = pick_random(rng, cross_pool, min(need, len(cross_pool)))
                cross_pub.extend(extra)
                cross_seen.update(extra)

            if len(cross_pub) < cross_ipc_hard_per_query:
                continue

        qf = QueryFinal(
            query_pubno=q.query_pubno,
            query_year=q.query_year,
            query_ipc4=q.query_ipc4,
            title=q.title,
            abstract=q.abstract,
            first_claim=q.first_claim,
            positive_pubnos=sorted(positive_set),
            positive_from_self=pos_self,
            positive_from_other=pos_other,
            bm25_hard_neg_pubnos=hard_pub,
            ipc_semantic_neg_pubnos=semantic_pub,
            ipc_random_neg_pubnos=rand_pub,
            cross_ipc_hard_neg_pubnos=cross_pub,
        )
        final.append(qf)

    if len(final) < query_size:
        if not allow_partial_final_queries:
            raise RuntimeError(
                f"Not enough finalized queries: need {query_size}, got {len(final)}. "
                "Increase query_pool_multiplier/max_query_scan or reduce negative counts."
            )

        min_need = max(1, int(min_final_queries)) if min_final_queries > 0 else 1
        if len(final) < min_need:
            raise RuntimeError(
                f"Not enough finalized queries even with partial mode: "
                f"target={query_size}, got={len(final)}, min_required={min_need}. "
                "Increase window/query pool or reduce negative constraints."
            )

        log(
            "WARNING: finalized queries below target but partial mode enabled: "
            f"target={query_size}, got={len(final)}, min_required={min_need}."
        )

    return final


def build_global_candidate_map(
    final_queries: Sequence[QueryFinal],
    positive_map: Dict[str, PatentText],
    ipc_reservoir: Dict[str, List[PatentText]],
    global_docs: Sequence[PatentText],
) -> Dict[str, PatentText]:
    need: Set[str] = set()
    for q in final_queries:
        need.update(q.positive_pubnos)
        need.update(q.bm25_hard_neg_pubnos)
        need.update(q.ipc_semantic_neg_pubnos)
        need.update(q.ipc_random_neg_pubnos)
        need.update(q.cross_ipc_hard_neg_pubnos)

    out: Dict[str, PatentText] = {}

    for p, rec in positive_map.items():
        if p in need:
            out[p] = rec

    for docs in ipc_reservoir.values():
        for rec in docs:
            if rec.pubno in need and rec.pubno not in out:
                out[rec.pubno] = rec

    for rec in global_docs:
        if rec.pubno in need and rec.pubno not in out:
            out[rec.pubno] = rec

    missing = [p for p in need if p not in out]
    if missing:
        raise RuntimeError(f"Missing candidate docs for {len(missing)} pubnos")

    return out


def calc_single_query_metrics(
    ranked_pubnos: Sequence[str],
    positive_set: Set[str],
    k_list: Sequence[int],
) -> Dict[str, float]:
    rel = np.asarray([1 if p in positive_set else 0 for p in ranked_pubnos], dtype=np.int32)
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2)) if len(rel) > 0 else np.asarray([], dtype=np.float64)
    pos_idx = np.where(rel == 1)[0]
    if pos_idx.size == 0:
        # Should not happen; keep safe default.
        first_rank = float(len(ranked_pubnos) + 1)
        rr = 0.0
        ap = 0.0
        ndcg = 0.0
    else:
        first_rank = float(pos_idx[0] + 1)
        rr = 1.0 / first_rank

        cumsum = np.cumsum(rel)
        precision_at_rel = cumsum[pos_idx] / (pos_idx + 1)
        ap = float(np.mean(precision_at_rel))

        dcg = float(np.sum(rel * discounts))
        ideal_rel = np.zeros_like(rel)
        ideal_rel[: len(pos_idx)] = 1
        idcg = float(np.sum(ideal_rel * discounts))
        ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    out: Dict[str, float] = {
        "first_rel_rank": first_rank,
        "rr": rr,
        "ap": ap,
        "ndcg": ndcg,
    }

    pos_count = max(1, len(positive_set))
    for k in k_list:
        top_rel = int(np.sum(rel[:k]))
        out[f"hit@{k}"] = 1.0 if top_rel > 0 else 0.0
        out[f"recall@{k}"] = float(top_rel / pos_count)

        if len(rel) == 0 or pos_idx.size == 0:
            out[f"ndcg@{k}"] = 0.0
            continue

        k_eff = min(k, len(rel))
        dcg_k = float(np.sum(rel[:k_eff] * discounts[:k_eff]))
        ideal_rel_k = np.zeros(k_eff, dtype=np.int32)
        ideal_rel_k[: min(len(pos_idx), k_eff)] = 1
        idcg_k = float(np.sum(ideal_rel_k * discounts[:k_eff]))
        out[f"ndcg@{k}"] = float(dcg_k / idcg_k) if idcg_k > 0 else 0.0

    return out


def aggregate_metrics(per_query_metrics: Sequence[Dict[str, float]], k_list: Sequence[int]) -> Dict[str, float]:
    if not per_query_metrics:
        raise ValueError("empty per_query_metrics")

    def mean_key(k: str) -> float:
        return float(np.mean([m[k] for m in per_query_metrics]))

    out = {
        "query_count": float(len(per_query_metrics)),
        "mean_rank": mean_key("first_rel_rank"),
        "mrr": mean_key("rr"),
        "map": mean_key("ap"),
        "ndcg": mean_key("ndcg"),
    }

    for k in k_list:
        out[f"hit@{k}"] = mean_key(f"hit@{k}")
        out[f"recall@{k}"] = mean_key(f"recall@{k}")
        out[f"ndcg@{k}"] = mean_key(f"ndcg@{k}")

    return out


def sign_test_pvalue_approx(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    z = (abs(wins - losses) - 1.0) / np.sqrt(n)
    cdf = 0.5 * (1.0 + math.erf(abs(float(z)) / np.sqrt(2.0)))
    p = 2.0 * (1.0 - cdf)
    return float(max(0.0, min(1.0, p)))


def evaluate_model(
    embedder: HFEmbedder,
    final_queries: Sequence[QueryFinal],
    candidate_map: Dict[str, PatentText],
    rank_chunk_size: int,
    k_list: Sequence[int],
    embed_cache_dir: Optional[Path] = None,
) -> Dict[str, object]:
    # Encode global candidate universe once.
    candidate_pubnos = sorted(candidate_map.keys())
    cand_texts = [candidate_map[p].as_input_text() for p in candidate_pubnos]
    cand_index = {p: i for i, p in enumerate(candidate_pubnos)}

    t0 = time.time()
    cand_emb = encode_with_resume_cache(embedder, cand_texts, embed_cache_dir, "candidate")
    t1 = time.time()

    query_texts = [q.as_input_text() for q in final_queries]
    query_emb = encode_with_resume_cache(embedder, query_texts, embed_cache_dir, "query")
    t2 = time.time()

    device = embedder.device
    cand_t = torch.from_numpy(cand_emb).to(device)

    per_query = []

    for start in range(0, len(final_queries), max(1, rank_chunk_size)):
        end = min(len(final_queries), start + max(1, rank_chunk_size))

        q_chunk = torch.from_numpy(query_emb[start:end]).to(device)
        for local_i, q_vec in enumerate(q_chunk):
            q = final_queries[start + local_i]
            cand_list = []
            cand_list.extend(q.positive_pubnos)
            cand_list.extend(q.bm25_hard_neg_pubnos)
            cand_list.extend(q.ipc_semantic_neg_pubnos)
            cand_list.extend(q.ipc_random_neg_pubnos)
            cand_list.extend(q.cross_ipc_hard_neg_pubnos)

            # Dedup preserving order.
            seen: Set[str] = set()
            cand_list_u = []
            for p in cand_list:
                if p in seen:
                    continue
                seen.add(p)
                cand_list_u.append(p)

            idxs = [cand_index[p] for p in cand_list_u]
            sims = torch.mv(cand_t[idxs], q_vec)
            order = torch.argsort(sims, descending=True)

            ranked_pub = [cand_list_u[int(i)] for i in order.detach().cpu().numpy().tolist()]
            positive_set = set(q.positive_pubnos)
            qm = calc_single_query_metrics(ranked_pub, positive_set, k_list)
            qm_row = {
                "query_pubno": q.query_pubno,
                "query_ipc4": q.query_ipc4,
                "positive_count": len(q.positive_pubnos),
                "candidate_count": len(cand_list_u),
                "top1_pubno": ranked_pub[0] if ranked_pub else "",
                "top1_is_positive": 1 if ranked_pub and ranked_pub[0] in positive_set else 0,
                **qm,
            }
            per_query.append(qm_row)

    summary_metrics = aggregate_metrics(per_query, k_list)

    return {
        "candidate_universe_size": len(candidate_pubnos),
        "per_query": per_query,
        "metrics": summary_metrics,
        "timing_seconds": {
            "encode_candidates": float(t1 - t0),
            "encode_queries": float(t2 - t1),
            "total": float(time.time() - t0),
        },
    }


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_queries_jsonl(path: Path) -> List[QueryFinal]:
    out: List[QueryFinal] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                out.append(QueryFinal(**obj))
            except Exception as e:
                raise RuntimeError(f"failed to parse query jsonl at {path}:{line_no}") from e
    if not out:
        raise RuntimeError(f"queries file is empty: {path}")
    return out


def load_candidate_map_jsonl(path: Path) -> Dict[str, PatentText]:
    out: Dict[str, PatentText] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                rec = PatentText(**obj)
            except Exception as e:
                raise RuntimeError(f"failed to parse candidate jsonl at {path}:{line_no}") from e
            if not rec.pubno:
                continue
            out[rec.pubno] = rec
    if not out:
        raise RuntimeError(f"candidates file is empty: {path}")
    return out


def validate_resume_dataset_meta(
    args: argparse.Namespace,
    dataset_meta: Dict[str, Any],
    start_year: int,
    end_year: int,
) -> None:
    mismatches: List[str] = []

    meta_query_size_target = dataset_meta.get("query_size_target")
    if meta_query_size_target is None and not args.allow_partial_final_queries:
        meta_query_size_target = dataset_meta.get("query_size")
    if meta_query_size_target is not None and meta_query_size_target != args.query_size:
        mismatches.append(
            f"query_size_target: meta={meta_query_size_target!r}, args={args.query_size!r}"
        )

    scalar_checks = {
        "query_year": args.query_year,
        "query_source_citation_csv": args.query_source_citation_csv,
        "xy_citation_only": args.xy_citation_only,
        "allow_partial_final_queries": args.allow_partial_final_queries,
        "min_final_queries": args.min_final_queries,
        "hard_neg_per_query": args.hard_neg_per_query,
        "ipc_semantic_per_query": args.ipc_semantic_per_query,
        "ipc_random_per_query": args.ipc_random_per_query,
        "cross_ipc_hard_per_query": args.cross_ipc_hard_per_query,
        "bm25_top_k": args.bm25_top_k,
        "cross_ipc_bm25_top_k": args.cross_ipc_bm25_top_k,
        "bm25_pool_per_ipc": args.bm25_pool_per_ipc,
        "global_bm25_pool_size": args.global_bm25_pool_size,
        "exclude_pubno_file": args.exclude_pubno_file,
    }

    for k, expected in scalar_checks.items():
        if k not in dataset_meta:
            continue
        if dataset_meta.get(k) != expected:
            mismatches.append(f"{k}: meta={dataset_meta.get(k)!r}, args={expected!r}")

    win = dataset_meta.get("candidate_window")
    expected_win = [start_year, end_year]
    if isinstance(win, list) and len(win) == 2 and win != expected_win:
        mismatches.append(f"candidate_window: meta={win!r}, args={expected_win!r}")

    if mismatches:
        detail = "\n  - ".join(mismatches)
        raise ValueError(
            "resume requested but existing dataset_meta does not match current arguments:\n"
            f"  - {detail}\n"
            "Please use the same run args or disable --resume."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-k citation retrieval (self/other cites as positives) with BM25 hard negatives + same-IPC random negatives."
    )
    p.add_argument("--db-path", type=str, default="/home/ljh/data1/patent/patent.sqlite")
    p.add_argument("--table", type=str, default="patents")
    p.add_argument("--finetuned-model-path", type=str, required=True)
    p.add_argument("--base-model-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="/home/ljh/data1/patent/eval_outputs")
    p.add_argument("--run-name", type=str, default="")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing run artifacts in output_dir/run_name. "
        "Requires existing queries.jsonl and candidates_global.jsonl.",
    )

    p.add_argument("--query-year", type=int, default=2020)
    p.add_argument("--window-years", type=int, default=3)
    p.add_argument("--query-size", type=int, default=2000)
    p.add_argument(
        "--allow-partial-final-queries",
        action="store_true",
        help="Allow evaluation to proceed with fewer finalized queries than --query-size.",
    )
    p.add_argument(
        "--min-final-queries",
        type=int,
        default=0,
        help="Minimum finalized queries required when --allow-partial-final-queries is set; 0 means no explicit floor.",
    )
    p.add_argument("--query-pool-multiplier", type=int, default=25)
    p.add_argument("--max-query-scan", type=int, default=300000)
    p.add_argument(
        "--xy-citation-only",
        action="store_true",
        help="Build queries/positives only from 引证专利 paired with 引证类别 in the patents table.",
    )
    p.add_argument(
        "--xy-categories",
        type=str,
        default="X,Y",
        help="Allowed citation categories when --xy-citation-only is enabled (e.g. X,Y).",
    )
    p.add_argument(
        "--citation-pubno-field",
        type=str,
        default="引证专利",
        help="Column name containing cited patent pubnos for XY mode.",
    )
    p.add_argument(
        "--citation-category-field",
        type=str,
        default="引证类别",
        help="Column name containing cited patent categories for XY mode.",
    )
    p.add_argument(
        "--query-source-citation-csv",
        type=str,
        default="",
        help="If set, build query pool from citation_quartet_scores.csv anchors/self/other instead of scanning yearly patents.",
    )

    p.add_argument("--hard-neg-per-query", type=int, default=10)
    p.add_argument("--ipc-semantic-per-query", type=int, default=0)
    p.add_argument("--ipc-random-per-query", type=int, default=10)
    p.add_argument("--cross-ipc-hard-per-query", type=int, default=0)
    p.add_argument("--bm25-top-k", type=int, default=200)
    p.add_argument("--cross-ipc-bm25-top-k", type=int, default=600)
    p.add_argument("--bm25-pool-per-ipc", type=int, default=5000)
    p.add_argument("--global-bm25-pool-size", type=int, default=0)

    p.add_argument("--exclude-pubno-file", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--rank-chunk-size", type=int, default=64)
    p.add_argument("--sql-fetch-batch", type=int, default=16384)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    xy_allowed_categories: Set[str] = set()

    if args.xy_citation_only:
        raw_cats = [x.strip().upper() for x in re.split(r"[;,，；|\s]+", clean_text(args.xy_categories)) if x.strip()]
        for c in raw_cats:
            m = CATEGORY_LETTER_RE.search(c)
            if m:
                xy_allowed_categories.add(m.group(0))
        if not xy_allowed_categories:
            raise ValueError("xy_categories parsed empty; please provide at least one category like X,Y")
        if args.query_source_citation_csv:
            raise ValueError("xy_citation_only is incompatible with query_source_citation_csv mode")

    if args.window_years <= 0:
        raise ValueError("window_years must be > 0")
    if args.hard_neg_per_query < 0 or args.ipc_semantic_per_query < 0 or args.ipc_random_per_query < 0:
        raise ValueError("negative counts must be >= 0")
    if args.cross_ipc_hard_per_query < 0:
        raise ValueError("cross_ipc_hard_per_query must be >= 0")
    if args.cross_ipc_hard_per_query > 0 and args.global_bm25_pool_size <= 0:
        raise ValueError("global_bm25_pool_size must be > 0 when cross_ipc_hard_per_query > 0")
    if args.min_final_queries < 0:
        raise ValueError("min_final_queries must be >= 0")
    if args.min_final_queries > 0 and not args.allow_partial_final_queries:
        raise ValueError("min_final_queries requires --allow-partial-final-queries")
    if args.min_final_queries > args.query_size:
        raise ValueError("min_final_queries cannot exceed query_size")

    start_year = args.query_year - args.window_years
    end_year = args.query_year - 1

    if args.run_name:
        run_dir = Path(args.output_dir) / args.run_name
    else:
        run_dir = Path(args.output_dir) / f"topk_citation_bm25_ipc_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run-name to locate existing artifacts")

    q_path = run_dir / "queries.jsonl"
    c_path = run_dir / "candidates_global.jsonl"
    dataset_meta_path = run_dir / "dataset_meta.json"
    ft_cache_path = run_dir / "eval_finetuned_cache.json"
    base_cache_path = run_dir / "eval_base_cache.json"

    excluded = load_excluded_pubnos(args.exclude_pubno_file)
    log(
        f"Output={run_dir}, query_year={args.query_year}, window=[{start_year},{end_year}], query_size={args.query_size}, "
        f"hard_neg_per_query={args.hard_neg_per_query}, ipc_semantic_per_query={args.ipc_semantic_per_query}, "
        f"ipc_random_per_query={args.ipc_random_per_query}, cross_ipc_hard_per_query={args.cross_ipc_hard_per_query}, "
        f"excluded={len(excluded)}"
    )

    if args.resume:
        if not q_path.exists() or not c_path.exists():
            raise FileNotFoundError(
                "resume requested but dataset artifacts are missing: "
                f"queries={q_path.exists()}, candidates={c_path.exists()}, run_dir={run_dir}"
            )

        final_queries = load_queries_jsonl(q_path)
        candidate_map = load_candidate_map_jsonl(c_path)
        if args.allow_partial_final_queries:
            min_need = max(1, int(args.min_final_queries)) if args.min_final_queries > 0 else 1
            if len(final_queries) < min_need:
                raise ValueError(
                    "resume dataset query size too small for partial mode: "
                    f"min_required={min_need}, existing={len(final_queries)}"
                )
        else:
            if len(final_queries) != args.query_size:
                raise ValueError(
                    f"resume dataset query size mismatch: args={args.query_size}, existing={len(final_queries)}"
                )

        if dataset_meta_path.exists():
            dataset_meta = read_json(dataset_meta_path)
            validate_resume_dataset_meta(args, dataset_meta, start_year, end_year)
        else:
            dataset_meta = {
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "db_path": args.db_path,
                "table": args.table,
                "query_year": args.query_year,
                "candidate_window": [start_year, end_year],
                "query_source_citation_csv": args.query_source_citation_csv,
                "xy_citation_only": args.xy_citation_only,
                "xy_categories": sorted(xy_allowed_categories),
                "citation_pubno_field": args.citation_pubno_field if args.xy_citation_only else "",
                "citation_category_field": args.citation_category_field if args.xy_citation_only else "",
                "query_size_target": args.query_size,
                "query_size": len(final_queries),
                "allow_partial_final_queries": args.allow_partial_final_queries,
                "min_final_queries": args.min_final_queries,
                "hard_neg_per_query": args.hard_neg_per_query,
                "ipc_semantic_per_query": args.ipc_semantic_per_query,
                "ipc_random_per_query": args.ipc_random_per_query,
                "cross_ipc_hard_per_query": args.cross_ipc_hard_per_query,
                "bm25_top_k": args.bm25_top_k,
                "cross_ipc_bm25_top_k": args.cross_ipc_bm25_top_k,
                "bm25_pool_per_ipc": args.bm25_pool_per_ipc,
                "global_bm25_pool_size": args.global_bm25_pool_size,
                "exclude_pubno_file": args.exclude_pubno_file,
                "exclude_pubno_count": len(excluded),
                "positive_counts": {
                    "total_from_self": int(sum(q.positive_from_self for q in final_queries)),
                    "total_from_other": int(sum(q.positive_from_other for q in final_queries)),
                    "avg_positive_per_query": float(np.mean([len(q.positive_pubnos) for q in final_queries])),
                },
                "negative_counts": {
                    "avg_bm25_hard_per_query": float(np.mean([len(q.bm25_hard_neg_pubnos) for q in final_queries])),
                    "avg_ipc_semantic_per_query": float(np.mean([len(q.ipc_semantic_neg_pubnos) for q in final_queries])),
                    "avg_ipc_random_per_query": float(np.mean([len(q.ipc_random_neg_pubnos) for q in final_queries])),
                    "avg_cross_ipc_hard_per_query": float(np.mean([len(q.cross_ipc_hard_neg_pubnos) for q in final_queries])),
                },
                "global_candidate_universe_size": len(candidate_map),
                "build_time_seconds": 0.0,
                "artifacts": {
                    "queries": str(q_path),
                    "candidates_global": str(c_path),
                },
            }

        dataset_meta["query_size_target"] = args.query_size
        dataset_meta["query_size"] = len(final_queries)
        dataset_meta["allow_partial_final_queries"] = args.allow_partial_final_queries
        dataset_meta["min_final_queries"] = args.min_final_queries
        dataset_meta["global_candidate_universe_size"] = len(candidate_map)
        dataset_meta.setdefault("build_time_seconds", 0.0)
        dataset_meta.setdefault("artifacts", {})
        dataset_meta["artifacts"]["queries"] = str(q_path)
        dataset_meta["artifacts"]["candidates_global"] = str(c_path)
        write_json(dataset_meta_path, dataset_meta)
        log(
            "Resume mode: loaded existing dataset artifacts "
            f"queries={len(final_queries)}, candidates={len(candidate_map)}"
        )
    else:
        conn = sqlite3.connect(args.db_path)
        conn.row_factory = sqlite3.Row

        try:
            t0 = time.time()
            if args.query_source_citation_csv:
                query_raw_pool = build_query_raw_pool_from_citation_csv(
                    conn=conn,
                    table=args.table,
                    citation_csv=args.query_source_citation_csv,
                    query_size=args.query_size,
                    query_pool_multiplier=args.query_pool_multiplier,
                    excluded_pubnos=excluded,
                    rng=rng,
                )
            else:
                query_raw_pool = build_query_raw_pool(
                    conn=conn,
                    table=args.table,
                    query_year=args.query_year,
                    query_size=args.query_size,
                    query_pool_multiplier=args.query_pool_multiplier,
                    max_query_scan=args.max_query_scan,
                    excluded_pubnos=excluded,
                    sql_fetch_batch=args.sql_fetch_batch,
                    xy_citation_only=args.xy_citation_only,
                    citation_pubno_field=args.citation_pubno_field,
                    citation_category_field=args.citation_category_field,
                    xy_allowed_categories=xy_allowed_categories,
                )

            if len(query_raw_pool) < args.query_size:
                raise RuntimeError(
                    f"query_raw_pool too small: need {args.query_size}, got {len(query_raw_pool)}"
                )

            need_pos: Set[str] = set()
            ipc_set: Set[str] = set()
            for q in query_raw_pool:
                need_pos.update(q.self_cites_raw)
                need_pos.update(q.other_cites_raw)
                if q.query_ipc4:
                    ipc_set.add(q.query_ipc4)

            positive_map, ipc_reservoir, global_reservoir = scan_window_collect(
                conn=conn,
                table=args.table,
                start_year=start_year,
                end_year=end_year,
                needed_positive_pubnos=need_pos,
                target_ipc_set=ipc_set,
                excluded_pubnos=excluded,
                bm25_pool_per_ipc=args.bm25_pool_per_ipc,
                global_bm25_pool_size=args.global_bm25_pool_size,
                rng=rng,
                sql_fetch_batch=args.sql_fetch_batch,
            )

            final_queries = build_final_queries(
                query_raw_pool=query_raw_pool,
                positive_map=positive_map,
                ipc_reservoir=ipc_reservoir,
                global_docs=global_reservoir,
                query_size=args.query_size,
                hard_neg_per_query=args.hard_neg_per_query,
                ipc_semantic_per_query=args.ipc_semantic_per_query,
                ipc_random_per_query=args.ipc_random_per_query,
                cross_ipc_hard_per_query=args.cross_ipc_hard_per_query,
                bm25_top_k=args.bm25_top_k,
                cross_ipc_bm25_top_k=args.cross_ipc_bm25_top_k,
                rng=rng,
                allow_partial_final_queries=args.allow_partial_final_queries,
                min_final_queries=args.min_final_queries,
            )

            candidate_map = build_global_candidate_map(final_queries, positive_map, ipc_reservoir, global_reservoir)
            t1 = time.time()

        finally:
            conn.close()

        with q_path.open("w", encoding="utf-8") as f:
            for q in final_queries:
                f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

        with c_path.open("w", encoding="utf-8") as f:
            for p in sorted(candidate_map.keys()):
                f.write(json.dumps(asdict(candidate_map[p]), ensure_ascii=False) + "\n")

        self_pos_total = int(sum(q.positive_from_self for q in final_queries))
        other_pos_total = int(sum(q.positive_from_other for q in final_queries))

        dataset_meta = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "db_path": args.db_path,
            "table": args.table,
            "query_year": args.query_year,
            "candidate_window": [start_year, end_year],
            "query_source_citation_csv": args.query_source_citation_csv,
            "xy_citation_only": args.xy_citation_only,
            "xy_categories": sorted(xy_allowed_categories),
            "citation_pubno_field": args.citation_pubno_field if args.xy_citation_only else "",
            "citation_category_field": args.citation_category_field if args.xy_citation_only else "",
            "query_size_target": args.query_size,
            "query_size": len(final_queries),
            "allow_partial_final_queries": args.allow_partial_final_queries,
            "min_final_queries": args.min_final_queries,
            "hard_neg_per_query": args.hard_neg_per_query,
            "ipc_semantic_per_query": args.ipc_semantic_per_query,
            "ipc_random_per_query": args.ipc_random_per_query,
            "cross_ipc_hard_per_query": args.cross_ipc_hard_per_query,
            "bm25_top_k": args.bm25_top_k,
            "cross_ipc_bm25_top_k": args.cross_ipc_bm25_top_k,
            "bm25_pool_per_ipc": args.bm25_pool_per_ipc,
            "global_bm25_pool_size": args.global_bm25_pool_size,
            "exclude_pubno_file": args.exclude_pubno_file,
            "exclude_pubno_count": len(excluded),
            "positive_counts": {
                "total_from_self": self_pos_total,
                "total_from_other": other_pos_total,
                "avg_positive_per_query": float(np.mean([len(q.positive_pubnos) for q in final_queries])),
            },
            "negative_counts": {
                "avg_bm25_hard_per_query": float(np.mean([len(q.bm25_hard_neg_pubnos) for q in final_queries])),
                "avg_ipc_semantic_per_query": float(np.mean([len(q.ipc_semantic_neg_pubnos) for q in final_queries])),
                "avg_ipc_random_per_query": float(np.mean([len(q.ipc_random_neg_pubnos) for q in final_queries])),
                "avg_cross_ipc_hard_per_query": float(np.mean([len(q.cross_ipc_hard_neg_pubnos) for q in final_queries])),
            },
            "global_candidate_universe_size": len(candidate_map),
            "build_time_seconds": float(t1 - t0),
            "artifacts": {
                "queries": str(q_path),
                "candidates_global": str(c_path),
            },
        }
        write_json(dataset_meta_path, dataset_meta)

    # Evaluate two models on exactly same query/candidate construction.
    k_list = [1, 3, 5, 10, 20, 50]

    if args.resume and ft_cache_path.exists():
        log(f"Resume mode: loading cached fine-tuned eval from {ft_cache_path}")
        ft_eval = read_json(ft_cache_path)
    else:
        log("Evaluating fine-tuned model...")
        ft = HFEmbedder(args.finetuned_model_path, args.device, args.max_length, args.batch_size)
        ft_eval = evaluate_model(
            ft,
            final_queries,
            candidate_map,
            args.rank_chunk_size,
            k_list,
            embed_cache_dir=(run_dir / "eval_finetuned_embed_cache") if args.resume else None,
        )
        write_json(ft_cache_path, ft_eval)
        del ft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.resume and base_cache_path.exists():
        log(f"Resume mode: loading cached base eval from {base_cache_path}")
        base_eval = read_json(base_cache_path)
    else:
        log("Evaluating base model...")
        base = HFEmbedder(args.base_model_path, args.device, args.max_length, args.batch_size)
        base_eval = evaluate_model(
            base,
            final_queries,
            candidate_map,
            args.rank_chunk_size,
            k_list,
            embed_cache_dir=(run_dir / "eval_base_embed_cache") if args.resume else None,
        )
        write_json(base_cache_path, base_eval)
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ft_ranks = np.asarray([x["first_rel_rank"] for x in ft_eval["per_query"]], dtype=np.float64)
    base_ranks = np.asarray([x["first_rel_rank"] for x in base_eval["per_query"]], dtype=np.float64)

    wins = int(np.sum(ft_ranks < base_ranks))
    losses = int(np.sum(ft_ranks > base_ranks))
    ties = int(np.sum(ft_ranks == base_ranks))

    # Per-query merged csv.
    per_csv = run_dir / "per_query_comparison.csv"
    with per_csv.open("w", encoding="utf-8") as f:
        f.write(
            "query_pubno,query_ipc4,positive_count,candidate_count,"
            "rank_finetuned,rank_base,rr_finetuned,rr_base,ap_finetuned,ap_base,"
            "ndcg_finetuned,ndcg_base,ndcg@5_finetuned,ndcg@5_base,top1_pubno_finetuned,top1_pubno_base\n"
        )
        for i, q in enumerate(final_queries):
            a = ft_eval["per_query"][i]
            b = base_eval["per_query"][i]
            f.write(
                f"{q.query_pubno},{q.query_ipc4},{int(a['positive_count'])},{int(a['candidate_count'])},"
                f"{a['first_rel_rank']:.0f},{b['first_rel_rank']:.0f},"
                f"{a['rr']:.8f},{b['rr']:.8f},{a['ap']:.8f},{b['ap']:.8f},"
                f"{a['ndcg']:.8f},{b['ndcg']:.8f},{float(a.get('ndcg@5', float('nan'))):.8f},{float(b.get('ndcg@5', float('nan'))):.8f},"
                f"{a['top1_pubno']},{b['top1_pubno']}\n"
            )

    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_meta,
        "models": {
            "finetuned_model_path": args.finetuned_model_path,
            "base_model_path": args.base_model_path,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "metrics": {
            "finetuned": ft_eval["metrics"],
            "base": base_eval["metrics"],
        },
        "paired_comparison": {
            "wins_ft_better": wins,
            "losses_ft_worse": losses,
            "ties": ties,
            "sign_test_pvalue_approx": sign_test_pvalue_approx(wins, losses),
            "mean_rank_delta_base_minus_ft": float(np.mean(base_ranks - ft_ranks)),
            "mrr_delta_ft_minus_base": float(ft_eval["metrics"]["mrr"] - base_eval["metrics"]["mrr"]),
            "map_delta_ft_minus_base": float(ft_eval["metrics"]["map"] - base_eval["metrics"]["map"]),
            "ndcg_delta_ft_minus_base": float(ft_eval["metrics"]["ndcg"] - base_eval["metrics"]["ndcg"]),
        },
        "timing_seconds": {
            "dataset_build": dataset_meta["build_time_seconds"],
            "finetuned_eval": ft_eval["timing_seconds"],
            "base_eval": base_eval["timing_seconds"],
        },
        "artifacts": {
            "queries": str(q_path),
            "candidates_global": str(c_path),
            "per_query_comparison": str(per_csv),
            "dataset_meta": str(dataset_meta_path),
            "eval_finetuned_cache": str(ft_cache_path),
            "eval_base_cache": str(base_cache_path),
        },
    }

    write_json(run_dir / "summary_topk_retrieval.json", summary)

    metrics_csv = run_dir / "summary_metrics_table.csv"
    with metrics_csv.open("w", encoding="utf-8") as f:
        f.write("metric,finetuned,base\n")
        metric_keys = ["mean_rank", "mrr", "map", "ndcg"]
        for k in [1, 3, 5, 10, 20, 50]:
            metric_keys.append(f"hit@{k}")
            metric_keys.append(f"recall@{k}")
            ndcgk = f"ndcg@{k}"
            if ndcgk in ft_eval["metrics"] and ndcgk in base_eval["metrics"]:
                metric_keys.append(ndcgk)
        for k in metric_keys:
            f.write(f"{k},{float(ft_eval['metrics'][k]):.8f},{float(base_eval['metrics'][k]):.8f}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] outputs saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
