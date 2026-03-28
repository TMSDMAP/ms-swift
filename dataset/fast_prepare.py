#!/usr/bin/env python3
"""
将原始专利 JSONL 转换为 ms-swift LLM InfoNCE 对比学习格式。

核心优化：
1. 使用 bm25s 替代 rank_bm25，彻底解决大规模矩阵运算耗时问题。
2. 强化了 Pass2 阶段的重复 ID 防御机制。
3. 对检索库返回的索引数据进行了健壮的类型投影和边界转换。
4. 【新增】支持 --resume 断点续传，自动修复中断导致的尾部损坏 JSON 行。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import jieba
import bm25s

_G_BM25 = None
_G_DOC_IDS: List[str] = []
_G_DOC_IPC: List[str] = []
_G_DOC_TITLE: List[str] = []
_G_DOC_ABSTRACT: List[str] = []
_G_TOP_K = 200
_G_SEED = 42

REQUIRED_FIELDS = [
    "id",
    "title",
    "abstract",
    "claims",
    "background",
    "primary_ipc_subgroup",
]

@dataclass
class PatentRecord:
    patent_id: str
    title: str
    abstract: str
    claims: str
    background: str
    primary_ipc_subgroup: str

    @property
    def title_abstract(self) -> str:
        return f"{self.title} {self.abstract}".strip()

def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_nested_value(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def get_first_non_empty(obj: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        value = get_nested_value(obj, key)
        text = clean_text(value)
        if text:
            return text
    return ""

def normalize_ipc_subgroup(raw_ipc: str) -> str:
    text = clean_text(raw_ipc)
    if not text:
        return ""
    first = text.split(";", 1)[0]
    return clean_text(first)

def extract_ipc_levels(raw_ipc: str) -> Tuple[str, str, str]:
    """解析 IPC 层级，返回 (subclass, maingroup, subgroup)。"""
    text = clean_text(raw_ipc).upper()
    if not text:
        return "", "", ""

    # 兼容 A01B 33/00 与 A01B33/00 两种常见写法。
    compact = re.sub(r"\s+", "", text)
    match = re.match(r"^([A-HY]\d{2}[A-Z])(\d+)/(\d+)", compact)
    if match:
        subclass = match.group(1)
        maingroup = f"{subclass} {match.group(2)}"
        subgroup = f"{maingroup}/{match.group(3)}"
        return subclass, maingroup, subgroup

    match_sub = re.search(r"[A-HY]\d{2}[A-Z]", compact)
    subclass = match_sub.group(0) if match_sub else compact[:4]
    if "/" in compact:
        maingroup = compact.split("/", 1)[0]
    else:
        maingroup = compact
    return subclass, maingroup, compact

def parse_and_validate_obj(obj: Any, item_no: int) -> Tuple[Optional[PatentRecord], Optional[str]]:
    if not isinstance(obj, dict):
        return None, f"item_{item_no}:not_object"

    patent_id = get_first_non_empty(obj, ["id", "publication_number", "专利号"])
    title = get_first_non_empty(obj, ["title", "标题", "标题 (中文)"])
    abstract = get_first_non_empty(obj, ["abstract", "摘要"])
    claims = get_first_non_empty(obj, ["claims", "first_claim", "权利要求文本", "权利要求"])
    background = get_first_non_empty(
        obj,
        ["background", "specification.background", "背景技术文本", "背景技术"],
    )
    subgroup = normalize_ipc_subgroup(
        get_first_non_empty(obj, ["primary_ipc_subgroup", "ipc_main", "ipc_full", "主分类号小组"])
    )

    if not patent_id:
        return None, f"item_{item_no}:empty_id"
    if not title:
        return None, f"item_{item_no}:empty_title"
    if not abstract:
        return None, f"item_{item_no}:empty_abstract"
    if not subgroup:
        return None, f"item_{item_no}:empty_primary_ipc_subgroup"
    if not claims:
        return None, f"item_{item_no}:empty_claims"
    if not background:
        return None, f"item_{item_no}:empty_background"

    return (
        PatentRecord(
            patent_id=patent_id,
            title=title,
            abstract=abstract,
            claims=claims,
            background=background,
            primary_ipc_subgroup=subgroup,
        ),
        None,
    )

def iter_source_objects(input_path: str) -> Iterator[Tuple[int, Optional[Any], Optional[str]]]:
    if input_path.endswith(".jsonl"):
        with open(input_path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    yield line_no, None, "empty_line"
                    continue
                try:
                    obj = json.loads(line)
                    yield line_no, obj, None
                except json.JSONDecodeError:
                    yield line_no, None, "invalid_json"
        return

    if input_path.endswith(".json"):
        try:
            import ijson
        except ImportError as exc:
            raise RuntimeError(
                "输入为 .json（数组）时需要安装 ijson：pip install ijson"
            ) from exc

        with open(input_path, "rb") as fin:
            for item_no, obj in enumerate(ijson.items(fin, "item"), start=1):
                yield item_no, obj, None
        return

    raise RuntimeError("仅支持 .jsonl 或 .json 输入文件")

def tokenize(text: str) -> List[str]:
    tokens = [tok.strip() for tok in jieba.cut(text) if tok.strip()]
    if not tokens and text:
        tokens = [text]
    return tokens

def make_chat_message(text: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": text}]

def build_sample_from_record(
    record: PatentRecord,
    item_no: int,
    seed: int,
    val_ratio: float,
    bm25,
    doc_ids: List[str],
    doc_ipc: List[str],
    doc_title: List[str],
    doc_abstract: List[str],
    bm25_top_k: int,
) -> Tuple[str, str]:
    rng = random.Random(seed + item_no * 1_000_003)

    view_a_content = f"标题：{record.title}\n摘要：{record.abstract}"
    
    p = rng.random()
    if p < 0.30:
        view_b_content = view_a_content
    elif p < 0.65:
        view_b_content = f"权利要求：{record.claims}"
    else:
        view_b_content = f"背景技术：{record.background}"

    if rng.random() < 0.5:
        raw_query, raw_pos = view_a_content, view_b_content
    else:
        raw_query, raw_pos = view_b_content, view_a_content

    final_query = raw_query
    final_pos = raw_pos

    sample = {
        "messages": make_chat_message(final_query),
        "positive_messages": [make_chat_message(final_pos)],
    }

    query_tokens = tokenize(record.title_abstract)
    
    try:
        results, _ = bm25.retrieve([query_tokens], k=bm25_top_k)
        raw_matches = results[0]  
    except Exception:
        raw_matches = []
        
    candidate_pool = []
    for match in raw_matches[10:]:
        try:
            if isinstance(match, dict) and "id" in match:
                idx = int(match["id"])
            else:
                idx = int(match)
            candidate_pool.append(idx)
        except (ValueError, TypeError):
            continue

    current_id = record.patent_id
    t_subclass, t_maingroup, t_subgroup = extract_ipc_levels(record.primary_ipc_subgroup)

    # IPC 分层回退池：优先同 subgroup，再到 maingroup、subclass，最后跨领域补齐。
    pool_l1 = []
    pool_l2 = []
    pool_l3 = []
    pool_any = []

    for idx in candidate_pool:
        if idx < 0 or idx >= len(doc_ids):
            continue
        if doc_ids[idx] == current_id:
            continue

        c_subclass, c_maingroup, c_subgroup = extract_ipc_levels(doc_ipc[idx])
        if c_subgroup == t_subgroup:
            pool_l1.append(idx)
        elif c_maingroup == t_maingroup:
            pool_l2.append(idx)
        elif c_subclass == t_subclass:
            pool_l3.append(idx)
        else:
            pool_any.append(idx)

    target_neg_count = 4
    chosen = []
    for pool in [pool_l1, pool_l2, pool_l3, pool_any]:
        if len(chosen) >= target_neg_count:
            break
        need = target_neg_count - len(chosen)
        if len(pool) <= need:
            chosen.extend(pool)
        else:
            chosen.extend(rng.sample(pool, k=need))

    # 兜底：池子总体不足时从已选样本中重复补齐，保证恒定 4 个负例。
    if chosen and len(chosen) < target_neg_count:
        original_chosen = list(chosen)
        while len(chosen) < target_neg_count:
            chosen.append(rng.choice(original_chosen))

    if chosen:
        sample["negative_messages"] = [
            make_chat_message(
                f"标题：{doc_title[idx]}\n摘要：{doc_abstract[idx]}"
            )
            for idx in chosen
        ]

    split = "val" if rng.random() < val_ratio else "train"
    return split, json.dumps(sample, ensure_ascii=False)

def _init_worker(
    bm25,
    doc_ids: List[str],
    doc_ipc: List[str],
    doc_title: List[str],
    doc_abstract: List[str],
    bm25_top_k: int,
    seed: int,
) -> None:
    global _G_BM25, _G_DOC_IDS, _G_DOC_IPC, _G_DOC_TITLE, _G_DOC_ABSTRACT, _G_TOP_K, _G_SEED
    _G_BM25 = bm25
    _G_DOC_IDS = doc_ids
    _G_DOC_IPC = doc_ipc
    _G_DOC_TITLE = doc_title
    _G_DOC_ABSTRACT = doc_abstract
    _G_TOP_K = bm25_top_k
    _G_SEED = seed

def _worker_process(task: Tuple[int, PatentRecord, float]) -> Tuple[str, str]:
    item_no, record, val_ratio = task
    return build_sample_from_record(
        record=record,
        item_no=item_no,
        seed=_G_SEED,
        val_ratio=val_ratio,
        bm25=_G_BM25,
        doc_ids=_G_DOC_IDS,
        doc_ipc=_G_DOC_IPC,
        doc_title=_G_DOC_TITLE,
        doc_abstract=_G_DOC_ABSTRACT,
        bm25_top_k=_G_TOP_K,
    )

def repair_and_count_lines(filepath: str) -> int:
    """自动修复损坏的 JSON 行并返回有效行数"""
    if not os.path.exists(filepath):
        return 0
    
    valid_count = 0
    valid_bytes = 0
    with open(filepath, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                # 尝试解析 JSON，成功则记录字节位置
                json.loads(line.decode("utf-8"))
                valid_count += 1
                valid_bytes += len(line)
            except json.JSONDecodeError:
                # 遇到由于中断导致写了一半的脏数据，停止读取
                print(f"检测到损坏的 JSON 行，已在 {filepath} 自动执行截断修复。")
                break
                
    # 截断掉末尾损坏的部分
    with open(filepath, "r+b") as f:
        f.truncate(valid_bytes)
        
    return valid_count

def run_pipeline(
    input_path: str,
    train_output_path: str,
    val_output_path: str,
    seed: int,
    val_ratio: float,
    bm25_top_k: int,
    num_workers: int,   
    chunksize: int,
    log_every: int,
    resume: bool,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}。")

    start_time = time.time()

    # Pass 1: 虽然断点续传了 Pass2，但 BM25 需要全局视野，Pass1 极快，每次启动正常跑一遍即可
    doc_ids: List[str] = []
    doc_ipc: List[str] = []
    doc_title: List[str] = []
    doc_abstract: List[str] = []
    tokenized_corpus: List[List[str]] = []
    id_to_index: Dict[str, int] = {}

    pass1_total = 0
    pass1_valid = 0
    pass1_skipped = 0
    duplicate_id_skipped = 0
    skip_reasons: Dict[str, int] = {}

    for item_no, obj, raw_err in iter_source_objects(input_path):
        pass1_total += 1

        if raw_err is not None:
            pass1_skipped += 1
            skip_reasons[raw_err] = skip_reasons.get(raw_err, 0) + 1
            continue

        record, err = parse_and_validate_obj(obj, item_no)
        if record is None:
            pass1_skipped += 1
            key = err.split(":", 1)[-1] if err else "unknown_error"
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
            continue

        if record.patent_id in id_to_index:
            pass1_skipped += 1
            duplicate_id_skipped += 1
            skip_reasons["duplicate_id"] = skip_reasons.get("duplicate_id", 0) + 1
            continue

        idx = len(doc_ids)
        id_to_index[record.patent_id] = idx
        doc_ids.append(record.patent_id)
        doc_ipc.append(record.primary_ipc_subgroup)
        doc_title.append(record.title)
        doc_abstract.append(record.abstract)
        tokenized_corpus.append(tokenize(record.title_abstract))
        pass1_valid += 1

        if pass1_total % 50000 == 0:
            print(f"[Pass1] processed={pass1_total} ...", flush=True)

    if not tokenized_corpus:
        raise RuntimeError("Pass1 后无有效样本，无法构建 BM25 索引。")

    print("正在构建高效 bm25s 索引，请稍候...", flush=True)
    bm25 = bm25s.BM25()
    bm25.index(tokenized_corpus)
    pass1_time = time.time()
    print(f"Pass1 及索引构建完成，耗时 {pass1_time - start_time:.2f} 秒", flush=True)

    # ---------------- 续传逻辑开始 ----------------
    train_count = 0
    val_count = 0
    total_resumed = 0
    
    if resume:
        print("启动断点续传，正在扫描已生成文件...")
        train_count = repair_and_count_lines(train_output_path)
        val_count = repair_and_count_lines(val_output_path)
        total_resumed = train_count + val_count
        print(f"扫描完毕：已成功继承之前跑完的 {total_resumed} 条记录 (Train: {train_count}, Val: {val_count})。")
        open_mode = "a"  # 追加模式
    else:
        open_mode = "w"  # 覆盖模式

    pass2_seen_valid = 0
    pass2_written = 0
    pass2_skipped = 0
    pass2_seen_ids = set()

    os.makedirs(os.path.dirname(train_output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(val_output_path) or ".", exist_ok=True)

    with (
        open(train_output_path, open_mode, encoding="utf-8") as ftrain,
        open(val_output_path, open_mode, encoding="utf-8") as fval,
    ):
        def valid_record_iter() -> Iterator[Tuple[int, PatentRecord, float]]:
            nonlocal pass2_skipped, pass2_seen_valid, pass2_seen_ids
            for item_no, obj, raw_err in iter_source_objects(input_path):
                if raw_err is not None:
                    pass2_skipped += 1
                    continue

                record, _ = parse_and_validate_obj(obj, item_no)
                if record is None:
                    pass2_skipped += 1
                    continue

                if record.patent_id not in id_to_index:
                    pass2_skipped += 1
                    continue
                    
                if record.patent_id in pass2_seen_ids:
                    pass2_skipped += 1
                    continue
                
                # 登记已阅 ID
                pass2_seen_ids.add(record.patent_id)
                pass2_seen_valid += 1
                
                # 断点续传核心拦截：如果当前有效记录仍在之前跑完的份额内，则直接跳过 yielding (节省重新运算的时间)
                if pass2_seen_valid <= total_resumed:
                    continue

                yield item_no, record, val_ratio

        processed_for_log = 0

        if num_workers <= 1:
            for item_no, record, vr in valid_record_iter():
                split, line_out = build_sample_from_record(
                    record=record,
                    item_no=item_no,
                    seed=seed,
                    val_ratio=vr,
                    bm25=bm25,
                    doc_ids=doc_ids,
                    doc_ipc=doc_ipc,
                    doc_title=doc_title,
                    doc_abstract=doc_abstract,
                    bm25_top_k=bm25_top_k,
                )

                if split == "val":
                    fval.write(line_out + "\n")
                    val_count += 1
                else:
                    ftrain.write(line_out + "\n")
                    train_count += 1

                pass2_written += 1
                processed_for_log += 1
                if log_every > 0 and processed_for_log % log_every == 0:
                    print(
                        f"[Pass2] 本次处理={processed_for_log}, 累计训练集={train_count}, 累计验证集={val_count}",
                        flush=True,
                    )
        else:
            with mp.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(bm25, doc_ids, doc_ipc, doc_title, doc_abstract, bm25_top_k, seed),
            ) as pool:
                for split, line_out in pool.imap(_worker_process, valid_record_iter(), chunksize=chunksize):
                    if split == "val":
                        fval.write(line_out + "\n")
                        val_count += 1
                    else:
                        ftrain.write(line_out + "\n")
                        train_count += 1

                    pass2_written += 1
                    processed_for_log += 1
                    if log_every > 0 and processed_for_log % log_every == 0:
                        print(
                            f"[Pass2] 本次处理={processed_for_log}, 累计训练集={train_count}, 累计验证集={val_count}",
                            flush=True,
                        )

    end_time = time.time()

    print("===== Done =====")
    print(f"Input file: {input_path}")
    print(f"Pass1 累计耗时: {pass1_time - start_time:.2f} 秒")
    print(f"本次实际新写入样本: {pass2_written}")
    print(f"加上继承数据后的最终总计 -> Train: {train_count}, Val: {val_count}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess patent JSONL into ms-swift LLM InfoNCE format with two-pass streaming."
    )
    parser.add_argument("--input", type=str, default="raw_patents.jsonl", help="原始专利输入路径")
    parser.add_argument("--train-output", type=str, default="train.jsonl", help="输出训练集 JSONL 路径")
    parser.add_argument("--val-output", type=str, default="val.jsonl", help="输出验证集 JSONL 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="验证集比例（默认 1%）")
    parser.add_argument("--bm25-top-k", type=int, default=200, help="BM25 检索 TopK，默认 200")
    parser.add_argument("--num-workers", type=int, default=1, help="Pass2 并行进程数。")
    parser.add_argument("--chunksize", type=int, default=64, help="多进程批大小，默认 64")
    parser.add_argument("--log-every", type=int, default=5000, help="Pass2 每处理多少条打印一次进度，<=0 关闭")
    
    # 新增 Resume 命令行参数
    parser.add_argument("--resume", action="store_true", help="开启断点续传（自动读取并继承已有文件的进度）")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        train_output_path=args.train_output,
        val_output_path=args.val_output,
        seed=args.seed,
        val_ratio=args.val_ratio,
        bm25_top_k=args.bm25_top_k,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
        log_every=args.log_every,
        resume=args.resume,
    )