#!/usr/bin/env python3
"""按数据快照顺序保留每个 post_id 的最终成功 JSONL 记录。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from core import DEFAULT_DATASET, append_jsonl, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="压缩断点续跑产生的 JSONL 历史记录")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    latest_ok: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(args.input):
        if row.get("status") == "ok":
            latest_ok[int(row["post_id"])] = row
    dataset_ids = [int(row["post_id"]) for row in load_jsonl(args.dataset)]
    missing = [post_id for post_id in dataset_ids if post_id not in latest_ok]
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个成功 post_id，示例: {missing[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for post_id in dataset_ids:
            append_jsonl(handle, latest_ok[post_id])
    print(f"output={args.output} rows={len(dataset_ids)}")


if __name__ == "__main__":
    main()
