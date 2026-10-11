#!/usr/bin/env python3
"""用独立的多模态 Judge 并发评估证据、IAB、Tag 粒度与可复用性。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

from core import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    EVIDENCE_JUDGE_PROMPT,
    EVIDENCE_JUDGE_PROMPT_VERSION,
    IABTaxonomy,
    QwenClient,
    append_jsonl,
    build_evidence_judge_user_text,
    completed_post_ids,
    load_jsonl,
    prompt_manifest,
)


VALID_SUPPORT = {"direct", "strong_inference", "weak_inference", "unsupported", "contradicted"}
VALID_VERDICTS = {"accept", "review", "reject"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag Quality Judge")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--generations", type=Path, default=DEFAULT_OUTPUT_DIR / "tags_10000_v33.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "evidence_10000_v33.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-pending", type=int, default=0,
                        help="同时排队的请求数；0 表示 workers 的两倍")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_judgment(result: dict[str, Any], expected_count: int) -> dict[str, Any]:
    judgments = result.get("judgments")
    errors: list[str] = []
    if not isinstance(judgments, list):
        return {"valid": False, "errors": ["judgments_not_list"], "expected": expected_count,
                "received": 0, "accepted": 0, "review": 0, "rejected": 0}
    seen: set[int] = set()
    counts = {"accept": 0, "review": 0, "reject": 0}
    for item in judgments:
        if not isinstance(item, dict):
            errors.append("judgment_not_object")
            continue
        try:
            index = int(item.get("tag_index"))
        except (TypeError, ValueError):
            errors.append("invalid_tag_index")
            continue
        if not 0 <= index < expected_count or index in seen:
            errors.append("duplicate_or_out_of_range_tag_index")
        seen.add(index)
        support = str(item.get("evidence_support", ""))
        verdict = str(item.get("verdict", ""))
        if not isinstance(item.get("evidence_exists"), bool):
            errors.append("invalid_evidence_exists")
        if support not in VALID_SUPPORT:
            errors.append("invalid_evidence_support")
        if not str(item.get("reason", "")).strip():
            errors.append("invalid_reason")
        if verdict not in VALID_VERDICTS:
            errors.append("invalid_verdict")
        else:
            counts[verdict] += 1
    if seen != set(range(expected_count)):
        errors.append("missing_tag_judgments")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "expected": expected_count,
        "received": len(judgments),
        "accepted": counts["accept"],
        "review": counts["review"],
        "rejected": counts["reject"],
    }


def pending_generations(
    path: Path, done: set[int], limit: int
) -> Iterator[dict[str, Any]]:
    yielded = 0
    for generation in load_jsonl(path):
        if generation.get("status") != "ok":
            continue
        post_id = int(generation["post_id"])
        if post_id in done:
            continue
        if limit and yielded >= limit:
            break
        yielded += 1
        yield generation


def judge_one(
    generation: dict[str, Any],
    *,
    post: dict[str, Any],
    client: QwenClient,
    taxonomy: IABTaxonomy,
) -> dict[str, Any]:
    post_id = int(generation["post_id"])
    tags = generation.get("result", {}).get("tags", [])
    record: dict[str, Any] = {
        "post_id": post_id,
        "generation_prompt_sha256": generation.get("prompt_sha256"),
        "judge_model": client.model,
        "judge_prompt_version": EVIDENCE_JUDGE_PROMPT_VERSION,
    }
    try:
        user_text = build_evidence_judge_user_text(post, generation, taxonomy)
        image_url = str(post.get("cover_url") or "") or None
        try:
            result, raw, usage = client.chat(
                EVIDENCE_JUDGE_PROMPT, user_text, image_url=image_url
            )
        except Exception as image_error:
            if not image_url:
                raise
            result, raw, usage = client.chat(EVIDENCE_JUDGE_PROMPT, user_text)
            record["input_fallback"] = "text_only_after_image_request_error"
            record["image_request_error"] = str(image_error)
        record.update(
            {
                "status": "ok",
                "result": result,
                "raw_response": raw,
                "usage": usage,
                "automatic_checks": validate_judgment(result, len(tags)),
            }
        )
    except Exception as error:  # noqa: BLE001
        record.update({"status": "error", "error": str(error)})
    return record


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    max_pending = args.max_pending or args.workers * 2
    if max_pending < args.workers:
        raise ValueError("--max-pending 必须 >= --workers")

    taxonomy = IABTaxonomy.load()
    client = QwenClient.from_env(
        model_env="QWEN_JUDGE_MODEL",
        default_model="qwen3.7-flash",
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
    )
    posts = {int(row["post_id"]): row for row in load_jsonl(args.dataset)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    done = completed_post_ids(args.output)
    manifest = {
        **prompt_manifest(
            EVIDENCE_JUDGE_PROMPT,
            version=EVIDENCE_JUDGE_PROMPT_VERSION,
            model=client.model,
        ),
        "dataset": str(args.dataset.resolve()),
        "generations": str(args.generations.resolve()),
        "output": str(args.output.resolve()),
        "workers": args.workers,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.output.with_suffix(".prompt.txt").write_text(EVIDENCE_JUDGE_PROMPT + "\n", encoding="utf-8")

    generations = iter(pending_generations(args.generations, done, args.limit))
    submitted = 0
    processed = 0
    pending: dict[Future[dict[str, Any]], int] = {}
    with args.output.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    generation = next(generations)
                except StopIteration:
                    exhausted = True
                    break
                post_id = int(generation["post_id"])
                post = posts.get(post_id)
                if not post:
                    raise KeyError(f"数据快照中找不到 post_id={post_id}")
                future = executor.submit(
                    judge_one,
                    generation,
                    post=post,
                    client=client,
                    taxonomy=taxonomy,
                )
                pending[future] = post_id
                submitted += 1
            if not pending:
                continue
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                post_id = pending.pop(future)
                try:
                    record = future.result()
                except Exception as error:  # pragma: no cover
                    record = {"post_id": post_id, "status": "error", "error": str(error)}
                append_jsonl(handle, record)
                processed += 1
                print(
                    f"judged={processed}/{submitted} post_id={post_id} "
                    f"status={record['status']} pending={len(pending)}",
                    flush=True,
                )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
