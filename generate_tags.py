#!/usr/bin/env python3
"""对固定帖子快照并发运行无 Registry 的 IAB Tag 生成。"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

from core import (
    DEFAULT_DATASET,
    DEFAULT_OUTPUT_DIR,
    GENERATOR_PROMPT_VERSION,
    IABTaxonomy,
    QwenClient,
    append_jsonl,
    build_generator_prompt,
    build_post_user_text,
    completed_post_ids,
    load_jsonl,
    normalize_generated_result,
    prompt_manifest,
    validate_generated_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 IAB 约束的多模态 Tags")
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "tags_10000_v33.jsonl")
    parser.add_argument("--max-tags", type=int, default=10)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="直接使用指定的完整 system prompt；用于确保人工修改的预览文件被原样采用",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-pending", type=int, default=0,
                        help="同时排队的请求数；0 表示 workers 的两倍")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="不保存重复的 raw_response 和逐 Tag 程序检查细节，用于大批量节省磁盘。",
    )
    return parser.parse_args()


def pending_posts(path: Path, done: set[int], limit: int) -> Iterator[dict[str, Any]]:
    yielded = 0
    for post in load_jsonl(path):
        if int(post["post_id"]) in done:
            continue
        if limit and yielded >= limit:
            break
        yielded += 1
        yield post


def generate_one(
    post: dict[str, Any],
    *,
    client: QwenClient,
    taxonomy: IABTaxonomy,
    system_prompt: str,
    prompt_sha256: str,
    max_tags: int,
) -> dict[str, Any]:
    post_id = int(post["post_id"])
    record: dict[str, Any] = {
        "post_id": post_id,
        "snapshot_id": post.get("snapshot_id"),
        "title": post.get("title", ""),
        "cover_url": post.get("cover_url", ""),
        "model": client.model,
        "prompt_version": GENERATOR_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
    }
    try:
        user_text = build_post_user_text(post)
        image_url = str(post.get("cover_url") or "") or None
        attempt_usages: list[dict[str, Any]] = []

        def request_once(request_text: str) -> tuple[dict[str, Any], str]:
            try:
                response_result, response_raw, response_usage = client.chat(
                    system_prompt, request_text, image_url=image_url
                )
            except Exception as image_error:
                if not image_url:
                    raise
                response_result, response_raw, response_usage = client.chat(system_prompt, request_text)
                record["input_fallback"] = "text_only_after_image_request_error"
                record["image_request_error"] = str(image_error)
            attempt_usages.append(response_usage)
            return response_result, response_raw

        def evidence_complete(value: dict[str, Any]) -> bool:
            tags = value.get("tags")
            return bool(tags) and isinstance(tags, list) and all(
                isinstance(tag, dict)
                and isinstance(tag.get("evidence"), list)
                and len(tag["evidence"]) == 1
                for tag in tags
            )

        result, raw = request_once(user_text)
        if not evidence_complete(result):
            record["generation_retry"] = "missing_or_invalid_evidence"
            repair_text = (
                user_text
                + "\n\n上一次响应中有 Tag 缺少或包含多条 evidence，因此响应无效。"
                "请重新生成完整 JSON，确保每个 Tag 恰好包含一条 evidence。"
            )
            result, raw = request_once(repair_text)
        usage = {
            key: sum(int(item.get(key) or 0) for item in attempt_usages)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        usage["request_count"] = len(attempt_usages)
        if not evidence_complete(result):
            raise ValueError("生成结果在自动重试后仍有 Tag 缺少或包含多条 evidence")
        normalizations = normalize_generated_result(result, post, taxonomy)
        record.update(
            {
                "status": "ok",
                "result": result,
                "raw_response": raw,
                "usage": usage,
                "automatic_normalizations": normalizations,
                "automatic_checks": validate_generated_result(
                    result, post, taxonomy, max_tags=max_tags
                ),
            }
        )
    except Exception as error:  # noqa: BLE001
        record.update({"status": "error", "error": str(error)})
    return record


def main() -> None:
    args = parse_args()
    if args.max_tags < 1:
        raise ValueError("--max-tags 必须 >= 1")
    if args.workers < 1:
        raise ValueError("--workers 必须 >= 1")
    max_pending = args.max_pending or args.workers * 2
    if max_pending < args.workers:
        raise ValueError("--max-pending 必须 >= --workers")

    taxonomy = IABTaxonomy.load()
    client = QwenClient.from_env(
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
    )
    system_prompt = (
        args.prompt_file.read_text(encoding="utf-8-sig").strip()
        if args.prompt_file
        else build_generator_prompt(taxonomy, max_tags=args.max_tags)
    )
    prompt_info = prompt_manifest(
        system_prompt, version=GENERATOR_PROMPT_VERSION, model=client.model
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    done = completed_post_ids(args.output)
    args.output.with_suffix(".prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    manifest = {
        **prompt_info,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "prompt_file": str(args.prompt_file.resolve()) if args.prompt_file else None,
        "max_tags": args.max_tags,
        "workers": args.workers,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    posts = iter(pending_posts(args.input, done, args.limit))
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
                    post = next(posts)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(
                    generate_one,
                    post,
                    client=client,
                    taxonomy=taxonomy,
                    system_prompt=system_prompt,
                    prompt_sha256=prompt_info["sha256"],
                    max_tags=args.max_tags,
                )
                pending[future] = int(post["post_id"])
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
                if args.compact and record.get("status") == "ok":
                    record.pop("raw_response", None)
                    checks = record.get("automatic_checks")
                    if isinstance(checks, dict):
                        record["automatic_checks"] = {
                            key: value for key, value in checks.items() if key != "tag_checks"
                        }
                append_jsonl(handle, record)
                processed += 1
                print(
                    f"processed={processed}/{submitted} post_id={post_id} "
                    f"status={record['status']} pending={len(pending)}",
                    flush=True,
                )
    print(f"output={args.output}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
