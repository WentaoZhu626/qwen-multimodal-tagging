#!/usr/bin/env python3
"""计算无金标准的 Evidence、Judge 和简化正向检索指标。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core import DEFAULT_OUTPUT_DIR, canonical_text, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag 模拟的无金标准评估")
    parser.add_argument("--generations", type=Path, default=DEFAULT_OUTPUT_DIR / "tags_10000_v32.jsonl")
    parser.add_argument("--judgments", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "evaluation_10000_v32.json")
    parser.add_argument("--tag-stats", type=Path, default=DEFAULT_OUTPUT_DIR / "tag_retrieval_stats_10000_v32.csv")
    parser.add_argument("--overbroad-share", type=float, default=0.10)
    return parser.parse_args()


def tag_key(tag: dict[str, Any]) -> str:
    return f"{tag.get('iab_id','')}:{canonical_text(tag.get('name'))}"


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def load_valid_judgments(
    path: Path | None,
) -> tuple[
    dict[int, set[int]],
    Counter[str],
    Counter[str],
    Counter[str],
    int,
    int,
]:
    accepted: dict[int, set[int]] = {}
    verdicts: Counter[str] = Counter()
    support: Counter[str] = Counter()
    evidence_exists: Counter[str] = Counter()
    if not path:
        return accepted, verdicts, support, evidence_exists, 0, 0
    latest: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("status") == "ok":
            latest[int(row["post_id"])] = row
    valid_shape_posts = sum(
        row.get("automatic_checks", {}).get("valid") is True for row in latest.values()
    )
    for post_id, row in latest.items():
        if row.get("automatic_checks", {}).get("valid") is not True:
            continue
        accepted[post_id] = set()
        for item in row.get("result", {}).get("judgments", []):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item["tag_index"])
            except (KeyError, TypeError, ValueError):
                continue
            verdict = str(item.get("verdict", ""))
            verdicts[verdict] += 1
            support[str(item.get("evidence_support", ""))] += 1
            evidence_exists[str(bool(item.get("evidence_exists"))).lower()] += 1
            if verdict == "accept":
                accepted[post_id].add(index)
    return (
        accepted,
        verdicts,
        support,
        evidence_exists,
        len(latest),
        valid_shape_posts,
    )


def main() -> None:
    args = parse_args()
    (
        accepted,
        judge_verdicts,
        judge_support,
        judge_evidence,
        judge_response_posts,
        judge_valid_shape_posts,
    ) = load_valid_judgments(args.judgments)

    latest_generations: dict[int, dict[str, Any]] = {}
    for record in load_jsonl(args.generations):
        if record.get("status") == "ok":
            latest_generations[int(record["post_id"])] = record

    posts: dict[int, list[dict[str, Any]]] = {}
    schema_valid_posts = 0
    program_tag_total = 0
    program_tag_has_evidence = 0
    text_evidence_total = 0
    text_evidence_exact = 0
    image_evidence_total = 0
    image_claim_nonempty = 0
    generated_tag_total = 0
    iab_name_copy_tags = 0

    for post_id, record in latest_generations.items():
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        tags = result.get("tags") if isinstance(result.get("tags"), list) else []
        checks = record.get("automatic_checks") if isinstance(record.get("automatic_checks"), dict) else {}
        if checks.get("schema_valid"):
            schema_valid_posts += 1
        tag_checks = checks.get("tag_checks") if isinstance(checks.get("tag_checks"), list) else []
        generated_tag_total += len(tags)
        filtered: list[dict[str, Any]] = []
        for index, tag in enumerate(tags):
            if not isinstance(tag, dict):
                continue
            program_tag_total += 1
            if str(tag.get("name", "")).strip() == str(tag.get("iab_name", "")).strip():
                iab_name_copy_tags += 1
            if index < len(tag_checks):
                tag_check = tag_checks[index]
                if tag_check.get("program_evidence_exists"):
                    program_tag_has_evidence += 1
                for evidence_check in tag_check.get("evidence_checks", []):
                    source = evidence_check.get("source")
                    if source in {"title", "content"}:
                        text_evidence_total += 1
                        text_evidence_exact += int(bool(evidence_check.get("exists")))
                    elif source == "image":
                        image_evidence_total += 1
                        image_claim_nonempty += int(bool(evidence_check.get("exists")))
            if args.judgments and index not in accepted.get(post_id, set()):
                continue
            filtered.append(tag)
        posts[post_id] = filtered

    tag_posts: dict[str, set[int]] = defaultdict(set)
    names: dict[str, Counter[str]] = defaultdict(Counter)
    iabs: dict[str, Counter[str]] = defaultdict(Counter)
    assignments = 0
    for post_id, tags in posts.items():
        seen_in_post: set[str] = set()
        for tag in tags:
            key = tag_key(tag)
            if key in seen_in_post:
                continue
            seen_in_post.add(key)
            tag_posts[key].add(post_id)
            names[key][str(tag.get("name", ""))] += 1
            iabs[key][str(tag.get("iab_id", ""))] += 1
            assignments += 1

    post_count = len(posts)
    df = {key: len(value) for key, value in tag_posts.items()}
    assignment_hits = sum(count for count in df.values() if count >= 2)
    singleton_tags = sum(count == 1 for count in df.values())
    overbroad_threshold = max(2, int(post_count * args.overbroad_share))
    overbroad_tags = sum(count > overbroad_threshold for count in df.values())
    posts_with_hit = sum(any(df[tag_key(tag)] >= 2 for tag in tags) for tags in posts.values())
    other_hits = [max(0, count - 1) for count in df.values()]
    judge_total = sum(judge_verdicts.values())

    metrics = {
        "generation_records_ok": post_count,
        "generated_tag_total": generated_tag_total,
        "mean_tags_per_post": round(generated_tag_total / post_count, 6) if post_count else None,
        "schema_valid_post_rate": rate(schema_valid_posts, post_count),
        "program_tag_has_any_evidence_rate": rate(program_tag_has_evidence, program_tag_total),
        "text_evidence_exact_quote_rate": rate(text_evidence_exact, text_evidence_total),
        "image_evidence_nonempty_claim_rate": rate(image_claim_nonempty, image_evidence_total),
        "tag_name_equals_iab_name_rate": rate(iab_name_copy_tags, program_tag_total),
        "evidence_judge_response_posts": judge_response_posts if args.judgments else None,
        "evidence_judge_valid_shape_posts": judge_valid_shape_posts if args.judgments else None,
        "evidence_judge_valid_shape_rate": rate(judge_valid_shape_posts, judge_response_posts)
        if args.judgments else None,
        "evidence_judge_accept_rate": rate(judge_verdicts["accept"], judge_total) if args.judgments else None,
        "evidence_judge_review_rate": rate(judge_verdicts["review"], judge_total) if args.judgments else None,
        "evidence_judge_reject_rate": rate(judge_verdicts["reject"], judge_total) if args.judgments else None,
        "evidence_judge_verdict_counts": dict(judge_verdicts) if args.judgments else None,
        "evidence_judge_support_counts": dict(judge_support) if args.judgments else None,
        "evidence_judge_evidence_exists_counts": dict(judge_evidence) if args.judgments else None,
        "tag_assignments_after_judge_filter": assignments,
        "unique_tag_keys": len(df),
        "assignment_hit_rate": rate(assignment_hits, assignments),
        "singleton_tag_rate": rate(singleton_tags, len(df)),
        "post_retrieval_hit_rate": rate(posts_with_hit, post_count),
        "overbroad_tag_rate": rate(overbroad_tags, len(df)),
        "overbroad_definition": f"document_frequency > {args.overbroad_share:.1%} of posts ({overbroad_threshold})",
        "median_other_posts_per_tag": statistics.median(other_hits) if other_hits else None,
        "mean_other_posts_per_tag": round(statistics.mean(other_hits), 6) if other_hits else None,
        "definitions": {
            "assignment_hit_rate": "帖子-Tag 关系中，该 Tag 还能命中至少一篇其他帖子的比例",
            "singleton_tag_rate": "只被一篇帖子使用的不同 Tag 比例",
            "post_retrieval_hit_rate": "至少有一个 Tag 能找到其他帖子的帖子比例",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tag_stats.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.tag_stats.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["tag_key", "canonical_name", "iab_id", "document_frequency",
                  "other_post_hits", "is_singleton", "is_overbroad"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(df, key=lambda value: (-df[value], value)):
            writer.writerow(
                {
                    "tag_key": key,
                    "canonical_name": names[key].most_common(1)[0][0],
                    "iab_id": iabs[key].most_common(1)[0][0],
                    "document_frequency": df[key],
                    "other_post_hits": max(0, df[key] - 1),
                    "is_singleton": df[key] == 1,
                    "is_overbroad": df[key] > overbroad_threshold,
                }
            )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"output={args.output}")
    print(f"tag_stats={args.tag_stats}")


if __name__ == "__main__":
    main()
