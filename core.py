#!/usr/bin/env python3
"""标签模拟实验的共享工具：IAB、Prompt、Qwen 请求、JSONL 与证据校验。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
import urllib.parse
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


HERE = Path(__file__).resolve().parent
TAG_SYSTEM_DIR = HERE.parent
DEFAULT_TAXONOMY = (
    HERE / "data" / "iab_taxonomy_zh.json"
)
DEFAULT_DATASET = HERE / "data" / "posts.example.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "outputs"


def load_local_env(path: Path = HERE / ".env") -> None:
    """Load simple KEY=VALUE entries without overriding shell environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()

GENERATOR_PROMPT_VERSION = "iab-tags-v3.6.1-evidence-required"
EVIDENCE_JUDGE_PROMPT_VERSION = "tag-quality-judge-v2.6-limited-relational-expansion"

VALID_EVIDENCE_SOURCES = {"title", "content", "image"}
GENERIC_SUFFIXES = ("相关", "内容", "分享", "推荐", "其他", "类")
TITLE_CONTENT_LIMIT = 30_000


@dataclass(frozen=True)
class IABTaxonomy:
    rows: list[dict[str, Any]]
    by_id: dict[str, dict[str, Any]]
    compact_text: str

    @classmethod
    def load(cls, path: Path = DEFAULT_TAXONOMY) -> "IABTaxonomy":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"IAB taxonomy 不是非空数组: {path}")
        by_id: dict[str, dict[str, Any]] = {}
        compact: list[str] = []
        for row in rows:
            item_id = str(row.get("id", "")).strip()
            name = str(row.get("zh", "")).strip()
            parent = str(row.get("parent", "")).strip() or "-"
            if not item_id or not name:
                raise ValueError(f"IAB 节点缺失 id/zh: {row}")
            by_id[item_id] = row
            compact.append(f"{item_id}|{parent}|{name}")
        return cls(rows=rows, by_id=by_id, compact_text="\n".join(compact))

    def name(self, item_id: str) -> str:
        return str(self.by_id.get(str(item_id), {}).get("zh", "")).strip()

    def path(self, item_id: str) -> list[dict[str, str]]:
        current = str(item_id)
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        while current and current in self.by_id and current not in seen:
            seen.add(current)
            row = self.by_id[current]
            result.append({"id": current, "name": str(row.get("zh", "")).strip()})
            current = str(row.get("parent", "")).strip()
        return list(reversed(result))


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON 解析失败: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            yield value


def append_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    handle.flush()


def completed_post_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {
        int(row["post_id"])
        for row in load_jsonl(path)
        if str(row.get("post_id", "")).isdigit() and row.get("status") == "ok"
    }


def canonical_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def char_ngrams(value: Any, n: int = 2) -> set[str]:
    text = canonical_text(value)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型响应中没有 JSON 对象")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型响应 JSON 不是对象")
    return value


def build_ai_image_url(url: str, process: str = "image/resize,w_1024/format,jpg") -> str:
    if not process or "aliyuncs.com" not in url or "x-oss-process=" in url:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("x-oss-process", process))
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query, safe="/,"),
            parsed.fragment,
        )
    )


class QwenClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        temperature: float = 0.1,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> None:
        if not api_key or api_key in {"sk-xxx", "..."} or not base_url or not model:
            raise ValueError("Qwen API 缺少 api_key/base_url/model")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

    @classmethod
    def from_env(
        cls,
        *,
        model_env: str = "QWEN_VL_MODEL",
        default_model: str = "qwen3.7-flash",
        temperature: float = 0.1,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> "QwenClient":
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.getenv("QWEN_BASE_URL", "").strip(),
            model=os.getenv(model_env, default_model).strip(),
            temperature=temperature,
            timeout=timeout,
            retries=retries,
        )

    def chat(
        self,
        system_prompt: str,
        user_text: str,
        *,
        image_url: str | None = None,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        if image_url:
            user_content: Any = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": build_ai_image_url(image_url)}},
            ]
        else:
            user_content = user_text
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if not response.ok:
                    detail = response.text.strip().replace("\n", " ")[:1000]
                    raise RuntimeError(f"HTTP {response.status_code}: {detail}")
                body = response.json()
                raw = str(body["choices"][0]["message"]["content"]).strip()
                return extract_json_object(raw), raw, body.get("usage", {})
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Qwen 请求失败: {last_error}")


GENERATOR_PROMPT_TEMPLATE = """你是 Nalook 的多模态内容 Tag 生成器。结合标题、正文和封面图，生成覆盖充分、可搜索、可复用的 Tag。

IAB Content Taxonomy 只用于限制内容范围并为 Tag 提供归属类目，不是 Tag 候选词表。Tag 负责表达帖子“具体在讲什么”，IAB 负责表达它“属于哪一类”。本任务不对 IAB 类目分主次；每个 Tag 独立选择一个最符合其语境的 IAB 节点。

## Tag 原则

1. 信息充分时生成 8～{max_tags} 个 Tag，目标为 9 个；不要因为担心重复或 Evidence 而主动少打。只有标题、正文和图像都极度缺乏信息时才允许少于 8 个，但不得少于 6 个。
2. 先在内部识别帖子最重要的主题和视觉元素，再生成 Tag；不要输出识别过程。
3. 标题明确出现的核心名词、动作、时间或风格，以及图像中清晰、显著的主体和关键元素，应优先被 Tag 覆盖。不得只用一个宽泛 Tag 吸收多个重要元素。
4. 每个 Tag 应表达一个主要检索概念，但自然短语、固定术语、约定俗成的品类名和专有名词可以保留为整体，不要机械拆词。
5. Tag 应使用跨帖子可复用的常用粒度，避免一次性、画面说明式的长标签和不必要的修饰词；适度覆盖相似帖子是好事。
6. 默认禁止 name 与其 IAB 节点名称相同。只有当 IAB 名称本身就是帖子明确表达、用户会直接搜索的核心主题时才可相同；每帖最多保留一个这种例外。
7. 不同 Tag 不得完全重复或仅作无意义改写，但允许有限的语义扩展：当搜索范围确实不同且与帖子直接相关时，可以同时保留具体词与上位词、常用近义词或关联概念，每帖最多 2 个此类扩展 Tag。
8. 允许基于标题、正文和清晰图像做合理的通用联想，例如从“跑步”联想到“有氧运动”，从“格纹西装”联想到“男装”，从“萧山”联想到“杭州”；不得凭空猜测人物身份、品牌、地点或具体事件。
9. Tag name 默认使用简体中文。即使标题、正文、账号名或图像文字是英文，普通概念以及可自然翻译的品类、颜色、动作、风格、服装和物件等也必须使用常用中文，不得直接照抄英文。例如：Thrift→二手时尚、Vintage→复古风格、Street Style→街头风格、Orange Clothing→橙色服装、Wide Leg Pants→阔腿裤、Crop Top→短款上衣或露脐上衣、Gold Chain Necklace→金色链条项链、Tattoo→纹身、Sunglasses→太阳镜。
10. 人名、品牌、作品、产品型号、组织、平台、账号名、缩写、技术名称以及约定俗成的中英混合词等特殊词，使用用户最常搜索的规范名称；只有缺乏通行中文译名，或英文形式本身就是主要检索入口时才保留英文。例如 Bob头、FIFA、Instagram 应保留原形式。已有成熟中文名称时优先使用中文；不得仅因为输入是英文就把普通概念输出为英文 Tag。
11. 文字和图像冲突时，结合全部输入选择有证据支持、最能表达帖子主题的 Tag。

## 有限关系扩展

在确实存在明确关系且能增加搜索召回时，可以从以下三类关系中增加扩展 Tag；这不是必选项，没有合适关系时不要硬加：

1. 地理包含关系：具体地点可以向上扩展一级，例如“萧山”→“杭州”、“西湖”→“杭州”。必须保留具体地点 Tag；不得连续扩展成“杭州”→“浙江”→“中国”，存在同名地点或归属不确定时不得扩展。
2. 人物、角色与作品关系：帖子明确涉及某个人物或角色时，可以补充其直接所属的作品、系列或内容宇宙，例如“五条悟”→“咒术回战”、“钢铁侠”→“漫威”。只有身份及所属关系有充分依据时才可扩展，不得凭相似外观猜测角色或作品。
3. 风格体系关系：具体风格可以向一个稳定、常用且更宽的风格概念扩展，例如“侘寂风”在内容确实体现相应特征时可扩展为“极简风”或“自然风”。风格关系具有主观性，必须有清晰的文字或视觉依据；不得仅凭单一颜色、材料或普通构图牵强扩展。

关系扩展必须遵守：
- 原始的具体 Tag 必须保留，扩展 Tag 不能替代它；
- 每篇通常只增加 1 个关系扩展 Tag；只有同时存在两个独立、明确且重要的关系时才允许增加 2 个，合计绝不能超过 2 个，并且每条关系只允许扩展一级；
- 扩展 Tag 必须能形成有意义的独立搜索入口，不得为了达到 Tag 数量要求而添加；
- 扩展 Tag 同样需要 Evidence；Evidence 应支持原始概念，二者之间必须是上述稳定关系。

## 覆盖要求

在输入确实包含相关信息时，尽量从以下维度覆盖至少 4 类，而不是集中描述一个方面：
- 核心主题或内容类型；
- 主要人物、物件、产品、动物或植物；
- 关键动作、用途或活动；
- 场景、地点类型或空间；
- 风格、设计流派或审美；
- 颜色、材料、纹理、图案或形态；
- 时间、天气、氛围或情绪；
- 明确出现的品牌、作品、人物或地点名称。

关键元素覆盖示例：
- 标题为“黄金时刻壁炉火”，图像清晰显示火焰时，“壁炉”和“火焰/火光”应分别覆盖；不能只输出“壁炉”或“露营”。
- “海军蓝室内设计”通常拆为“海军蓝”和“室内设计”。
- “跑步”与“有氧运动”搜索范围不同，可以同时存在；“跑步”与“跑步运动”几乎没有新增检索价值，不要同时输出。

生成前在内部逐项检查，但不要输出检查过程：
- 这个 name 是否只是复制 IAB 名称？
- 它是否比 IAB 提供了新的、可搜索的信息？
- 标题和图像中的每个关键元素是否至少有一个 Tag 覆盖？
- Tag 是否覆盖了多个信息维度，而不是都集中在同一方面？
- 它与另一个 Tag 是无意义重复，还是能带来不同搜索范围的合理扩展？
- 关系扩展是否属于地理包含、人物/角色与作品、风格体系三类之一，是否通常只有 1 个、最多 2 个、只扩展一级且没有为了凑数量硬加？
- 它能否自然复用于其他同主题帖子？
- 普通英文概念是否已转换为常用简体中文，保留的英文是否确属专有词或主要检索入口？
- Evidence 是否足以支持这个 Tag？

## IAB 规则

1. 只输出 iab_id；iab_name 将由程序根据词表自动填充。
2. iab_id 只能填每行第一列的纯 ID，绝不能复制整行。例如词表行 `16|1|车型`，必须输出 `"iab_id":"16"`，不得输出 `"16|1|车型"`。
3. 选择最深且证据充分的节点；无法支持具体子类时选更稳妥的上级节点。
4. 战争与冲突、灾害、犯罪、敏感话题都是正常可选类目，不得回避。

## Evidence 规则

1. 每个 Tag 提供一条 evidence。
2. 文字证据 source 使用 title 或 content，quote 必须是输入文本中真实存在的连续短语，不得改写或翻译；原文是英文时 quote 也必须保留英文原文。
3. source 必须指向 quote 真正所在的字段；标题为空时不得使用 title evidence。
4. 一条 quote 只能引用一个连续片段，禁止用 `...`、省略号或空格拼接多个不连续词组。
5. 图像证据 source 使用 image，visual_claim 用简体中文写一句简短事实描述可观察内容；其中确有必要的专有词可以保留英文。不得写分析过程、自我质疑、修正意见或“应移除”等内部备注。
6. 对合理联想或上位概念，可以引用支持该联想的标题、正文或视觉事实；不得把联想本身伪装成直接可见事实。

## 输出

只输出一个 JSON 对象，不要 Markdown、分析过程或额外解释：

{{
  "tags": [
    {{
      "name": "默认使用简体中文的简洁、可搜索、可复用规范Tag",
      "iab_id": "IAB节点ID",
      "evidence": [{{"source": "title|content", "quote": "原文连续短语"}}]
    }}
  ]
}}

只允许输出上面列出的字段，不要添加其他字段。

## IAB Content Taxonomy

每行格式为 id|parent_id|name，parent_id 为 - 表示根节点：

{taxonomy}
"""


def build_generator_prompt(
    taxonomy: IABTaxonomy,
    *,
    max_tags: int,
) -> str:
    return GENERATOR_PROMPT_TEMPLATE.format(
        max_tags=max_tags,
        taxonomy=taxonomy.compact_text,
    ).strip()


def build_post_user_text(post: dict[str, Any]) -> str:
    title = str(post.get("title") or "").strip()
    content = str(post.get("content") or "").strip()
    if len(content) > TITLE_CONTENT_LIMIT:
        content = content[:TITLE_CONTENT_LIMIT]
    payload = {"title": title, "content": content}
    return (
        "请为下列帖子生成 Tag。title 与 content 是两个独立字段，"
        "Evidence source 必须与 quote 所在字段一致。"
        "每个 Tag 都必须包含且只包含一条 evidence，任何 Tag 缺少 evidence 都视为整个响应无效。\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def normalize_generated_result(
    result: dict[str, Any],
    post: dict[str, Any],
    taxonomy: IABTaxonomy,
) -> list[dict[str, Any]]:
    """Apply only deterministic formatting repairs and record every change."""
    actions: list[dict[str, Any]] = []
    tags = result.get("tags")
    if not isinstance(tags, list):
        return actions
    title = str(post.get("title") or "")
    content = str(post.get("content") or "")
    for index, tag in enumerate(tags):
        if not isinstance(tag, dict):
            continue
        raw_iab_id = str(tag.get("iab_id", "")).strip()
        if "|" in raw_iab_id:
            candidate = raw_iab_id.split("|", 1)[0].strip()
            if candidate in taxonomy.by_id:
                tag["iab_id"] = candidate
                actions.append(
                    {
                        "tag_index": index,
                        "field": "iab_id",
                        "from": raw_iab_id,
                        "to": candidate,
                        "reason": "extracted_leftmost_taxonomy_id",
                    }
                )
        normalized_iab_id = str(tag.get("iab_id", "")).strip()
        if normalized_iab_id in taxonomy.by_id:
            expected_name = taxonomy.name(normalized_iab_id)
            previous_name = tag.get("iab_name")
            if previous_name != expected_name:
                tag["iab_name"] = expected_name
                actions.append(
                    {
                        "tag_index": index,
                        "field": "iab_name",
                        "from": previous_name,
                        "to": expected_name,
                        "reason": "filled_from_taxonomy_by_iab_id",
                    }
                )
        evidences = tag.get("evidence")
        if not isinstance(evidences, list):
            continue
        for evidence_index, evidence in enumerate(evidences):
            if not isinstance(evidence, dict):
                continue
            source = str(evidence.get("source", ""))
            quote = str(evidence.get("quote", ""))
            if not quote or source not in {"title", "content"}:
                continue
            claimed_text = title if source == "title" else content
            other_source = "content" if source == "title" else "title"
            other_text = content if source == "title" else title
            if quote not in claimed_text and quote in other_text:
                evidence["source"] = other_source
                actions.append(
                    {
                        "tag_index": index,
                        "evidence_index": evidence_index,
                        "field": "evidence.source",
                        "from": source,
                        "to": other_source,
                        "reason": "quote_exists_exactly_in_other_text_field",
                    }
                )
    return actions


def validate_generated_result(
    result: dict[str, Any],
    post: dict[str, Any],
    taxonomy: IABTaxonomy,
    *,
    max_tags: int,
) -> dict[str, Any]:
    tags = result.get("tags")
    errors: list[str] = []
    if not isinstance(tags, list):
        tags = []
        errors.append("tags_not_list")

    title = str(post.get("title") or "")
    content = str(post.get("content") or "")
    normalized_names: list[str] = []
    tag_checks: list[dict[str, Any]] = []
    valid_tag_count = 0
    for index, tag in enumerate(tags):
        check: dict[str, Any] = {"index": index, "errors": []}
        if not isinstance(tag, dict):
            check["errors"].append("tag_not_object")
            tag_checks.append(check)
            continue
        name = str(tag.get("name", "")).strip()
        iab_id = str(tag.get("iab_id", "")).strip()
        iab_name = str(tag.get("iab_name", "")).strip()
        evidences = tag.get("evidence")
        if not name:
            check["errors"].append("empty_name")
        else:
            normalized_names.append(canonical_text(name))
        if iab_id not in taxonomy.by_id:
            check["errors"].append("invalid_iab_id")
        elif iab_name != taxonomy.name(iab_id):
            check["errors"].append("iab_name_mismatch")
        check["name_equals_iab_name"] = bool(name and iab_name and name == iab_name)
        if not isinstance(evidences, list) or not evidences:
            check["errors"].append("missing_evidence")
            evidences = []

        evidence_checks: list[dict[str, Any]] = []
        for evidence_index, evidence in enumerate(evidences):
            echeck: dict[str, Any] = {"index": evidence_index, "exists": False, "errors": []}
            if not isinstance(evidence, dict):
                echeck["errors"].append("evidence_not_object")
                evidence_checks.append(echeck)
                continue
            source = str(evidence.get("source", "")).strip()
            echeck["source"] = source
            if source not in VALID_EVIDENCE_SOURCES:
                echeck["errors"].append("invalid_source")
            elif source in {"title", "content"}:
                quote = str(evidence.get("quote", ""))
                haystack = title if source == "title" else content
                position = haystack.find(quote) if quote else -1
                echeck.update(
                    {
                        "quote": quote,
                        "exists": position >= 0,
                        "start": position if position >= 0 else None,
                        "end": position + len(quote) if position >= 0 else None,
                    }
                )
                if position < 0:
                    echeck["errors"].append("quote_not_in_source")
            elif source == "image":
                claim = str(evidence.get("visual_claim", "")).strip()
                echeck.update({"visual_claim": claim, "exists": bool(claim)})
                if not claim:
                    echeck["errors"].append("empty_visual_claim")
            evidence_checks.append(echeck)
        check["evidence_checks"] = evidence_checks
        check["program_evidence_exists"] = any(item["exists"] for item in evidence_checks)
        if not check["errors"]:
            valid_tag_count += 1
        tag_checks.append(check)

    duplicates = len([name for name in normalized_names if name]) - len(
        {name for name in normalized_names if name}
    )
    if duplicates:
        errors.append("duplicate_tag_names")
    if len(tags) > max_tags:
        errors.append("too_many_tags")
    if not tags:
        errors.append("no_tags")
    generic_count = sum(
        any(str(tag.get("name", "")).endswith(suffix) for suffix in GENERIC_SUFFIXES)
        for tag in tags
        if isinstance(tag, dict)
    )
    iab_name_copy_count = sum(
        bool(check.get("name_equals_iab_name")) for check in tag_checks
    )
    return {
        "schema_valid": not errors and valid_tag_count == len(tags),
        "errors": sorted(set(errors)),
        "tag_count": len(tags),
        "valid_tag_count": valid_tag_count,
        "duplicate_tag_count": duplicates,
        "generic_suffix_tag_count": generic_count,
        "iab_name_copy_tag_count": iab_name_copy_count,
        "tag_checks": tag_checks,
    }


EVIDENCE_JUDGE_PROMPT = """你是独立的 Tag Quality Judge。你评估帖子是否支持已生成的 Tag，以及 Tag 是否有实际搜索与召回价值。你不会看到程序检查结论，必须独立阅读原始 title、content、图像和完整 Tag 集合。

IAB 是 Tag 的归属类目，不是 Tag 候选词表。优质 Tag 应有证据基础、粒度自然并可跨帖子复用。不要把“越具体越好”或“只要存在上下位关系就是重复”作为判断原则。

判断前在内部完成以下检查，但不要单独输出检查结果：
- Tag 与 IAB 节点的归属是否合理。
- Tag 是否包含证据不支持的限定，或细到只能描述当前画面的临时组合。
- Tag 与另一 Tag 是否完全同义、仅作无意义改写；还是虽然相关，却能覆盖不同搜索范围。
- 上位词、常用近义词或关联概念是否与帖子直接相关，并能增加召回价值。
- 复合名称是否为专有名词、固定术语、约定俗成的品类名或稳定搜索概念；这类名称不得机械拆分。
- Tag name 是否遵守中文规范：普通概念以及可自然翻译的品类、颜色、动作、风格、服装和物件必须使用常用简体中文；只有缺乏通行中文译名或英文形式本身是主要检索入口的专有词才可保留英文。
- 文字 quote 必须保留输入原文语言；图像 visual_claim 应使用简体中文，必要的专有词除外。
- 若 Tag 属于关系扩展，是否确属地理包含、人物/角色与作品、风格体系三类之一；是否保留了原始具体 Tag、通常只有 1 个且最多 2 个、只扩展一级且没有牵强补充。

1. evidence_exists：给定证据是否确实来自它声明的 title、content 或图像；source 写反时必须为 false。
2. evidence_support：只能是 direct、strong_inference、weak_inference、unsupported、contradicted。
3. verdict：只能是 accept、review、reject。
   - accept：证据充分或有合理强推断，IAB 归属合理，Tag 本身可搜索且能提供独立或扩展的召回价值。
   - review：联想关系较弱、IAB 归属有争议、无法确定两个相关 Tag 是否带来不同搜索范围、地理归属或风格体系关系存在歧义，或 Tag 本身合格但 image visual_claim 未按要求使用中文。
   - reject：证据不支持或冲突、IAB 归属错误、名称含无依据限定、过度细碎、与另一 Tag 的搜索意图几乎完全相同且没有新增信息、把可自然翻译的普通概念直接输出成英文 Tag，或关系扩展超出一级、所属关系错误、没有保留具体概念且只剩宽泛概念。

判断示例：
- “壁炉”与“火焰”描述不同关键元素，可以同时 accept。
- “跑步”与“有氧运动”搜索范围不同；帖子明确涉及跑步时，“有氧运动”可以作为合理上位扩展，不应仅因父子关系 reject。
- “格纹西装”与“男装”可以分别承担具体对象和上位召回，只要图文支持即可接受。
- “跑步”与“跑步运动”几乎没有新增信息，后者应 reject。
- “海军蓝室内设计”通常应拆为“海军蓝”和“室内设计”；若作为临时组合且无稳定搜索价值，应 reject。
- 英文输入中的 Vintage、Street Style、Wide Leg Pants、Crop Top、Tattoo、Sunglasses 等普通概念应分别规范为“复古风格”“街头风格”“阔腿裤”“短款上衣/露脐上衣”“纹身”“太阳镜”；若直接使用英文 Tag，应 reject。
- Bob头、FIFA、Instagram、iPhone、Nike、ChatGPT、Python、NBA、Etsy、OOTD 等约定俗成的中英混合词、品牌、平台、产品、技术名、组织名或通行缩写可以保留原形式；不得仅因含英文字母就机械 reject。
- 帖子明确涉及“萧山”时，同时保留“萧山”和一级上位地点“杭州”可以增加地理召回，不应仅因上下位关系 reject；若继续补充“浙江”“中国”则属于过度扩展。
- 帖子明确涉及“五条悟”时，同时保留“五条悟”和其作品“咒术回战”可以增加作品检索召回；身份或所属作品无法确认时应 review 或 reject。
- “侘寂风”扩展到“极简风”或“自然风”必须有整体风格依据；若仅因画面出现米色或木材就扩展，应 reject。

reason 必须简短说明真正影响 verdict 的原因；若判定重复，写明与哪个 Tag 的搜索意图几乎相同。不要因为 Tag 与 IAB 同名或存在上下位关系就机械 reject。

只输出 JSON，不要 Markdown 或额外解释：
{"judgments":[{"tag_index":0,"evidence_exists":true,"evidence_support":"direct","verdict":"accept","reason":"简短原因"}]}
"""


def build_evidence_judge_user_text(
    post: dict[str, Any],
    generation_record: dict[str, Any],
    taxonomy: IABTaxonomy,
) -> str:
    result = generation_record.get("result", {})
    tags = result.get("tags", []) if isinstance(result, dict) else []
    judge_items: list[dict[str, Any]] = []
    for index, tag in enumerate(tags):
        if not isinstance(tag, dict):
            continue
        iab_id = str(tag.get("iab_id", ""))
        judge_items.append(
            {
                "tag_index": index,
                "tag": tag,
                "iab_path": taxonomy.path(iab_id),
            }
        )
    title = str(post.get("title") or "")
    content = str(post.get("content") or "")
    if len(content) > TITLE_CONTENT_LIMIT:
        content = content[:TITLE_CONTENT_LIMIT]
    return (
        f"标题：{title}\n\n正文：\n{content}\n\n"
        "待评估 Tag：\n"
        + json.dumps(judge_items, ensure_ascii=False)
    )


def prompt_manifest(prompt: str, *, version: str, model: str) -> dict[str, Any]:
    return {
        "version": version,
        "sha256": sha256_text(prompt),
        "characters": len(prompt),
        "model": model,
    }


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return sum(items) / len(items) if items else None
