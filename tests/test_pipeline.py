from __future__ import annotations

import sys
import unittest
from pathlib import Path


SIMULATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

from core import (  # noqa: E402
    EVIDENCE_JUDGE_PROMPT,
    IABTaxonomy,
    build_ai_image_url,
    build_generator_prompt,
    normalize_generated_result,
    validate_generated_result,
)
from judge_evidence import validate_judgment  # noqa: E402


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = IABTaxonomy.load()

    def test_taxonomy_and_prompt(self) -> None:
        prompt = build_generator_prompt(self.taxonomy, max_tags=10)
        self.assertGreater(len(self.taxonomy.rows), 700)
        self.assertIn("不是 Tag 候选词表", prompt)
        self.assertIn("信息充分时生成 8～10 个 Tag", prompt)
        self.assertIn("不得少于 6 个", prompt)
        self.assertIn("关键元素是否至少有一个 Tag 覆盖", prompt)
        self.assertIn("允许有限的语义扩展", prompt)
        self.assertIn("壁炉”和“火焰/火光", prompt)
        self.assertIn("海军蓝室内设计", prompt)
        self.assertIn("适度覆盖相似帖子是好事", prompt)
        self.assertIn("Tag name 默认使用简体中文", prompt)
        self.assertIn("Bob头、FIFA、Instagram 应保留原形式", prompt)
        self.assertIn("不得改写或翻译", prompt)
        self.assertIn("visual_claim 用简体中文", prompt)
        self.assertIn("从“萧山”联想到“杭州”", prompt)
        self.assertIn("每篇通常只增加 1 个关系扩展 Tag", prompt)
        self.assertIn("合计绝不能超过 2 个", prompt)
        self.assertIn("人物、角色与作品关系", prompt)
        self.assertIn("风格体系关系", prompt)
        self.assertNotIn("带限定词的具体主题", prompt)
        self.assertNotIn("“主题+形式”“对象+用途”", prompt)
        self.assertIn("只输出 iab_id", prompt)
        self.assertNotIn('"iab_name": "IAB中文名称"', prompt)
        self.assertNotIn("Registry", prompt)
        self.assertNotIn("canonical_tag_id", prompt)
        self.assertNotIn('"type"', prompt)
        self.assertNotIn('"confidence"', prompt)
        self.assertNotIn('"review_flags"', prompt)
        self.assertIn("150|-|景点与娱乐设施", prompt)
        self.assertNotIn("redundancy", EVIDENCE_JUDGE_PROMPT)
        self.assertNotIn("search_value", EVIDENCE_JUDGE_PROMPT)
        self.assertNotIn("suggested_name", EVIDENCE_JUDGE_PROMPT)
        self.assertNotIn('"score"', EVIDENCE_JUDGE_PROMPT)
        self.assertNotIn('"iab_fit"', EVIDENCE_JUDGE_PROMPT)
        self.assertNotIn('"specificity"', EVIDENCE_JUDGE_PROMPT)
        self.assertIn("扩展的召回价值", EVIDENCE_JUDGE_PROMPT)
        self.assertIn("可跨帖子复用", EVIDENCE_JUDGE_PROMPT)
        self.assertIn("可自然翻译的普通概念直接输出成英文 Tag", EVIDENCE_JUDGE_PROMPT)
        self.assertIn("Bob头、FIFA、Instagram", EVIDENCE_JUDGE_PROMPT)
        self.assertIn("跑步”与“有氧运动", EVIDENCE_JUDGE_PROMPT)
        self.assertNotIn("跑步”与“健身", EVIDENCE_JUDGE_PROMPT)

    def test_program_evidence_check(self) -> None:
        post = {"title": "胡适生平讲座", "content": "今天介绍胡适的学术生涯。"}
        result = {
            "tags": [
                {
                    "name": "胡适",
                    "iab_id": "150",
                    "evidence": [{"source": "title", "quote": "胡适"}],
                }
            ]
        }
        normalize_generated_result(result, post, self.taxonomy)
        self.assertEqual(result["tags"][0]["iab_name"], "景点与娱乐设施")
        checks = validate_generated_result(result, post, self.taxonomy, max_tags=10)
        self.assertTrue(checks["schema_valid"])
        self.assertTrue(checks["tag_checks"][0]["program_evidence_exists"])

        result["tags"][0]["evidence"] = [{"source": "title", "quote": "不存在的原文"}]
        checks = validate_generated_result(result, post, self.taxonomy, max_tags=10)
        self.assertFalse(checks["tag_checks"][0]["program_evidence_exists"])

    def test_judge_shape_and_oss_url(self) -> None:
        result = {
            "judgments": [
                {
                    "tag_index": 0,
                    "evidence_exists": True,
                    "evidence_support": "direct",
                    "verdict": "accept",
                    "reason": "证据直接支持该具体标签",
                }
            ]
        }
        self.assertTrue(validate_judgment(result, 1)["valid"])
        processed = build_ai_image_url("https://bucket.oss-cn.test.aliyuncs.com/a.jpg")
        self.assertIn("x-oss-process=image/resize,w_1024/format,jpg", processed)

    def test_deterministic_generation_repairs(self) -> None:
        post = {"title": "", "content": "#人工智能 #midjourney"}
        result = {
            "tags": [
                {
                    "name": "人工智能",
                    "iab_id": "16|1|车型",
                    "evidence": [{"source": "title", "quote": "#人工智能"}],
                }
            ]
        }
        actions = normalize_generated_result(result, post, self.taxonomy)
        self.assertEqual(result["tags"][0]["iab_id"], "16")
        self.assertEqual(result["tags"][0]["iab_name"], "车型")
        self.assertEqual(result["tags"][0]["evidence"][0]["source"], "content")
        self.assertEqual(len(actions), 3)


if __name__ == "__main__":
    unittest.main()
