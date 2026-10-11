# Qwen Multimodal Tagging & Evidence Evaluation

千问多模态内容打标实验：根据标题、正文与封面生成可搜索标签，映射 IAB 分类，再用程序和独立模型校验证据与标签质量。

## 管线

`generate_tags.py` → `judge_evidence.py` → `evaluate.py`

- 保存原有 prompt 版本，支持并发请求、失败重试和按帖子 ID 续跑。
- 文本证据必须能在原文中找到；图像证据与标签有效性交给独立 judge 检查。
- `prompts/` 保存历史 prompt；`results/` 保存原有多次实验的聚合指标，没有帖子明细。
- `data/iab_taxonomy_zh.json` 是既有 IAB 中文分类映射；`posts.example.jsonl` 为新编的两条合成示例，与真实业务数据无关。

## 使用

```bash
pip install -r requirements.txt
cp .env.example .env
# 在本地 .env 填入自己的 DASHSCOPE_API_KEY，并确认模型与 API 地址可用
python generate_tags.py --input data/posts.example.jsonl --output outputs/tags.jsonl --workers 1
python judge_evidence.py --dataset data/posts.example.jsonl --generations outputs/tags.jsonl --output outputs/judgments.jsonl --workers 1
python evaluate.py --generations outputs/tags.jsonl --judgments outputs/judgments.jsonl --output outputs/evaluation.json --tag-stats outputs/tag_stats.csv
python -m unittest discover -s tests -v
```

模型名称保留实验时的配置，运行前请核实服务可用性。仓库没有真实帖子、生产数据导出脚本、实际 `.env` 或 API Key。原始实验指标与当前代码版本不一定一一对应，应按文件名和 prompt 版本阅读。本次整理未重新请求模型 API。

IAB 分类与所依赖的模型/服务保留其原有条款；本仓库未额外为第三方材料授予许可证。
