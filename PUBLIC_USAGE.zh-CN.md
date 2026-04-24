# AIRC 中文使用说明

Artificial Intelligence Rewrite Content  
面向 markdown/txt 的结构保护型学术改写工作流。

[README 中文版](./README.zh-CN.md) · [English Usage](./PUBLIC_USAGE.md) · [English README](./README.md)

## 概述

AIRC 是一个本地 academic rewrite workflow，用于帮助 agent 对学术草稿进行结构保护型改写。它会先识别哪些内容不能动，再对安全的正文块执行正文重构式改写，最后通过审核阶段和写入门控决定是否写出文件。

固定标识：

- 项目名：AIRC
- CLI：`airc`
- Skill ID：`airc-academic-revision`
- Python package / import：`airc-skill` / `airc_skill`

## 安装

```bash
git clone https://github.com/zhangxiuwen040831/airc-skill.git
cd airc-skill
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

macOS / Linux：

```bash
source .venv/bin/activate
```

如需处理 `.docx` 或 `.doc`，请安装 pandoc：

```bash
winget install JohnMacFarlane.Pandoc
```

Pandoc 只影响 `.docx` / `.doc` 输入；`.md` 和 `.txt` 不需要 pandoc。

## CLI 使用

生成 agent 指令上下文：

```bash
airc guide paper.md --as-agent-context
```

预览改写结果，不写文件：

```bash
airc run paper.md --preset academic_natural --dry-run
```

执行完整流程，并在通过写入门控后输出文件：

```bash
airc run paper.md --preset academic_natural
```

对普通正文执行更强的重构式改写：

```bash
airc run paper.md --preset aggressive_rewrite --json-report
```

审核或写入外部 agent 生成的候选文件：

```bash
airc review paper.md candidate.md
airc write paper.md candidate.md
```

## Agent 使用

可复制的调用模板：

```text
Use the skill "airc-academic-revision".
Revise the academic markdown file with structure-preserving developmental rewrite.
Preserve titles, formulas, citations, numbers, figures, placeholders, and markdown structure.
Run: guide -> rewrite -> review -> write.
```

Agent 应先读取 `guide` 输出的块级策略：

- `do_not_touch`：原样保留，只做格式检查。
- `high_risk`：默认保守，只允许极轻微清理。
- `light_edit`：允许句级润色，但不做大幅重构。
- `rewritable`：必须执行正文重构式改写，不能只做句首替换或近义词替换。

## Python 使用

```python
from airc_skill import SkillInputSchema, guide_document, run_revision

schema = SkillInputSchema.from_path("paper.md", preset="academic_natural")
guidance, plan = guide_document(schema)
result = run_revision(schema)

print(result.output_schema.status)
print(result.output_schema.rewritten_file_path)
print(result.output_schema.write_gate_decision)
```

含义：

- `SkillInputSchema` 描述输入路径、预设、报告输出等调用参数。
- `guide_document` 返回 guidance 与执行计划，供 agent 判断哪些块能改。
- `run_revision` 执行完整流程，并返回结构化结果。

如果候选文件由其他 agent 生成，可以使用：

```python
from airc_skill import review_candidate, write_candidate

review = review_candidate("paper.md", "candidate.md", schema)
write = write_candidate("paper.md", "candidate.md", "paper.airc.md", schema)
```

## 输出文件说明

常见输出：

- `paper.airc.md`：通过写入门控后生成的改写文件。
- `paper.airc.report.json`：JSON 报告，适合 agent、脚本或 Web/API 服务读取。

报告中建议重点查看：

- `execution_plan`：块级策略、改写约束和保护项。
- `review.rewrite_coverage`：正文改写覆盖率。
- `review.discourse_change_score`：论述层变化分数。
- `review.cluster_rewrite_score`：句群重构分数。
- `review.natural_revision_checklist`：自然度检查清单。
- `write_gate`：是否允许写入及原因。
- `failure_transparency`：失败块、未满足约束和下一步建议。

## 注意事项

- 稳定输入格式为 `.md`、`.txt`、`.docx`。
- `.doc` 支持为实验性；转换失败时建议先转为 `.docx`。
- AIRC 不会修改标题、公式、引用、数值、专业术语、路径、checkpoint、图注、表格、代码块、占位符或 Markdown 结构。
- AIRC 不会编造文献、数据、指标、样本、案例或实验结果。
- 如果文档中术语、引用、数字或路径非常密集，AIRC 会自动偏向保守处理。
- 最终论文质量仍需要人工确认，AIRC 提供的是结构保护、正文重构和审核门控能力。
