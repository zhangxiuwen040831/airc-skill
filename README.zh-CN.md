# AIRC

Artificial Intelligence Rewrite Content  
面向 markdown/txt 的结构保护型学术改写工作流。

[English README](./README.md) · [English Usage](./PUBLIC_USAGE.md) · [中文使用说明](./PUBLIC_USAGE.zh-CN.md)

## 一句话简介

AIRC 用于帮助 agent 在严格保留学术文档结构和核心信息的前提下，对普通中文正文进行正文重构式改写，让表达更自然、更连贯，也更接近人工反复修改后的论文文本。

## AIRC 是什么

AIRC 是一个面向 agent 的学术改写工作流。它不是简单的词语替换器，而是先扫描文档、识别风险块、生成块级策略和改写约束，再对可安全处理的正文段进行受控改写，最后通过审核阶段和写入门控决定是否输出文件。

项目中的几个技术标识保持固定：

- 项目名：AIRC
- CLI：`airc`
- Skill ID：`airc-academic-revision`
- Python package / import：`airc-skill` / `airc_skill`

## 为什么使用 AIRC

- 结构保护：标题、公式、引用、数值、占位符、术语、路径、checkpoint 和 Markdown 结构会被严格保护。
- 正文重构：对普通中文正文执行句群级、论述级的自然化改写，而不是只替换几个词。
- Agent-first 工作流：先生成 guidance，让 agent 明确哪些块能改、怎么改、哪些内容不能动。
- 先审后写：只有通过核心一致性、格式完整性、自然度和改写深度检查后，才会写出文件。

## 核心能力

- 块级策略：将内容分类为 `do_not_touch`、`high_risk`、`light_edit` 或 `rewritable`。
- 改写约束：为可改块生成句级、句群级和论述级动作要求。
- 正文重构式改写：对普通正文段进行句群重组、主语链压缩、元话语压缩和衔接调整。
- 自然度检查清单：检查学术语气、重复主语、模板连接词、句子节奏和格式稳定性。
- 审核阶段 / 写入门控：若保护项被破坏，或改写质量不足，输出会被拦截。
- CLI 与 agent 调用：既可以在本地命令行使用，也可以作为 Codex / agent skill 调用。
- Python API：可被本地工具、脚本或未来服务封装直接导入。

## 快速开始

```bash
git clone https://github.com/zhangxiuwen040831/airc-skill.git
cd airc-skill
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .

airc guide examples/sample.md --as-agent-context
airc run examples/sample.md --preset academic_natural --dry-run
```

macOS / Linux 激活虚拟环境：

```bash
source .venv/bin/activate
```

开发测试依赖：

```bash
python -m pip install -e ".[dev]"
```

## 工作流程

```text
guide -> rewrite -> review -> write
```

1. `guide`：扫描文档，生成块级策略、保护项和 agent 可执行的改写说明。
2. `rewrite`：只对允许处理的块执行改写，并满足对应改写约束。
3. `review`：检查核心内容、格式、自然度、改写覆盖率和论述变化。
4. `write`：只有通过写入门控后，才生成改写后的文件。

## CLI 用法

查看命令：

```bash
airc --help
```

生成 agent 上下文：

```bash
airc guide examples/sample.md --as-agent-context
```

只预览，不写文件：

```bash
airc run examples/sample.md --preset academic_natural --dry-run
```

运行并在通过审核后写出：

```bash
airc run examples/sample.md --preset academic_natural
```

使用更强的正文改写预设：

```bash
airc run examples/sample.md --preset aggressive_rewrite --json-report
```

审核或写入外部候选文件：

```bash
airc review original.md candidate.md
airc write original.md candidate.md
```

## 其他人如何使用 AIRC

### 1. 本地 CLI

```bash
airc run paper.md --preset academic_natural
```

这条命令会自动执行 `guide -> rewrite -> review -> write` 流程。若候选文本没有通过核心内容、格式完整性或自然度检查，AIRC 不会写出最终文件，而是返回失败原因与报告信息。

### 2. Agent / Codex

可复制的调用模板：

```text
Use the skill "airc-academic-revision".
Revise the academic markdown file with structure-preserving developmental rewrite.
Preserve titles, formulas, citations, numbers, figures, placeholders, and markdown structure.
Run: guide -> rewrite -> review -> write.
```

这段提示让 agent 使用 Skill ID `airc-academic-revision`，先读取块级策略，再对可改正文块执行正文重构式改写，并在写入前完成审核。

### 3. Python API

```python
from airc_skill import SkillInputSchema, guide_document, run_revision

schema = SkillInputSchema.from_path(
    "paper.md",
    preset="academic_natural",
    emit_json_report=True,
)

guidance, execution_plan = guide_document(schema)
result = run_revision(schema)

print(result.output_schema.status)
print(result.output_schema.rewrite_coverage)
print(result.output_schema.write_gate_decision)
```

`guidance` 和 `execution_plan` 描述哪些块可改、哪些块必须保留、每个块需要执行哪些动作；`result` 包含输出路径、改写覆盖率、论述变化分数、警告和写入门控结果。

## Presets

- `academic_natural`：默认预设，适合普通学术正文的自然化润色。
- `aggressive_rewrite`：更强的正文重构式改写，适合希望普通正文变化更明显的场景。
- `conservative`：更保守的轻改模式，适合风险较高的文档。
- `body_only`：跳过标题、摘要、图注、参考文献等高风险部分，聚焦正文段。

## AIRC 会保护什么

AIRC 默认保护：

- 标题、小标题和标题编号
- 公式与公式编号
- 引用标记与引用顺序
- 数值、年份、阈值、变量、样本量和实验指标
- 占位符、图片引用、链接、表格和代码块
- Markdown 结构和特殊符号
- 专业术语、模型名、数据集名、算法名和系统名
- 文件路径、checkpoint 名称、文件名和版本号

只要这些保护项在候选文本中发生不一致，审核阶段会失败，写入门控会阻止输出。

## AIRC 会重写什么

AIRC 只重写安全的正文块。它可以：

- 压缩连续出现的 `本研究 / 本文 / 该系统` 主语链
- 减少 `首先 / 其次 / 此外 / 因此 / 同时` 等模板连接词
- 控制不必要的 `了`、堆叠的 `的` 和重复的 `在……中`
- 打散过于整齐的排比句式
- 重建长短句节奏
- 重组相邻句群
- 将结论承接句自然吸收到上下文论述中

术语密集、引用密集或格式敏感的技术段会保持保守。

## 输入与输出

| 格式 | 状态 | 行为 |
| --- | --- | --- |
| `.md` | 稳定 | 直接作为 Markdown 读取。 |
| `.txt` | 稳定 | 作为纯文本读取，并归一化为 Markdown。 |
| `.docx` | 稳定 | 通过 pandoc 转为 Markdown。 |
| `.doc` | 实验性 | 尝试通过 pandoc 转换；失败时建议先转为 `.docx`。 |

Pandoc 只在处理 `.docx` 或 `.doc` 时需要。

典型输出：

- `paper.airc.md`：通过写入门控后生成的改写 Markdown。
- `paper.airc.report.json`：机器可读的改写报告。

报告中常用字段：

- `execution_plan`：块级策略、改写约束、保护项和 agent 指令。
- `review.rewrite_coverage`：正文句级 / 句群级改写覆盖率。
- `review.discourse_change_score`：论述层变化分数。
- `review.cluster_rewrite_score`：相邻句群重构信号。
- `review.natural_revision_checklist`：自然度检查清单。
- `write_gate`：最终写入决策和原因码。
- `failure_transparency`：失败块、未满足约束、当前分数和重试建议。

## 使用范围与限制

AIRC 适合：

- 学术 markdown / txt 草稿
- 中文学术正文段
- 背景、意义、风险分析、方法解释等普通叙述段
- 需要透明报告与写入门控的本地工作流

AIRC 不会：

- 改变事实、结论、引用、公式、数值或术语
- 编造文献、数据集、指标、案例或实验结果
- 自由改写图注、表格、代码块、公式、链接、占位符或路径
- 替代人工对最终论文质量的判断

默认改写器是确定性、离线运行的。`.docx` 支持依赖 pandoc；`.doc` 支持仍为实验性。

## 开发状态 / 发布状态

当前版本：`0.2.0b0` public beta。

查看英文发布记录：[CHANGELOG.md](./CHANGELOG.md)  
查看中文操作说明：[PUBLIC_USAGE.zh-CN.md](./PUBLIC_USAGE.zh-CN.md)

运行测试：

```bash
python -m pytest -q
```

## License

MIT License. See [LICENSE](./LICENSE).
