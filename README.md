# AIRC.skill

AIRC.skill 是一个面向 agent 的学术润色规程与本地工具集，用于处理本地 `.md` 与 `.txt` 文件，在不改变论文核心内容的前提下，优化叙述方式、提升自然度，并让文本更接近人工润色后的学术表达。

它的核心定位不是“黑盒全文替换器”，而是一个 agent-first 的论文润色规程系统：

- 先扫描文档，识别不可改块、高风险段、轻改段和可改段
- 再向 agent 提供段落级 rewrite guidance、禁止动作和自检清单
- 然后由 agent 按规则逐段执行改写
- 最后通过 review 与 write gate 决定是否允许落盘

当前版本是面向 GitHub 试用的 beta 版：

- 支持 `guide` 模式，先输出 block policy 与 agent guidance，再决定是否进入改写
- 支持 `run --preset` 公共接口，输出改写文件与 JSON rewrite report
- 支持 `.docx` 输入归一化，优先使用 pandoc 转成 Markdown
- 支持独立 `review` 与 `write` 步骤
- 默认使用本地规则法改写，不依赖在线 API
- 支持 `conservative` / `balanced` / `strong` 三种改写强度
- 支持 `dry-run`，只输出 diff 预览，不写文件
- 默认只在通过最小质量门槛且产生有效改写时写文件
- 支持“作者补充建议”，只提示可补充的真实信息类型，不伪造任何数据或文献

## 支持范围

- Stable: `.md`、`.txt`、`.docx`
- Experimental: `.doc`
- 输入必须是本地文件路径
- 输出为新文件，不覆盖原文件
- 默认输出文件命名为 `xxx.airc.md` 或 `xxx.airc.txt`
- 通过 `run` 处理 `.txt`、`.docx`、`.doc` 时，会先归一化为 Markdown，再进入现有 rewrite / review / write gate

## 不支持的输入

- `pdf`
- `html`
- 图片
- 表格型二进制文件

`.doc` 是实验性支持格式：AIRC 会尝试调用 pandoc，但如果转换失败，会明确提示 `doc support is experimental, please convert to docx first`。

## 设计边界

- 不伪造事实、年份、作者、文献、样本量、实验结果
- 不修改标题、数学公式、专业术语、数值、代码块、URL、引用编号
- 不将该工具用于规避检测或绕过任何审查/识别机制
- 只改“怎么说”，不改“说了什么”
- 当改写结果触发核心内容保护或未达到有效改写门槛时，默认不写出结果文件

## 安装方式

建议使用 Python 3.10+。

```bash
cd airc
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

如果你想把项目按包方式安装，也可以执行：

```bash
pip install -e .
```

### Pandoc 依赖

`.md` 和 `.txt` 不需要 pandoc。`.docx` 和 `.doc` 输入需要系统已安装 pandoc，并且 `pandoc` 可在 `PATH` 中被找到。

常见安装方式：

```bash
# Windows: 推荐使用 winget 或官方安装包
winget install JohnMacFarlane.Pandoc

# macOS
brew install pandoc

# Ubuntu / Debian
sudo apt-get install pandoc
```

如果 pandoc 缺失，AIRC 不会静默降级，会直接报错并建议安装 pandoc，或先把文件转换为 `.md` / `.docx`。

## CLI 使用示例

生产级一键接口：

```bash
airc run examples/sample.md --preset academic_natural
python -m airc_skill.cli run examples/sample.md --preset academic_natural
```

`run` 会执行 `guide -> controlled rewrite -> review -> write gate`，通过后写出 `xxx.airc.md` / `xxx.airc.txt`，并生成 `xxx.airc.report.json`。报告包含输入 schema、block execution plan、rewrite coverage、discourse / cluster score、changed/skipped blocks、warnings 与 write gate 结果。

处理 `.docx`：

```bash
airc run paper.docx --preset academic_natural
```

保留中间 Markdown：

```bash
airc run paper.docx --preset academic_natural --keep-normalized
```

失败案例示例：

```text
Input normalization failed for .docx: pandoc is missing. Install pandoc and ensure it is on PATH, or convert the file to .md/.docx first.
```

可用 preset：

- `academic_natural`：默认生产配置，适合正式学术文本，要求较高 coverage 与句群级重写
- `aggressive_rewrite`：更强 developmental rewrite，适合普通中文正文需要更大幅度重构时使用
- `conservative`：仅轻改，适合敏感或高风险文本

标准推荐流程：

```bash
python -m airc_skill.cli guide examples/sample.md --as-agent-context
python -m airc_skill.cli review original.md candidate.md
python -m airc_skill.cli write original.md candidate.md
```

便捷模式 `rewrite` 仍然保留，但它只是 `guide -> controlled rewrite -> review -> write gate` 的一键 convenience mode：

```bash
python -m airc_skill.cli rewrite examples/sample.md
```

指定改写强度：

```bash
python -m airc_skill.cli rewrite examples/sample.md --mode balanced
python -m airc_skill.cli rewrite examples/sample.txt --mode strong
```

只看 diff，不写文件：

```bash
python -m airc_skill.cli rewrite examples/sample.md --dry-run
```

生成“作者补充建议”：

```bash
python -m airc_skill.cli suggest examples/sample.md
python -m airc_skill.cli suggest examples/sample.txt
```

## Dry-run 示例

执行：

```bash
python -m airc_skill.cli rewrite examples/sample.txt --dry-run
```

将会输出统一 diff 预览。若没有可见变化，会提示当前模式下未生成差异，并且不会创建 `.airc.*` 文件。

默认情况下，只有满足以下条件时才会写文件：

- `decision = pass`
- 或 `decision = pass_with_minor_risk` 且 `effective_change = true`
- `rewrite_coverage` 达到写入门槛
- 核心内容与格式完整性检查通过

如果最终候选只是轻微词级替换、结构几乎未变，CLI 会明确提示：

- `No effective rewrite was produced.`
- `Output file was not written.`

## 最小测试示例

运行全部测试：

```bash
pytest
```

运行单个测试文件：

```bash
pytest tests/test_pipeline_md.py
```

## 模块说明

- `validator.py`：检查文件存在、扩展名、编码和大小上限
- `input_normalizer.py`：检测 `.md/.txt/.docx/.doc`，将公开 `run` 输入归一化为 Markdown
- `markdown_guard.py`：保护 Markdown 中的代码块、行内代码、链接、图片、公式等内容
- `chunker.py`：按段落或 Markdown 结构分块
- `models.py`：定义 `BlockPolicy`、`GuidanceReport`、`RewriteCandidate`、`ReviewReport`、`WriteGateDecision`
- `skill_protocol.py`：定义公网调用协议、execution plan、agent instructions 与 output schema
- `guidance.py`：扫描文档并输出 `block_policies`、`do_not_touch_blocks`、`high_risk_blocks`、`rewrite_actions_by_block` 等 guidance
- `rewriter.py`：离线规则改写器，并预留可替换后端接口
- `core_guard.py`：保护标题、公式、术语、引用与数值等核心项
- `reviewer.py`：校验核心内容、格式完整性、模板风险与自然度
- `suggester.py`：识别空泛段落并给出真实信息补充建议
- `pipeline.py`：维护 `guide -> block rewrite -> review -> write gate` 的 agent-first 流程
- `cli.py`：提供 `guide / review / write / rewrite / suggest` 命令

## 后续扩展方向

- 接入可选 LLM 后端，并继续保留离线回退路径
- 增强中英文混合文本的句法改写能力
- 增强 Markdown 列表与引用块的细粒度重写能力
- 增加更严格的术语保护与章节级审校策略

## License

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

## Release Notes

详见 [CHANGELOG.md](CHANGELOG.md)。当前 `0.2.0b0` 是 GitHub beta 试用版，适合外部用户安装、运行、提交 issue 与反馈样例；发布到 PyPI 前建议继续收集 `.docx` 转换边界样例和跨平台 pandoc 安装反馈。
