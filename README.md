# AIRC

**Artificial Intelligence Rewrite Content**: a structure-preserving academic rewrite skill for agents, local CLI workflows, and Python automation.

AIRC revises academic Markdown or plain-text drafts while protecting headings, formulas, citations, numbers, figure references, image Markdown, paths, checkpoints, technical terms, and document structure. It is designed for controlled academic revision, not free-form generation.

- Product name: `AIRC`
- CLI: `airc`
- Python import: `airc_skill`
- Skill ID: `airc-academic-revision`

[Public Usage](./PUBLIC_USAGE.md) · [中文说明](./README.zh-CN.md) · [中文使用说明](./PUBLIC_USAGE.zh-CN.md) · [Changelog](./CHANGELOG.md)

## What AIRC Does

- Guides agents through `guide -> run -> review -> write`.
- Rewrites editable body prose while keeping protected academic content stable.
- Uses chapter-aware rewrite policy, paragraph skeleton checks, sentence readability repair, semantic role integrity, enumeration integrity, evidence fidelity, and academic sentence naturalization.
- Supports target-style alignment with `--target-style-file`, using a reference document as a language-distribution target without copying its facts or wording.
- Emits a JSON report with rewrite coverage, target-style metrics, preservation checks, warnings, and write-gate decision.

## What AIRC Does Not Do

- It does not invent facts, citations, datasets, metrics, experiments, or external background.
- It does not rewrite formulas, code blocks, image Markdown, paths, checkpoints, tables, references, or protected technical identifiers.
- It does not treat a target-style file as evidence.
- It does not overwrite the source file unless the caller explicitly gives that output path.

## Features

- Structure-preserving academic rewrite for `.md` and `.txt`.
- Body-only rewrite metrics and long-document quota.
- Chapter-aware rewrite distribution.
- Paragraph topic-sentence and skeleton preservation.
- Semantic role and enumeration protection.
- Evidence fidelity and thesis tone restraint.
- Optional preset: `zh_academic_l2_mild`.
- Optional target-style alignment: `--target-style-file`.
- Public CLI, agent prompt workflow, and Python API.

## Installation

```bash
git clone https://github.com/zhangxiuwen040831/airc-skill.git
cd airc-skill
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

## Quickstart

```bash
airc --help
airc guide examples/sample.md --as-agent-context
airc run examples/sample.md --preset academic_natural
airc run examples/sample.md --preset zh_academic_l2_mild
airc run examples/sample.md --target-style-file examples/_user_excerpt.airc.md
```

## CLI Usage

Generate agent-facing guidance:

```bash
airc guide paper.md --as-agent-context
```

Run ordinary academic revision:

```bash
airc run paper.md --preset academic_natural
```

Run the mild non-native Chinese academic preset:

```bash
airc run paper.md --preset zh_academic_l2_mild
```

Preview without writing files:

```bash
airc run paper.md --preset academic_natural --dry-run
```

Review an external candidate:

```bash
airc review paper.md candidate.md
```

## Agent / Codex Usage

Copyable prompt:

```text
Use the airc-academic-revision skill to revise my Markdown/TXT academic document.
Run guide first, then run the revision pipeline, then review the report.
Preserve headings, formulas, citations, numbers, paths, figure references, image markdown, checkpoints, and technical terms.
If I provide a target-style file, use it only for language-distribution alignment, not as a source of facts.
Write the revised file next to the source file and provide the report path.
```

## Target-Style Alignment Usage

Use a real hand-edited reference when you want the output to match a user-specific language distribution:

```bash
airc run paper.md --target-style-file target-style.md
```

or combine it with a preset:

```bash
airc run paper.md --preset zh_academic_l2_mild --target-style-file target-style.md
```

Target-style alignment compares sentence length, function-word density, explanatory phrasing, compactness, native-fluency gap, L2 texture, and paragraph-class style match. It is distribution fitting, not copying. The original document remains the evidence source.

## Python API Usage

```python
from airc_skill import SkillInputSchema, guide_document, run_revision

schema = SkillInputSchema.from_path(
    "paper.md",
    preset="academic_natural",
    target_style_file="target-style.md",
    emit_json_report=True,
)

guidance, plan = guide_document(schema)
result = run_revision(schema)

print(result.output_schema.status)
print(result.output_schema.write_gate_decision)
print(result.output_schema.report_path)
```

## Inputs and Outputs

Stable inputs:

- `.md`
- `.txt`

Typical outputs:

- `paper.airc.md`
- `paper.airc.report.json`

The report includes:

- `body_rewrite_coverage`
- `style_distribution_match_ratio`
- `class_aware_style_match_ratio`
- `terminology_drift`
- `evidence_drift`
- `write_gate`

## Safety / Preservation Boundaries

What AIRC Preserves:

- titles, headings, and heading numbering
- formulas, variables, and formula numbering
- citations, citation order, years, numbers, thresholds, metrics, sample sizes, and model names
- figures, image Markdown, captions, tables, links, placeholders, paths, checkpoints, filenames, and version strings
- technical terms, dataset names, mechanism names, and Markdown structure

AIRC rejects candidates when protected content changes, evidence drift appears, target-style repair introduces unsupported claims, or chapter/body rewrite obligations are not met.

## Examples

Public examples are intentionally short and safe:

- `examples/sample.md`
- `examples/sample.txt`
- `examples/_user_excerpt.airc.md`

## Development and Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
airc run examples/sample.md --preset academic_natural --dry-run
airc run examples/sample.md --target-style-file examples/_user_excerpt.airc.md --dry-run
```

## Release Status

Current public release target: `v0.1.0-public-callable`.

The project is ready for public CLI, agent, and Python API use as a controlled academic revision skill.

## License

MIT License. See [LICENSE](./LICENSE).
