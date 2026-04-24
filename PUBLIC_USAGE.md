# AIRC Public Usage

AIRC means **Artificial Intelligence Rewrite Content**. It is a structure-preserving academic rewrite skill for Markdown and plain-text papers.

Identifiers:

- Product: `AIRC`
- CLI: `airc`
- Skill ID: `airc-academic-revision`
- Python package/import: `airc-skill` / `airc_skill`

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

## Supported Inputs

AIRC is stable for:

- `.md`
- `.txt`

It does not overwrite the source file by default. `airc run paper.md` writes a revised file next to the source, normally as `paper.airc.md`, plus a JSON report.

## CLI Usage

Create agent context:

```bash
airc guide paper.md --as-agent-context
```

Run the default preset:

```bash
airc run paper.md --preset academic_natural
```

Run the mild non-native Chinese academic preset:

```bash
airc run paper.md --preset zh_academic_l2_mild
```

Run target-style alignment:

```bash
airc run paper.md --target-style-file target-style.md
```

Preview without writing:

```bash
airc run paper.md --preset academic_natural --dry-run
```

Review an external candidate:

```bash
airc review paper.md candidate.md
```

## Agent Usage

```text
Use the airc-academic-revision skill to revise my Markdown/TXT academic document.
Run guide first, then run the revision pipeline, then review the report.
Preserve headings, formulas, citations, numbers, paths, figure references, image markdown, checkpoints, and technical terms.
If I provide a target-style file, use it only for language-distribution alignment, not as a source of facts.
Write the revised file next to the source file and provide the report path.
```

## Presets

### `academic_natural`

Default preset. It revises ordinary body prose into a natural academic style while preserving facts, terms, formulas, citations, numbers, paths, figures, and Markdown structure.

### `zh_academic_l2_mild`

Optional preset. It keeps the text academic, readable, and correct, while making the Chinese slightly less native-like and somewhat more explanatory.

Allowed:

- mild function-word increase
- slightly lower sentence compression
- small harmless repetition
- clearer explanatory phrasing

Not allowed:

- grammar errors
- web or colloquial language
- unsupported external facts
- technical term, formula, citation, number, path, figure, or Markdown drift

## Target-Style File

`--target-style-file` lets AIRC compare the output against a reference document's language distribution.

It checks:

- sentence length distribution
- function-word density
- helper verb usage
- explanatory phrasing
- compactness gap
- native-fluency gap
- paragraph-class style match

The target-style file is **not** a source of facts. AIRC must not copy target sentences, import target claims, or change source evidence to match the reference.

## Output Files

Default output:

- `paper.airc.md`
- `paper.airc.report.json`

You can also choose explicit paths:

```bash
airc run paper.md --output revised.md --report revised.report.json
```

## Reading the Report

Useful fields:

- `write_gate.decision`: must be `pass` or `pass_with_minor_risk` before trusting the output.
- `review.body_rewrite_coverage`: how much editable body prose changed.
- `review.target_style_alignment.style_distribution_match_ratio`: whole-document style match, usually good around `0.70+`.
- `review.target_style_alignment.class_aware_style_match_ratio`: paragraph-class style match, useful for long papers.
- `review.target_style_alignment.terminology_drift`: should be `0`.
- `review.target_style_alignment.evidence_drift`: should be `0`.

## Common Failure Reasons

- Protected content changed: formulas, numbers, citations, paths, headings, captions, image Markdown, or technical terms moved outside safe bounds.
- Evidence drift: the candidate adds unsupported facts, statistics, claims, or examples.
- Target style mismatch: the output remains too far from the reference style distribution.
- Chapter quota not met: high-priority body chapters were not rewritten enough.
- Paragraph skeleton issue: topic sentence or paragraph flow was damaged.

## How To Rerun Safely

1. Fix obvious input issues first: broken Markdown, accidental pasted images, malformed headings, or mixed private notes.
2. Run `airc guide paper.md --as-agent-context`.
3. Run `airc run paper.md --dry-run`.
4. Inspect the report warnings.
5. Run without `--dry-run` only when the write gate is acceptable.

## Notes and Limitations

- AIRC is a controlled academic revision workflow, not a fact generator.
- Technical-dense sections are intentionally conservative.
- Target-style alignment learns language distribution only.
- AIRC does not intentionally generate grammar errors, even under `zh_academic_l2_mild`.
