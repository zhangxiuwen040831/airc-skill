---
name: airc-academic-revision
description: AIRC workflow for revising markdown or txt academic writing while preserving headings, formulas, terminology, citations, numbers, captions, placeholders, English spacing, paths, and Markdown structure. Use when Codex needs to scan a paper, classify blocks by rewrite policy, revise safe narration blocks, review integrity, and pass a write gate before output.
---

# AIRC

**Artificial Intelligence Rewrite Content**

AIRC is a structure-preserving academic rewrite skill. It helps an agent revise academic Markdown or TXT documents while keeping protected academic content stable.

- Product name: `AIRC`
- CLI: `airc`
- Skill ID: `airc-academic-revision`
- Python import: `airc_skill`

## What This Skill Does

AIRC scans a document, assigns block policies, performs controlled developmental rewrite on editable body prose, reviews preservation and style metrics, and writes output only when the write gate allows it.

## When To Use

Use AIRC when the user asks to revise an academic Markdown or TXT document and needs:

- protected headings, formulas, citations, numbers, paths, captions, image Markdown, checkpoints, and technical terms
- academic body-prose rewriting rather than simple synonym replacement
- report-backed review before writing
- optional style preset control
- optional target-style alignment against a reference document

## When Not To Use

Do not use AIRC to:

- generate new facts, citations, datasets, experiments, metrics, or external background
- change formulas, captions, tables, references, code blocks, paths, checkpoints, image Markdown, or protected technical terms
- rewrite non-academic text freely
- imitate a target-style file as content or copy target wording
- bypass review or write gate checks

## Workflow

1. `guide`
2. `run`
3. `review report`
4. `write output`

Command flow:

```bash
airc guide paper.md --as-agent-context
airc run paper.md --preset academic_natural
```

Target-style flow:

```bash
airc run paper.md --target-style-file target-style.md
```

The target file is a style reference only. It is not an evidence source. Never copy target text directly.

## Block Policies

AIRC classifies blocks before rewriting:

- `do_not_touch`: preserve verbatim or compare only protected spans.
- `high_risk`: keep original unless a very narrow cleanup is safe.
- `light_edit`: small sentence-level cleanup.
- `rewritable`: body prose eligible for deeper sentence-cluster and discourse-level revision.

Agent instructions may include `required_sentence_actions`, `required_cluster_actions`, `clarity_constraints`, `preserve_items`, and forbidden patterns.

## Preservation Rules

Never change:

- titles, headings, and heading numbering
- formulas, variables, and formula numbering
- citations, citation order, years, numeric values, thresholds, metrics, sample sizes, and model names
- technical terms, dataset names, algorithm names, mechanism names, and system names
- captions, figures, image Markdown, links, placeholders, tables, references, and code blocks
- paths, checkpoints, filenames, and version strings
- Markdown structure and symbols

If a rewrite risks any protected content, keep the source block and report the reason.

## Target-Style Alignment Rule

When `--target-style-file` is provided:

- use the target file only to fit language-distribution signals
- preserve the source document's facts, terms, numbers, formulas, citations, paths, figures, and Markdown
- never import target claims or external evidence
- never copy target sentences directly
- Do not copy target wording or target sentence structure as content.
- reject or warn if `terminology_drift`, `evidence_drift`, grammar risk, or protected-content drift appears

Target-style metrics include:

- `style_distribution_match_ratio`
- `class_aware_style_match_ratio`
- `terminology_drift`
- `evidence_drift`
- paragraph-class breakdown

## Output Contract

AIRC writes:

- revised `.airc.md`, or the requested output path
- `.report.json`, or the requested report path

The final response should summarize:

- output path
- report path
- `write_gate.decision`
- core warnings
- `body_rewrite_coverage`
- target-style metrics when enabled
- any blocker if the write gate rejects

The candidate should be treated as usable only when the write gate decision is `pass` or `pass_with_minor_risk`.

## Presets

- `academic_natural`: default controlled academic revision.
- `zh_academic_l2_mild`: mildly non-native Chinese academic prose. It may be slightly more explanatory and less compressed, but must remain grammatical, academic, factual, and protected-content safe.

Other local presets may exist for specialized workflows, but public use should start with `academic_natural`.

## Agent Prompt Template

```text
Use the airc-academic-revision skill to revise my Markdown/TXT academic document.
Run guide first, then run the revision pipeline, then review the report.
Preserve headings, formulas, citations, numbers, paths, figure references, image markdown, checkpoints, and technical terms.
If I provide a target-style file, use it only for language-distribution alignment, not as a source of facts.
Write the revised file next to the source file and provide the report path.
```
