# AIRC Changelog

AIRC — Artificial Intelligence Rewrite Content.

## v0.1.0-public-callable - Public callable skill

- Published AIRC as a structure-preserving academic rewrite skill with public CLI, Agent/Codex, and Python API usage.
- Added chapter-aware rewrite, body-only coverage, long-document quota, and report-backed write gate.
- Added paragraph skeleton protection, topic-sentence preservation, and sentence readability repair.
- Added semantic role and enumeration integrity checks for innovation points, mechanism statements, and conclusion sentences.
- Added evidence fidelity and thesis tone restraint to block unsupported facts, commentary style, metaphors, and external claims.
- Added academic sentence naturalization and optional `zh_academic_l2_mild` preset.
- Added target-style alignment with `--target-style-file`, class-aware style metrics, and zero-drift guardrails.
- Cleaned public examples and release documentation for clone/install/use workflows.

## 0.2.0b0 - GitHub beta

- Added the public skill protocol layer with input schema, execution plan, agent instructions, and output schema.
- Added the production `airc run <file> --preset academic_natural` interface with JSON rewrite reports.
- Added preset support for `academic_natural`, `aggressive_rewrite`, `conservative`, `body_only`, and `zh_academic_l2_mild`.
- Added the optional `zh_academic_l2_mild` profile for mildly non-native Chinese academic prose that remains factual, grammatical, restrained, and protected-content safe.
- Added target-style alignment with paragraph-class breakdown, source-backed drift checks, and public `--target-style-file` CLI usage.
- Calibrated write-gate integrity checks to reduce false positives in numeric, terminology, caption, do-not-touch, paragraph skeleton, and mixed medium-priority chapter quota checks.
- Added automatic retry escalation for low coverage or low discourse-change candidates.
- Added markdown normalization for `.md`, `.txt`, `.docx`, and experimental `.doc` inputs.
- Added pandoc-based `.docx` conversion and transparent failure messages for missing/failed pandoc conversions.
- Added strict report fields for input normalization, rewrite coverage, discourse score, cluster score, changed/skipped blocks, and write-gate outcomes.
- Added AIRC doctrine metadata, agent-facing rewrite obligations, natural revision checklist fields, and public usage documentation for GitHub beta users.
- Cleaned public documentation for GitHub release, including CLI, agent, Python, preset, output, and protection-boundary guidance.

## 0.1.0 - Local MVP

- Added agent-first guidance, block policies, review reports, and write gate.
- Added offline markdown/txt rewrite pipeline with hard preservation of headings, formulas, citations, terminology, numbers, paths, captions, placeholders, and Markdown structure.
- Added developmental rewrite metrics, coverage gate, discourse/cluster scores, and public JSON reporting.
