# Changelog

## 0.2.0b0 - GitHub beta

- Added the public skill protocol layer with input schema, execution plan, agent instructions, and output schema.
- Added the production `airc run <file> --preset academic_natural` interface with JSON rewrite reports.
- Added preset support for `academic_natural`, `aggressive_rewrite`, and `conservative`.
- Added automatic retry escalation for low coverage or low discourse-change candidates.
- Added markdown normalization for `.md`, `.txt`, `.docx`, and experimental `.doc` inputs.
- Added pandoc-based `.docx` conversion and transparent failure messages for missing/failed pandoc conversions.
- Added strict report fields for input normalization, rewrite coverage, discourse score, cluster score, changed/skipped blocks, and write-gate outcomes.

## 0.1.0 - Local MVP

- Added agent-first guidance, block policies, review reports, and write gate.
- Added offline markdown/txt rewrite pipeline with hard preservation of headings, formulas, citations, terminology, numbers, paths, captions, placeholders, and Markdown structure.
- Added developmental rewrite metrics, coverage gate, discourse/cluster scores, and public JSON reporting.
