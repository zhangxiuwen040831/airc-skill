---
name: airc-academic-revision
description: Agent-first workflow for revising markdown or txt academic writing while preserving headings, formulas, terminology, citations, numbers, captions, placeholders, English spacing, paths, and Markdown structure. Use when Codex needs to scan a paper, classify blocks by rewrite policy, revise only safe narration blocks, review integrity, and pass a write gate before output.
---

# AIRC Academic Revision

Use AIRC as a guidance-and-review skill for agents, not as a blind global replacer.

## Standard Workflow

1. Read the document and call `guide` first.
2. Use block policy to decide what must stay verbatim, what is high risk, what only permits light editing, and what can be rewritten.
3. Apply the returned `rewrite_intensity` for each editable block: `light` for light-edit blocks, `medium` for ordinary body narration, and `high` for continuous background, significance, or analysis paragraphs.
4. Rewrite only `light_edit` and `rewritable` blocks, one block at a time.
5. Run `review` on the candidate document.
6. Write output only if the write gate passes.

The default mental model is:

- `guide_document_text(...)`
- `rewrite_block_with_guidance(...)` or `agent_rewrite_from_guidance(...)`
- `review_rewrite(...)`
- `decide_write_gate(...)`

`rewrite` CLI remains available as a convenience/fallback mode, but it is not the recommended primary workflow.

## Public Protocol

External agents should treat AIRC as a protocol, not as an implicit text filter.

Input schema:

- `text_path`: local markdown, text, docx, or experimental doc file.
- `mode`: `balanced`, `strong`, or `custom`.
- `target_style`: `academic_natural`, `concise`, or `explanatory`.
- `rewrite_scope`: `full`, `body_only`, or `selected_blocks`.
- `preservation_level`: `strict` or `standard`.

Execution plan fields returned from guidance:

- `block_policies`
- `rewrite_intensity_by_block`
- `required_actions_by_block`
- `forbidden_actions_by_block`
- `minimum_rewrite_coverage`

Output schema:

- `rewritten_file_path`
- `input_normalization`
- `rewrite_coverage`
- `discourse_change_score`
- `cluster_rewrite_score`
- `blocks_changed`
- `blocks_skipped`
- `warnings`

## How This Skill Works

AIRC is a structure-level academic revision protocol. It first freezes core and format-sensitive content, then classifies document blocks, then requires sentence-cluster and discourse-level rewriting only where the block policy permits it. The goal is to reorganize ordinary Chinese body narration while preserving paper content and Markdown structure.

The skill prioritizes expression restructuring over word substitution:

- Prefer sentence-cluster rewriting over synonym replacement.
- Prefer transition absorption over adding new template connectors.
- Prefer subject-chain compression over repeated `本研究 / 本文` openers.
- Prefer narrative-flow rebuilding for continuous background, significance, or analysis paragraphs.
- Keep original wording when the available rewrite would be stiff or would risk a protected item.

## Input Normalization

The public `run` entrypoint normalizes supported inputs before the rewrite pipeline:

| Input | Status | Behavior |
| --- | --- | --- |
| `.md` | stable | Use directly as Markdown. |
| `.txt` | stable | Read as plain text and wrap as Markdown. |
| `.docx` | stable | Convert to Markdown with pandoc. |
| `.doc` | experimental | Try pandoc; if it fails, instruct the user to convert to `.docx` first. |

Pandoc is only required for `.docx` and `.doc`. If pandoc is missing or conversion fails, do not continue silently. Surface the normalization error and include it in failure transparency when a report path is provided.

External agents should consume the normalized Markdown exactly as the input to `guide`, `rewrite`, `review`, and `write_gate`; they must not bypass the existing preservation and write-gate checks.

## Execution Enforcer

Agents using AIRC must follow these rules:

- Do not submit prefix-only, word-only, or connector-only rewrites for `rewritable` blocks.
- Do not skip a `rewritable` block unless guidance marks it as protected, high-risk, or unsafe to edit.
- Execute the required discourse actions listed for each block.
- Execute at least one cluster-level rewrite in developmental rewrite blocks.
- Meet the preset rewrite coverage threshold before asking the write gate to approve output.
- If coverage or discourse score is below threshold, retry with higher `rewrite_intensity` and force `sentence_cluster_rewrite` plus `narrative_flow_rebuilder`.
- Never change protected headings, formulas, terms, citations, numbers, paths, captions, placeholders, tables, code blocks, or Markdown symbols.
- If input normalization fails, stop and report the original type, converter, pandoc availability, and suggested next step.

## Presets

- `academic_natural`: default production preset. Uses balanced mode, medium/high body intensity, high coverage, and cluster rewrite.
- `aggressive_rewrite`: strong mode with high intensity and broad discourse transformation for ordinary body text.
- `conservative`: light-edit-only behavior for high-risk or very sensitive documents.

## What AIRC Is Responsible For

- Scanning the document and classifying blocks.
- Marking `do_not_touch`, `high_risk`, `light_edit`, and `rewritable` blocks.
- Returning block-level rewrite actions and forbidden actions.
- Returning rewrite depth, rewrite intensity, discourse obligations, and minimum rewrite coverage requirements.
- Protecting core content and format-sensitive content.
- Reviewing rewritten text for integrity, naturalness, template risk, discourse change, cluster rewriting, and rewrite coverage.
- Deciding whether the candidate is eligible to be written.

## What The Agent Is Responsible For

- Reading AIRC guidance before editing.
- Revising only the blocks that guidance allows.
- Choosing the most natural academic phrasing within AIRC’s boundaries.
- Keeping the original sentence when the available rewrite would sound stiff.
- Submitting the candidate back to AIRC for review and write-gate checks.

## Block Policy

- `do_not_touch`: Preserve verbatim. Only fix obvious formatting corruption.
- `high_risk`: Prefer the original. Allow only minimal cleanup when needed.
- `light_edit`: Reduce repeated subjects, compress meta discourse, soften mechanical connectors, and perform at least sentence-level change when the block is edited.
- `rewritable`: Perform developmental rewrite. Use sentence-cluster rewriting, discourse compression, transition absorption, conclusion absorption, enumeration reframing, or narrative-flow rebuilding.

## Developmental Rewrite Gate

- `rewrite_intensity=light`: use conservative sentence-level adjustment only.
- `rewrite_intensity=medium`: rewrite ordinary Chinese body narration with sentence-cluster changes.
- `rewrite_intensity=high`: rebuild continuous narration paragraphs by changing information order, fusing adjacent sentences, or absorbing conclusions into the surrounding argument.
- Full pass requires rewrite coverage of at least 60% over edited body sentences.
- `pass_with_minor_risk` may be allowed when coverage is at least 40%, the hard preserve layer passes, and remaining issues are minor.
- Reject prefix-only, word-only, or template-only rewrites even if the text is otherwise format-safe.

## Hard Boundaries

Never rewrite:

- Titles and heading numbering
- English titles and heading spacing
- Formulas and formula numbering
- Technical terms, model names, dataset names, algorithm names, system names
- Citations and citation order
- Numbers, years, thresholds, sample sizes, parameters, variable names
- Paths, checkpoints, version strings, file names
- Captions, placeholders, tables, code blocks, links
- Chapter conclusions or the direction of the argument

## Style Principles

- Only improve narration, sentence organization, paragraph flow, and naturalness.
- Keep the paper formal and restrained.
- Reduce template-heavy openers by deleting, absorbing, or merging them when possible.
- Prefer semantic carryover over explicit connectors when the logic is already clear.
- Compress repeated `本研究 / 本文` chains only when clarity is preserved.
- If removing `本研究` weakens reference clarity, keep it.

## Decision Principles

- 少改优于错改
- 清晰优于去重
- 格式优先级高于自然度
- 术语优先级高于风格

## Suggested CLI Sequence

1. `python -m airc_skill.cli guide <file> --as-agent-context`
2. Agent rewrites blocks according to the returned policy.
3. `python -m airc_skill.cli review <original> <candidate>`
4. `python -m airc_skill.cli write <original> <candidate>`

Public single-command interface:

- `airc run <file> --preset academic_natural`
- `python -m airc_skill.cli run <file> --preset academic_natural`

The `run` command writes the reviewed output only after the write gate passes and also emits a JSON rewrite report with the protocol input, execution plan, output schema, review metrics, write-gate decision, and failure transparency.

The JSON report includes `input_normalization` with:

- `original_path`
- `original_type`
- `normalized_type`
- `converter_used`
- `pandoc_used`
- `normalization_success`
- `normalization_error`

## References

- Read [references/agent-guided-workflow.md](references/agent-guided-workflow.md) for the block taxonomy, rewrite action mapping, and final self-checklist.
