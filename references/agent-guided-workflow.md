# Agent-Guided Workflow

AIRC is an agent-first structure-preserving academic rewrite workflow. Treat it as a policy, guidance, and review system: the agent performs the rewrite block by block, while AIRC defines what can be changed, what must be preserved, and whether the final candidate may be written.

For Codex maintenance rules that govern changes to the AIRC codebase itself, see [../AGENTS.md](../AGENTS.md).

Recommended sequence:

1. `guide_document_text(...)` or `airc guide <file> --as-agent-context`
2. Read `block_policies`, `rewrite_depth`, required actions, forbidden actions, and preserve items.
3. Rewrite only `light_edit` and `rewritable` blocks.
4. Submit the candidate to `review_rewrite(...)` or `airc review <original> <candidate>`.
5. Use `decide_write_gate(...)` or `airc write <original> <candidate>` before writing.

`airc run <file> --preset academic_natural` is the public convenience path. It still follows the same `guide -> controlled rewrite -> review -> write gate` chain.

## 1. Scan First

Classify every block before editing:

- `do_not_touch`
  Preserve headings, English title blocks, English abstracts, formulas, figure/table captions, placeholders, tables, code blocks, links, and Markdown structure verbatim.
- `high_risk`
  Use for checkpoint paths, version strings, number-dense paragraphs, citation-dense paragraphs, term-dense deployment notes, and mixed technical blocks. Prefer the original and only clean obvious formatting issues.
- `light_edit`
  Use for method descriptions, system implementation paragraphs, definition-like paragraphs, and concise conclusion summaries. Apply sentence-level polish, not paragraph-level reshaping.
- `rewritable`
  Use for background, research significance, risk discussion, general analysis, and ordinary Chinese method-explanation narration. Apply developmental academic rewrite.

## 2. Chapter-Aware Rewrite Policy

Do not treat a long paper as a flat list of paragraphs. Each body block carries chapter fields in the guide output:

- `chapter_type`
- `chapter_rewrite_priority`
- `chapter_rewrite_quota`
- `chapter_rewrite_intensity`

Use the nearest heading first. If a heading is unavailable, infer function from paragraph content.

High-priority chapters need stronger reconstruction:

- `background`, `significance`, `literature_review`, `challenge_analysis`
- `result_analysis`, `error_analysis`, `conclusion`, `future_work`

For these, change sentence clusters, explanation path, local discourse order, and conclusion absorption. A few synonym changes or a single polished sentence is not enough.

Medium-priority chapters allow controlled developmental rewrite:

- `method_design`, `mechanism_explanation`, `training_strategy`
- `dataset_description`, `system_architecture`, `system_workflow`

For these, improve flow around the technical content while keeping terms, numbers, paths, citations, and stated mechanisms stable.

Conservative chapters should remain light:

- `problem_definition`, `loss_function`, `evaluation_metrics`, `experiment_setup`
- deployment details with paths, versions, thresholds, filenames, checkpoints, or parameters
- formula-, number-, citation-, or path-dense paragraphs

For these, prefer partial keep, subject-chain compression, connector cleanup, and minor sentence rhythm changes. Avoid broad sentence-cluster reconstruction.

The reviewer enforces chapter quotas. High-priority chapters must show chapter-level coverage, cluster-level rewrite, and discourse-level change; medium chapters must show visible sentence or cluster change; conservative chapters are rejected or warned when they are over-rewritten.

## 3. When To Use Developmental Rewrite

Use developmental rewrite when a block is ordinary Chinese body narration and not dense with protected terms, numbers, citations, paths, formulas, or figure/table references.

Required behavior for `rewritable` blocks:

- Compress repeated `本研究 / 本文 / 该系统` subject chains.
- Compress meta discourse such as `本研究的主题为` when the next sentence already describes the work.
- Absorb mechanical connectors such as `因此 / 同时 / 此外` into the surrounding sentence when the logic is already clear.
- Rebuild sentence rhythm by combining brittle short sentences or splitting overloaded long sentences.
- Restructure adjacent two- to five-sentence clusters rather than replacing only the first phrase.
- Apply one or two revision patterns per body block: `compress`, `expand`, `merge`, `split`, `reorder`, `soften`, `reframe`, `partial_keep`, or `rewrite_all`.
- Keep the pattern distribution uneven across the document; do not edit every block with the same intensity.
- Break overly tidy parallel sentence skeletons.
- Preserve technical terms, citations, numbers, paths, captions, placeholders, and Markdown symbols exactly.

Developmental rewrite should change narration structure while keeping the paper’s facts, claims, and argument direction unchanged.

## 4. When To Use Light Edit

Use light edit when the block is editable but close to high risk.

Allowed actions:

- `reduce_function_word_overuse`
- `weaken_template_connectors`
- `compress_meta_discourse`
- `compress_subject_chain`
- `rebuild_sentence_rhythm`
- `rewrite_dense_nominal_phrases`

Do not force large changes. If the safest natural version is close to the source, keep it close.

## 5. When To Keep The Original

Keep the original block when:

- The block is a heading, caption, formula, table, code block, placeholder, or link.
- The block contains checkpoint paths, version strings, thresholds, variables, or file names whose wording must stay exact.
- The block is dense with citations, numbers, or named technical concepts.
- The available rewrite would sound like another template.
- Removing `本研究` or another explicit subject would weaken clarity.

Principles:

- 少改优于错改
- 清晰优于去重
- 格式优先级高于自然度
- 术语优先级高于风格

## 6. Natural Revision Actions

Agents should apply these as executable actions, not vague style advice:

- `reduce_function_word_overuse`
  Control unnecessary `了`, stacked `的`, and repeated `在……中` structures only when precision remains stable.
- `weaken_template_connectors`
  Delete, absorb, or merge mechanical connectors instead of swapping them for another template.
- `compress_meta_discourse`
  Fold theme statements into work-description sentences when the next sentence already carries the real content.
- `compress_subject_chain`
  Merge or recast adjacent `本研究 / 本文 / 该系统` openings while preserving clear reference.
- `rebuild_sentence_rhythm`
  Use natural long-short variation without arbitrary splitting.
- `break_parallelism`
  Recast one sentence or merge adjacent sentences when the paragraph feels too evenly patterned.
- `rewrite_dense_nominal_phrases`
  Simplify heavy nominal structures without changing terminology.
- `preserve_explicit_subject_if_clarity_needed`
  Keep `本研究` when weakening the subject would make the sentence vague.
- `keep_original_if_technical_density_is_high`
  Do not expand or reframe term-dense technical content.
- `sentence_cluster_merge`
  Merge adjacent sentences when a human editor would absorb setup, transition, or conclusion into the same movement.
- `sentence_cluster_split`
  Split an overloaded cluster only when readability improves and protected content remains stable.
- `discourse_reordering`
  Reorder local sentence groups to clarify cause-effect, background-result, or claim-support flow.
- `narrative_path_rewrite`
  Change the explanation route of a block rather than swapping synonyms inside the same structure.
- `conclusion_absorption`
  Absorb formulaic conclusion follow-ups into the claim they support.
- `uneven_rewrite_distribution`
  Leave some precise sentences close to the source while rewriting other sentences or blocks more heavily.

Avoid colloquial or web-style wording such as `这块`, `这边`, `大家`, `我们`, `里头`, `超`, `超级`, `真的`, `其实`, or `大白话`.

## 7. Post-Rewrite Checklist

Core consistency:

- Meaning unchanged
- Facts unchanged
- Conclusions and argument direction unchanged
- Terms, numbers, citations, paths, and checkpoints unchanged

Format consistency:

- Heading hierarchy and heading text unchanged
- English spacing unchanged
- No `..` inserted into English abstracts
- No `：。` introduced into captions
- Placeholders and links unchanged
- No trailing spaces or broken Markdown line structure

Natural revision quality:

- `本研究 / 本文` chains are reduced where clarity allows
- Mechanical connectors are reduced or absorbed
- Function-word overuse is controlled
- Slogan-like parallelism is reduced
- Sentence rhythm has natural variation
- Meta discourse is compressed
- Narrative flow is rebuilt in rewritable body blocks
- Sentence-cluster changes are present in long rewrites
- Revision patterns are not mechanically uniform across blocks
- Human-like variation is present without colloquializing the text
- Chapter quotas match chapter function: narrative sections are opened up, technical sections stay controlled
- Tone remains formal, restrained, and academic
