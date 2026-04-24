# AGENTS.md

## Purpose

This file tells Codex how to maintain AIRC.

AIRC is a structure-preserving academic rewrite workflow and an agent-first academic revision system. It is not a generic text rewriter and not a place to pile on ad hoc rules. It is a behavior system made of guidance, rewriting, review, ranking, and a write gate. The main product promise is:

- protect headings, formulas, terms, citations, numbers, figures, paths, placeholders, and Markdown structure
- rebuild ordinary body prose naturally and academically
- verify coverage, chapter behavior, paragraph skeletons, and integrity before writing

Use this file when modifying AIRC itself. `SKILL.md` explains how to use AIRC as a skill; this file explains how Codex should change and maintain the AIRC codebase.

## Core Principles

### 1. Think Before Coding

Before editing code, identify:

- the real problem, not only the visible symptom
- the likely causes in guidance, rewrite, review, ranking, or write gate logic
- the smallest change path that can fix it
- the success criteria and which tests or real documents prove the fix

Do not write code first and justify it afterward. Do not make broad rewrites based on a single surface example. Inspect the actual input, actual output, report JSON, failed metrics, and relevant tests first.

### 2. Prefer Simplicity

Make the smallest effective change.

- If one existing module can solve the issue, do not introduce three new modules.
- If an existing data model can be extended cleanly, do not create a parallel framework.
- Do not add configuration flags for hypothetical future needs.
- Do not make rules more general than the problem requires.

AIRC already has many interacting layers. Extra abstraction is a cost unless it clearly reduces real complexity or prevents a known failure class.

### 3. Edit Precisely

Only change files that are directly connected to the current task.

- Do not opportunistically rename, reformat, or refactor unrelated code.
- Do not delete pre-existing behavior unless the task requires it.
- Do not weaken protection checks to make a rewrite pass.
- Every changed line should trace back to the current goal.

When a fix touches protection, body metrics, rewrite obligations, reviewer checks, ranking, or write gate logic, explain why that layer is the right place for the change.

### 4. Execute Toward Verifiable Goals

Translate every task into measurable outcomes, for example:

- fix body coverage inflation
- prevent image markdown from entering template checks
- reject long papers that only rewrite a few lines
- preserve paragraph topic sentences at paragraph openings
- keep technical-dense loss, metric, path, and checkpoint blocks conservative

Then verify the outcome with tests and the real `test.md` workflow. A passing test suite is necessary but not enough; inspect the generated text and report metrics when behavior quality matters.

## Project-Specific Rules for AIRC

### Priority Order

When rules conflict, use this order:

1. Core content and format protection must not break.
2. Body rewrite metrics must reflect real editable prose, not headings, images, formulas, captions, or blank structure.
3. Long documents must receive enough body-prose rewrite scale.
4. Paragraph skeletons and valid topic sentence openings must remain stable.
5. Human-like variation must improve readability, not create noise or disorder.

Never increase rewrite scale by weakening preservation or by rewriting high-risk technical content.

### Be Careful Around These Areas

Treat these as high-risk maintenance zones:

- `core_guard` and protected snapshot comparison
- validators and write-gate integrity checks
- formula, citation, number, path, checkpoint, heading, image, placeholder, caption, and Markdown preservation
- body-only denominator logic
- chapter quota enforcement
- paragraph skeleton and opening-style checks
- candidate ranking and write-gate reason codes

Changes here require targeted tests and a real `test.md` run.

### Known Historical Failure Modes

AIRC has previously failed in these ways. Check for regressions:

- rewrite coverage looked high because non-body content polluted the denominator
- headings, image markdown, captions, or blank lines affected body metrics
- long documents rewrote only a small local area but still passed
- candidate ranking favored local polish over global rewrite scale
- `![](...)` image markdown was treated as template repetition
- topic sentences moved away from paragraph openings
- paragraph openings became dangling transitions such as `并进一步`, `在这种情况下`, `相应地`, `围绕这一点`, `其中`, `此外`, or `由此`
- sentence-cluster rewrite broke the academic paragraph skeleton
- technical sections were rewritten too aggressively for the sake of coverage
- English title or abstract blocks entered natural revision even when they should remain frozen

### AIRC Behavior Boundaries

AIRC may restructure ordinary body prose, but it must not change:

- facts, claims, conclusions, or argument direction
- equations, variables, formula numbering, metrics, thresholds, paths, or filenames
- citations, citation order, years, numbers, model names, dataset names, or technical terms
- headings, image markdown, links, placeholders, tables, code, references, and Markdown structure

If a rewrite would sound more natural but risks any of the above, keep the source text or make a narrower edit.

## Validation Requirements

After each AIRC maintenance change, run:

```powershell
pytest -q
python -m airc_skill.cli guide C:\Users\32902\.codex\skills\AIRC\test.md --as-agent-context
python -m airc_skill.cli run C:\Users\32902\.codex\skills\AIRC\test.md --preset academic_natural --output C:\Users\32902\.codex\skills\AIRC\test_airc_v2.md --report C:\Users\32902\.codex\skills\AIRC\test_airc_v2.report.json
```

Run the same validation for documentation changes as well, because maintainer rules can change future engineering behavior. If a command cannot run in the local environment, keep the local edits, report the exact failure, and do not pretend the validation passed.

For runtime behavior changes, inspect and report:

- `body_rewrite_coverage`
- `body_changed_blocks / body_blocks_total`
- `body_changed_sentences / body_sentences_total`
- `document_scale`
- `rewrite_quota_met`
- chapter metrics: type, priority, coverage, changed blocks, discourse score, cluster score, quota status
- paragraph checks: topic sentence preserved, opening style valid, skeleton consistent, no dangling opening, topic sentence not demoted
- core content integrity and format integrity
- write-gate decision and reason codes

When text quality is the issue, compare `test.md` against the generated output. Do not rely only on aggregate metrics.

## Output Requirements

At the end of a maintenance task, report:

- files changed
- the behavior or documentation contract that changed
- tests and real-document validation performed
- important metrics or failure reasons
- commit and push status, if git is available

If core behavior logic changed, include the key function code in the final answer, not only a summary. Include the relevant function bodies or the smallest meaningful excerpts for:

- guidance functions that classify blocks, chapters, body prose, or agent obligations
- rewriter functions that transform sentences, clusters, paragraph openings, or protected text
- reviewer functions that judge naturalness, body metrics, chapter quotas, paragraph skeletons, or integrity
- pipeline functions that rank candidates, decide the write gate, or serialize reports

Do not paste large unrelated files. Show the code that proves the behavior changed.

## What Not To Do

- Do not calculate coverage only over blocks that were already changed.
- Do not include headings, images, image markdown, captions, formulas, references, blank lines, paths, or structural markdown in body rewrite coverage.
- Do not let image markdown or figure lines participate in template repetition checks.
- Do not improve coverage by rewriting technical-dense, formula-dense, number-dense, path-heavy, or citation-heavy content.
- Do not move a valid paragraph topic sentence into the middle of a paragraph.
- Do not start paragraphs with dangling transition fragments unless the sentence independently states the paragraph topic.
- Do not use synonym swaps, prefix changes, or connector replacement as a substitute for body-prose reconstruction.
- Do not make human variation into randomness, noise, or broken logic.
- Do not treat a global pass as sufficient when a high-priority chapter remained barely changed.
- Do not treat passing tests as proof that the revised paper reads well.
- Do not give only a narrative report when core behavior code changed; include key code snippets.

## Relationship To Other Docs

- `SKILL.md`: user-facing and agent-facing instructions for invoking AIRC on academic documents.
- `references/agent-guided-workflow.md`: operational rewrite workflow for agents using AIRC.
- `AGENTS.md`: maintainer-facing rules for Codex while changing AIRC's code, tests, reports, and documentation.
