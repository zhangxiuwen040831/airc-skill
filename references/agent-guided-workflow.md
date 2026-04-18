# Agent-Guided Workflow

Recommended sequence:

1. `guide_document_text(...)`
2. `rewrite_block_with_guidance(...)` or `agent_rewrite_from_guidance(...)`
3. `review_rewrite(...)`
4. `decide_write_gate(...)`

`rewrite_file(...)` is only a convenience wrapper around the same policy chain.

## 1. Scan First

Classify each block before editing:

- `do_not_touch`
  Use for headings, English title blocks, English abstract blocks, formulas, captions, placeholders, tables, code blocks, links, and other structural Markdown blocks.
- `high_risk`
  Use for deployment notes, checkpoint paths, number-dense paragraphs, citation-dense paragraphs, or mixed technical blocks where wording drift is risky.
- `light_edit`
  Use for method descriptions, system implementation paragraphs, definition-like paragraphs, and conclusion summaries.
- `rewritable`
  Use for background, significance, risk discussion, and general analytical narration.

## 2. Rewrite Only What Merits Rewrite

### Preferred actions

- `reduce_repeated_subjects`
  Merge or absorb adjacent sentences before inventing a new opener.
- `compress_meta_discourse`
  Fold theme sentences into work-description sentences when that improves flow.
- `merge_or_split_sentence_cluster`
  Split overly dense long sentences or merge brittle short follow-ups when the logic stays intact.
- `soften_template_connectors`
  Delete or absorb `同时 / 因此 / 此外 / 由此` when the transition is already obvious.
- `keep_explicit_subject_if_needed`
  Preserve `本研究` when a checkpoint, path, version, or deployment statement needs a clear owner.
- `keep_original_if_rewrite_would_be_stiff`
  Stop rewriting when every available change sounds more mechanical than the source.

### Never use

- Colloquial fillers such as `这块`, `这边`, `大家`, `我们`, `里头`
- Blog-like emphasis, emotional phrasing, or self-media rhythm
- Generic replacements that blur terms, equations, metrics, or deployment details

## 3. Frozen Content

Freeze and restore these blocks around any rewrite pass:

- headings
- english title blocks
- english abstract blocks
- formula blocks
- citation blocks
- technical term blocks
- numeric blocks
- placeholder blocks
- caption blocks
- table blocks
- code blocks
- path and checkpoint blocks

## 4. Post-Rewrite Checklist

### Core consistency

- Meaning unchanged
- Conclusions unchanged
- Terms unchanged
- Numbers and citations unchanged

### Format consistency

- Heading hierarchy unchanged
- English spacing unchanged
- No `..` inserted into English abstract blocks
- No `：。` introduced into captions
- Placeholders and links unchanged
- No new trailing spaces or broken line breaks

### Expression quality

- No obvious repeated `本研究 / 本文` chain unless clarity requires it
- No mechanical template stacking
- No colloquial drift
- No low-value rewrite performed just to force visible change
