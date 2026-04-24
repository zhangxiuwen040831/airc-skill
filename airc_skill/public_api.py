from __future__ import annotations

from pathlib import Path

from .config import RewriteMode
from .models import GuidanceReport
from .pipeline import PublicRunResult, ReviewFileResult, WriteResult, guide_file, review_file, run_file, write_file
from .skill_protocol import SkillExecutionPlan, SkillInputSchema, build_execution_plan


def run_revision(schema: SkillInputSchema | str | Path) -> PublicRunResult:
    request = schema if isinstance(schema, SkillInputSchema) else SkillInputSchema.from_path(schema)
    return run_file(
        path=request.source_path,
        preset=request.preset,
        output_path=request.output_path,
        mode=request.mode,
        max_retry_passes=request.max_retry_passes,
        dry_run=False,
        debug_rewrite=False,
        keep_intermediate=False,
        emit_agent_context=request.emit_agent_context,
        emit_json_report=request.emit_json_report,
    )


def guide_document(schema: SkillInputSchema | str | Path) -> tuple[GuidanceReport, SkillExecutionPlan]:
    request = schema if isinstance(schema, SkillInputSchema) else SkillInputSchema.from_path(schema)
    from .io_utils import read_text_file
    from .validator import validate_input_file
    from .guidance import guide_document_text

    validated = validate_input_file(request.source_path)
    guidance = guide_document_text(
        read_text_file(validated),
        metadata={"suffix": validated.suffix, "source_path": validated.path, "target_style": request.target_style},
    )
    return guidance, build_execution_plan(guidance, request)


def review_candidate(
    original_path: str | Path,
    candidate_path: str | Path,
    schema: SkillInputSchema | None = None,
) -> ReviewFileResult:
    mode = schema.resolved_mode() if schema is not None else RewriteMode.BALANCED
    return review_file(original_path=original_path, candidate_path=candidate_path, mode=mode)


def write_candidate(
    original_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path | None = None,
    schema: SkillInputSchema | None = None,
    dry_run: bool = False,
) -> WriteResult:
    mode = schema.resolved_mode() if schema is not None else RewriteMode.BALANCED
    return write_file(
        original_path=original_path,
        candidate_path=candidate_path,
        output_path=output_path,
        mode=mode,
        dry_run=dry_run,
    )
