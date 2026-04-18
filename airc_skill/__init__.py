"""AIRC.skill package."""

from .config import RewriteMode
from .guidance import guide_document_text
from .input_normalizer import (
    InputNormalizationError,
    InputNormalizationReport,
    check_pandoc_available,
    convert_with_pandoc,
    detect_input_type,
    normalize_text_input,
    normalize_to_markdown,
)
from .pipeline import (
    agent_rewrite_from_guidance,
    decide_write_gate,
    guide_file,
    review_file,
    run_file,
    rewrite_block_with_guidance,
    rewrite_file,
    write_file,
)
from .reviewer import review_rewrite
from .skill_protocol import (
    SkillExecutionPlan,
    SkillInputSchema,
    SkillOutputSchema,
    build_execution_plan,
    generate_agent_instructions,
)

__all__ = [
    "RewriteMode",
    "InputNormalizationReport",
    "InputNormalizationError",
    "SkillInputSchema",
    "SkillExecutionPlan",
    "SkillOutputSchema",
    "guide_document_text",
    "build_execution_plan",
    "generate_agent_instructions",
    "rewrite_block_with_guidance",
    "agent_rewrite_from_guidance",
    "review_rewrite",
    "decide_write_gate",
    "guide_file",
    "review_file",
    "write_file",
    "rewrite_file",
    "run_file",
    "detect_input_type",
    "normalize_to_markdown",
    "normalize_text_input",
    "check_pandoc_available",
    "convert_with_pandoc",
    "__version__",
]

__version__ = "0.2.0b0"
