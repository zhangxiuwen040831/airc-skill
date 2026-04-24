"""AIRC package."""

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
from .natural_revision_profile import ACADEMIC_NATURAL_STUDENTLIKE, NaturalRevisionProfile
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
from .public_api import guide_document, review_candidate, run_revision, write_candidate
from .reviewer import review_rewrite
from .revision_doctrine import (
    ACADEMIC_NATURAL_REVISION_DOCTRINE,
    RevisionDoctrine,
    doctrine_for_agent_context,
)
from .skill_protocol import (
    AgentInstructionBundle,
    SkillExecutionPlan,
    SkillInputSchema,
    SkillOutputSchema,
    build_execution_plan,
    generate_agent_instructions,
    validate_execution_against_plan,
)

__all__ = [
    "RewriteMode",
    "InputNormalizationReport",
    "InputNormalizationError",
    "SkillInputSchema",
    "SkillExecutionPlan",
    "SkillOutputSchema",
    "AgentInstructionBundle",
    "NaturalRevisionProfile",
    "RevisionDoctrine",
    "ACADEMIC_NATURAL_STUDENTLIKE",
    "ACADEMIC_NATURAL_REVISION_DOCTRINE",
    "doctrine_for_agent_context",
    "guide_document_text",
    "build_execution_plan",
    "generate_agent_instructions",
    "validate_execution_against_plan",
    "run_revision",
    "guide_document",
    "review_candidate",
    "write_candidate",
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
