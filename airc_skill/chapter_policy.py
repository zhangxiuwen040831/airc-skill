from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .config import RewriteMode
from .rewriter import RewriteStats, split_sentences

CHAPTER_TYPES = (
    "background",
    "significance",
    "literature_review",
    "challenge_analysis",
    "method_design",
    "problem_definition",
    "model_architecture",
    "mechanism_explanation",
    "training_strategy",
    "loss_function",
    "dataset_description",
    "experiment_setup",
    "evaluation_metrics",
    "result_analysis",
    "error_analysis",
    "system_architecture",
    "system_workflow",
    "deployment_description",
    "conclusion",
    "future_work",
)

HIGH_PRIORITY_TYPES = {
    "background",
    "significance",
    "literature_review",
    "challenge_analysis",
    "result_analysis",
    "error_analysis",
    "conclusion",
    "future_work",
}
MEDIUM_PRIORITY_TYPES = {
    "method_design",
    "mechanism_explanation",
    "training_strategy",
    "dataset_description",
    "system_workflow",
    "system_architecture",
    "model_architecture",
}
CONSERVATIVE_PRIORITY_TYPES = {
    "problem_definition",
    "loss_function",
    "evaluation_metrics",
    "experiment_setup",
    "deployment_description",
}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")
_CITATION_RE = re.compile(r"\[\d+\](?:\[\d+\])*|\([A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?\)|（[A-Z][A-Za-z\-]+,\s*\d{4}[a-z]?）")
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\[^\s<>()]+|(?:[\w.-]+/)+[\w./-]+|checkpoint[s]?/[^\s<>()]+|[\w.-]+\.(?:pth|pt|ckpt|png|jpg|jpeg|pdf))")
_FORMULA_RE = re.compile(r"\$\$|\\\(|\\\[|\\tag\{|\\begin\{|\\end\{")
_TECH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-/]{2,}|checkpoint[s]?/[^\s<>()]+|[\w.-]+\.(?:pth|pt|ckpt)")
_NORMALIZE_RE = re.compile(r"\s+")

_TITLE_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("future_work", ("展望", "未来工作", "后续工作", "future")),
    ("conclusion", ("总结", "结论", "本章小结", "小结", "conclusion")),
    ("error_analysis", ("错误分析", "误差分析", "失败分析", "消融", "边界样本", "error")),
    ("result_analysis", ("结果分析", "实验结果", "关键结果", "结果", "result")),
    ("evaluation_metrics", ("评价指标", "评估指标", "指标", "metrics")),
    ("experiment_setup", ("实验设置", "实验环境", "参数设置", "实验配置", "setup")),
    ("dataset_description", ("数据集", "数据说明", "数据来源", "dataset")),
    ("deployment_description", ("部署", "运行环境", "服务实现", "推理服务", "deployment")),
    ("system_workflow", ("系统流程", "工作流", "页面", "交互", "流程", "workflow")),
    ("system_architecture", ("系统总体架构", "系统架构", "系统实现", "功能设计", "前端", "后端")),
    ("loss_function", ("损失函数", "损失", "loss")),
    ("training_strategy", ("训练策略", "课程学习", "训练", "优化策略", "training")),
    ("mechanism_explanation", ("判别机制", "机制", "原理", "模块", "可解释性")),
    ("model_architecture", ("模型架构", "网络结构", "模型结构", "架构")),
    ("problem_definition", ("问题形式化", "问题定义", "任务定义", "定义", "符号")),
    ("method_design", ("方法设计", "研究方法", "方案设计", "方法", "method")),
    ("literature_review", ("国内外研究现状", "相关研究", "文献综述", "研究现状", "综述", "review")),
    ("challenge_analysis", ("挑战", "风险", "问题分析", "捷径问题", "困难", "分布偏移")),
    ("significance", ("研究意义", "意义", "价值")),
    ("background", ("研究背景", "绪论", "引言", "背景", "introduction")),
]

_CONTENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("future_work", ("未来工作", "后续工作", "展望", "未来可")),
    ("conclusion", ("总结而言", "综上", "本研究的核心价值", "本文完成")),
    ("error_analysis", ("误报", "错误", "失败", "边界样本", "漂移")),
    ("result_analysis", ("实验结果", "结果表明", "结果显示", "分析可以看出")),
    ("evaluation_metrics", ("accuracy", "precision", "recall", "f1", "auc", "评价指标", "召回率", "精度")),
    ("experiment_setup", ("实验设置", "batch", "epoch", "learning rate", "参数", "阈值")),
    ("dataset_description", ("数据集", "样本", "训练集", "验证集", "测试集")),
    ("deployment_description", ("部署", "flask", "接口", "版本", "checkpoint", "运行环境")),
    ("system_workflow", ("页面", "交互", "上传", "检测流程", "展示")),
    ("system_architecture", ("前端", "后端", "架构", "模块", "系统")),
    ("loss_function", ("损失", "loss", "公式", "目标函数")),
    ("training_strategy", ("训练", "课程学习", "优化", "缓冲区")),
    ("mechanism_explanation", ("机制", "判别", "决策路径", "分支", "融合")),
    ("model_architecture", ("模型", "架构", "网络", "语义分支", "频域分支")),
    ("problem_definition", ("定义为", "建模为", "表示为", "给定", "输出概率")),
    ("method_design", ("方法", "设计", "方案", "框架")),
    ("literature_review", ("现有研究", "已有方法", "研究中", "文献", "相关工作")),
    ("challenge_analysis", ("挑战", "风险", "问题", "不足", "困难")),
    ("significance", ("意义", "价值", "重要", "有助于")),
    ("background", ("近年来", "随着", "背景", "现实场景")),
]


@dataclass(frozen=True)
class ChapterQuota:
    coverage_min: float
    changed_block_ratio_min: float
    discourse_min: int
    cluster_min: int
    max_coverage: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "coverage_min": self.coverage_min,
            "changed_block_ratio_min": self.changed_block_ratio_min,
            "discourse_min": self.discourse_min,
            "cluster_min": self.cluster_min,
        }
        if self.max_coverage is not None:
            payload["max_coverage"] = self.max_coverage
        return payload


def clean_heading_title(text: str) -> str:
    stripped = " ".join(line.strip() for line in text.splitlines() if line.strip())
    stripped = re.sub(r"^\s{0,3}#{1,6}\s*", "", stripped)
    return stripped.strip()


def classify_chapter_type(title: str, text: str = "") -> str:
    title_norm = _normalize(title).lower()
    for chapter_type, markers in _TITLE_PATTERNS:
        if any(marker.lower() in title_norm for marker in markers):
            return chapter_type

    text_norm = _normalize(text).lower()
    for chapter_type, markers in _CONTENT_PATTERNS:
        if any(marker.lower() in text_norm for marker in markers):
            return chapter_type
    return "background" if not title_norm else "method_design"


def technical_density_level(text: str) -> str:
    citation_count = len(_CITATION_RE.findall(text))
    number_count = len(_NUMBER_RE.findall(text))
    path_count = len(_PATH_RE.findall(text))
    formula_count = len(_FORMULA_RE.findall(text))
    tech_count = len(_TECH_TOKEN_RE.findall(text))
    sentence_count = max(1, len(split_sentences(text)))

    if path_count or formula_count or (citation_count >= 2 and number_count >= 2):
        return "high"
    if number_count >= 4 or tech_count >= 6 or citation_count >= 3:
        return "high"
    if number_count >= 2 or tech_count >= 3 or citation_count >= 1:
        return "medium"
    if sentence_count <= 1 and (number_count or tech_count):
        return "medium"
    return "low"


def priority_for_chapter(chapter_type: str, text: str = "") -> str:
    technical_density = technical_density_level(text)
    if chapter_type == "error_analysis" and technical_density == "high":
        return "medium"
    if chapter_type == "deployment_description" and technical_density in {"medium", "high"}:
        return "conservative"
    if chapter_type in CONSERVATIVE_PRIORITY_TYPES:
        return "conservative"
    if chapter_type in HIGH_PRIORITY_TYPES:
        return "high"
    if technical_density == "high":
        return "conservative"
    if chapter_type in MEDIUM_PRIORITY_TYPES:
        return "medium"
    return "medium"


def quota_for_priority(priority: str) -> ChapterQuota:
    if priority == "high":
        return ChapterQuota(
            coverage_min=0.60,
            changed_block_ratio_min=0.50,
            discourse_min=2,
            cluster_min=1,
        )
    if priority == "medium":
        return ChapterQuota(
            coverage_min=0.40,
            changed_block_ratio_min=0.25,
            discourse_min=1,
            cluster_min=0,
        )
    return ChapterQuota(
        coverage_min=0.0,
        changed_block_ratio_min=0.0,
        discourse_min=0,
        cluster_min=0,
        max_coverage=0.75,
    )


def target_intensity_for_priority(priority: str) -> str:
    return {"high": "high", "medium": "medium", "conservative": "light"}.get(priority, "medium")


def chapter_summary_from_blocks(block_policies: list[Any]) -> list[dict[str, object]]:
    seen: set[tuple[int, str]] = set()
    summaries: list[dict[str, object]] = []
    for block in block_policies:
        key = (int(getattr(block, "chapter_id", 0)), str(getattr(block, "chapter_title", "")))
        if key in seen:
            continue
        seen.add(key)
        summaries.append(
            {
                "chapter_id": key[0],
                "chapter_title": key[1] or "Inferred section",
                "chapter_type": getattr(block, "chapter_type", "unknown"),
                "chapter_rewrite_priority": getattr(block, "chapter_rewrite_priority", "medium"),
                "chapter_rewrite_intensity": getattr(block, "chapter_rewrite_intensity", "medium"),
                "chapter_rewrite_quota": dict(getattr(block, "chapter_rewrite_quota", {}) or {}),
            }
        )
    return summaries


def compute_chapter_rewrite_metrics(
    *,
    guidance: Any | None,
    rewrite_stats: list[RewriteStats],
    original: str = "",
    revised: str = "",
    suffix: str = ".txt",
) -> list[dict[str, object]]:
    if guidance is None:
        return []

    from .body_metrics import (
        _body_blocks_from_guidance_or_text,
        _changed_sentence_count,
        _infer_cluster_rewrite_score,
        _infer_discourse_change_score,
        _normalize_text,
        _revised_body_blocks,
        is_body_policy,
    )

    stats_by_id = {stats.block_id: stats for stats in rewrite_stats if getattr(stats, "block_id", 0)}
    revised_by_id: dict[int, str] = {}
    if original and revised:
        original_body_blocks = _body_blocks_from_guidance_or_text(original, guidance=guidance, suffix=suffix)
        revised_by_id = {
            block_id: text
            for block_id, text in _revised_body_blocks(original_body_blocks, revised, guidance=guidance, suffix=suffix)
        }
    grouped: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for block in getattr(guidance, "block_policies", []):
        if is_body_policy(block):
            key = (
                int(getattr(block, "chapter_id", 0)),
                str(getattr(block, "chapter_title", "") or "Inferred section"),
            )
            grouped[key].append(block)

    metrics: list[dict[str, object]] = []
    for (chapter_id, chapter_title), blocks in grouped.items():
        first = blocks[0]
        priority = str(getattr(first, "chapter_rewrite_priority", "medium") or "medium")
        chapter_type = str(getattr(first, "chapter_type", "unknown") or "unknown")
        quota = quota_for_priority(priority)
        total_blocks = len(blocks)
        changed_blocks = 0
        total_sentences = 0
        changed_sentences = 0
        discourse_score = 0
        cluster_score = 0
        over_rewrite_risk = False
        high_sensitivity_blocks = sum(1 for block in blocks if getattr(block, "high_sensitivity_prose", False))

        for block in blocks:
            text = str(getattr(block, "original_text", "") or "")
            sentence_total = max(1, len(split_sentences(text)))
            total_sentences += sentence_total
            stats = stats_by_id.get(int(getattr(block, "block_id", 0)))
            if stats is None or not stats.changed:
                revised_text = revised_by_id.get(int(getattr(block, "block_id", 0)), text)
                if stats is None and revised_text and _normalize_text(revised_text) != _normalize_text(text):
                    changed_blocks += 1
                    direct_changed = _changed_sentence_count(text, revised_text)
                    changed_sentences += min(sentence_total, max(1, direct_changed))
                    direct_discourse = _infer_discourse_change_score(text, revised_text)
                    direct_cluster = _infer_cluster_rewrite_score(text, revised_text)
                    if direct_cluster <= 0 and direct_changed >= 2 and direct_discourse >= 2:
                        direct_cluster = 1
                    discourse_score += direct_discourse
                    cluster_score += direct_cluster
                    if priority == "conservative" and (
                        direct_cluster >= 1
                        and direct_discourse >= 7
                        and (sentence_total <= 2 or direct_changed / sentence_total >= 0.85)
                    ):
                        over_rewrite_risk = True
                continue
            changed_blocks += 1
            changed_units = max(stats.sentence_level_changes, stats.cluster_changes)
            if changed_units <= 0 and stats.rewrite_coverage > 0:
                changed_units = round(stats.rewrite_coverage * sentence_total)
            changed_sentences += min(sentence_total, max(1, changed_units))
            discourse_score += stats.discourse_change_score
            stats_cluster_changes = stats.cluster_changes
            if (
                priority == "high"
                and stats_cluster_changes <= 0
                and stats.sentence_level_changes >= 2
                and stats.discourse_change_score >= 2
            ):
                stats_cluster_changes = 1
            cluster_score += stats_cluster_changes
            if priority == "conservative" and _has_heavy_conservative_action(stats):
                over_rewrite_risk = True

        coverage = changed_sentences / total_sentences if total_sentences else 0.0
        changed_block_ratio = changed_blocks / total_blocks if total_blocks else 0.0
        effective_quota = quota
        if priority == "high" and total_sentences < 2:
            effective_quota = ChapterQuota(
                coverage_min=quota.coverage_min,
                changed_block_ratio_min=quota.changed_block_ratio_min,
                discourse_min=quota.discourse_min,
                cluster_min=0,
            )
        elif priority == "high" and total_blocks <= 4 and total_sentences <= 12:
            effective_quota = ChapterQuota(
                coverage_min=min(quota.coverage_min, 0.30),
                changed_block_ratio_min=min(quota.changed_block_ratio_min, 0.25),
                discourse_min=quota.discourse_min,
                cluster_min=quota.cluster_min,
            )
        elif priority == "high" and total_blocks and high_sensitivity_blocks / total_blocks >= 0.5:
            effective_quota = ChapterQuota(
                coverage_min=min(quota.coverage_min, 0.50),
                changed_block_ratio_min=quota.changed_block_ratio_min,
                discourse_min=quota.discourse_min,
                cluster_min=quota.cluster_min,
            )
        elif priority == "medium" and _mixed_chapter_priority_split(blocks):
            effective_quota = chapter_quota_by_editable_prose(
                quota=quota,
                total_blocks=total_blocks,
                total_sentences=total_sentences,
                changed_blocks=changed_blocks,
                changed_sentences=changed_sentences,
            )
        elif priority == "medium" and chapter_type in {
            "dataset_description",
            "experiment_setup",
            "evaluation_metrics",
            "system_architecture",
            "system_workflow",
            "deployment_description",
            "training_strategy",
        }:
            effective_quota = ChapterQuota(
                coverage_min=min(quota.coverage_min, 0.30),
                changed_block_ratio_min=quota.changed_block_ratio_min,
                discourse_min=quota.discourse_min,
                cluster_min=quota.cluster_min,
            )
        reasons = _chapter_quota_reasons(
            priority=priority,
            coverage=coverage,
            changed_block_ratio=changed_block_ratio,
            discourse_score=discourse_score,
            cluster_score=cluster_score,
            over_rewrite_risk=over_rewrite_risk,
            quota=effective_quota,
        )
        metrics.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "chapter_type": chapter_type,
                "chapter_rewrite_priority": priority,
                "chapter_rewrite_intensity": target_intensity_for_priority(priority),
                "chapter_rewrite_quota": effective_quota.to_dict(),
                "chapter_blocks_total": total_blocks,
                "chapter_changed_blocks": changed_blocks,
                "chapter_sentences_total": total_sentences,
                "chapter_changed_sentences": changed_sentences,
                "chapter_rewrite_coverage": min(1.0, coverage),
                "chapter_changed_block_ratio": changed_block_ratio,
                "chapter_discourse_change_score": discourse_score,
                "chapter_cluster_rewrite_score": cluster_score,
                "chapter_rewrite_quota_met": not reasons,
                "chapter_policy_consistent": not over_rewrite_risk,
                "reason_codes": reasons,
            }
        )
    return metrics


def body_only_chapter_metrics(metric: dict[str, object]) -> dict[str, object]:
    """Return chapter metrics normalized to editable body prose fields."""

    total_blocks = max(1, int(metric.get("chapter_blocks_total", 0) or 0))
    total_sentences = max(1, int(metric.get("chapter_sentences_total", 0) or 0))
    changed_blocks = int(metric.get("chapter_changed_blocks", 0) or 0)
    changed_sentences = int(metric.get("chapter_changed_sentences", 0) or 0)
    return {
        **metric,
        "chapter_changed_block_ratio": changed_blocks / total_blocks,
        "chapter_rewrite_coverage": min(1.0, changed_sentences / total_sentences),
    }


def chapter_quota_by_editable_prose(
    *,
    quota: ChapterQuota,
    total_blocks: int,
    total_sentences: int,
    changed_blocks: int,
    changed_sentences: int,
) -> ChapterQuota:
    """Relax medium mixed-chapter quotas only when editable prose is tiny or already protected-heavy."""

    if total_blocks <= 1 or total_sentences <= 2:
        return ChapterQuota(
            coverage_min=0.0,
            changed_block_ratio_min=0.0,
            discourse_min=0,
            cluster_min=0,
        )
    if total_blocks <= 2 and changed_blocks >= 1:
        return ChapterQuota(
            coverage_min=min(quota.coverage_min, 0.25),
            changed_block_ratio_min=min(quota.changed_block_ratio_min, 0.25),
            discourse_min=0,
            cluster_min=0,
        )
    if total_sentences <= 5 and changed_sentences >= 1:
        return ChapterQuota(
            coverage_min=min(quota.coverage_min, 0.20),
            changed_block_ratio_min=min(quota.changed_block_ratio_min, 0.20),
            discourse_min=0,
            cluster_min=0,
        )
    return quota


def mixed_chapter_priority_split(blocks: list[Any]) -> bool:
    """Detect medium chapters where a small editable prose surface sits inside protected structure."""

    return _mixed_chapter_priority_split(blocks)


def _mixed_chapter_priority_split(blocks: list[Any]) -> bool:
    if len(blocks) <= 2:
        return True
    technicalish = 0
    for block in blocks:
        text = str(getattr(block, "original_text", "") or "")
        if re.search(r"(图|表|系统|界面|数据库|接口|路径|版本|checkpoint|阈值|字段|E-R)", text, re.I):
            technicalish += 1
    return technicalish / max(1, len(blocks)) >= 0.5


def _has_heavy_conservative_action(stats: RewriteStats) -> bool:
    heavy_actions = {
        "sentence_cluster_rewrite",
        "sentence_cluster_merge",
        "proposition_reorder",
        "discourse_reordering",
        "narrative_path_rewrite",
        "conclusion_absorption",
        "conclusion_absorb",
        "paragraph_reorder",
    }
    actions = {*getattr(stats, "discourse_actions_used", []), *getattr(stats, "structural_actions", [])}
    if any(action in heavy_actions for action in actions):
        return True
    return bool(
        getattr(stats, "rewrite_depth", "") == "developmental_rewrite"
        and getattr(stats, "rewrite_coverage", 0.0) >= 0.85
        and getattr(stats, "discourse_change_score", 0) >= 5
    )


def chapter_quota_reason_codes(metrics: list[dict[str, object]]) -> list[str]:
    reasons: list[str] = []
    for metric in metrics:
        for reason in metric.get("reason_codes", []) or []:
            reasons.append(f"chapter_{metric.get('chapter_id')}_{reason}")
    return reasons


def _chapter_quota_reasons(
    *,
    priority: str,
    coverage: float,
    changed_block_ratio: float,
    discourse_score: int,
    cluster_score: int,
    over_rewrite_risk: bool,
    quota: ChapterQuota,
) -> list[str]:
    reasons: list[str] = []
    if priority == "high":
        if coverage < quota.coverage_min:
            reasons.append("high_priority_chapter_coverage_below_quota")
        if changed_block_ratio < quota.changed_block_ratio_min:
            reasons.append("high_priority_chapter_changed_blocks_below_quota")
        if discourse_score < quota.discourse_min:
            reasons.append("high_priority_chapter_discourse_below_quota")
        if cluster_score < quota.cluster_min:
            reasons.append("high_priority_chapter_cluster_below_quota")
    elif priority == "medium":
        if coverage < quota.coverage_min:
            reasons.append("medium_priority_chapter_coverage_below_quota")
        if changed_block_ratio < quota.changed_block_ratio_min:
            reasons.append("medium_priority_chapter_changed_blocks_below_quota")
        if discourse_score < quota.discourse_min and cluster_score <= 0:
            reasons.append("medium_priority_chapter_sentence_or_cluster_change_missing")
    elif priority == "conservative" and over_rewrite_risk:
        reasons.append("conservative_chapter_over_rewritten")
    return reasons


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text or "")
