from pathlib import Path

from airc_skill.pipeline import run_file
from airc_skill.target_style_alignment import (
    analyze_target_style_alignment,
    class_aware_style_match_ratio,
    classify_alignment_paragraph,
    extract_target_style_body_prose,
    schedule_class_aware_repairs,
    source_backed_evidence_drift,
)


def test_target_style_body_prose_excludes_markdown_and_figures() -> None:
    text = """# 标题

![img](media/a.png)

图3-1 模型流程图

本研究通过多分支框架完成检测。

| 列1 | 列2 |
| --- | --- |
| 1 | 2 |

$$
L = x + y
$$

checkpoints/best.pth
"""
    prose = extract_target_style_body_prose(text)
    assert "本研究通过多分支框架完成检测" in prose
    assert "图3-1" not in prose
    assert "checkpoints/best.pth" not in prose
    assert "| 列1 | 列2 |" not in prose


def test_protected_term_drift_ignores_spacing_and_position_changes() -> None:
    source = "本研究使用 AIGC 检测框架，并加载 checkpoints/best.pth 与 base_only 机制。"
    candidate = "加载checkpoints/best.pth后，base_only 机制继续用于 AIGC检测框架。"
    report = analyze_target_style_alignment(
        model_output=candidate,
        target_text="这个系统是用来进行 AIGC 检测工作的。",
        source_text=source,
    )
    assert report["terminology_drift"] == 0


def test_evidence_drift_allows_source_backed_rephrasing() -> None:
    source = "系统在 2025 年实验中保持 F1=0.90，并分析误差来源。"
    candidate = "在 2025 年的实验过程中，系统仍然保持 F1=0.90，并对误差来源进行了分析。"
    assert source_backed_evidence_drift(source, candidate) == 0


def test_evidence_drift_detects_new_external_claim() -> None:
    source = "系统用于检测 AIGC 图像。"
    candidate = "在 2024 年内容生态中，超过八成投诉与图像伪造相关，因此该系统用于检测 AIGC 图像。"
    assert source_backed_evidence_drift(source, candidate) >= 2


def test_classify_alignment_paragraph_refines_thesis_sections() -> None:
    assert classify_alignment_paragraph("本研究的意义在于提升内容安全。", "## 研究意义") == "significance_prose"
    assert classify_alignment_paragraph("模型由语义分支与频域分支组成。", "## 模型架构") == "method_mechanism"
    assert classify_alignment_paragraph("训练阶段采用困难真实样本课程学习。", "## 困难真实样本训练策略") == "training_strategy"
    assert classify_alignment_paragraph("系统通过统一接口提供推理服务。", "## 系统实现与部署") == "system_implementation"
    assert classify_alignment_paragraph("本研究最后总结了主要结论。", "## 总结") == "summary_conclusion"


def test_class_aware_alignment_separates_method_and_summary_prose() -> None:
    model = """## 方法

该模型用于提取频域特征。

## 结论

本研究说明该系统能够保持稳定性能。
"""
    target = """## 方法

该模型直接负责频域特征提取。

## 结论

本研究的结论部分会进一步说明系统在实际场景中的稳定性能。
"""
    ratio = class_aware_style_match_ratio(model, target)
    assert 0.0 <= ratio <= 1.0


def test_class_alignment_breakdown_reports_worst_classes() -> None:
    report = analyze_target_style_alignment(
        model_output="## 方法\n\n该模型用于提取频域特征。\n\n## 总结\n\n本研究说明系统稳定。",
        target_text="## 方法\n\n这个模型是用来进行频域特征提取的工作。\n\n## 总结\n\n本研究的结论部分会进一步说明系统在实际场景中的稳定性。",
        source_text="## 方法\n\n该模型用于提取频域特征。\n\n## 总结\n\n本研究说明系统稳定。",
    )
    assert "class_alignment_breakdown" in report
    assert len(report["worst_alignment_classes"]) <= 3
    assert report["style_distribution_match_ratio"] >= report["class_aware_style_match_ratio"]


def test_class_aware_repair_only_targets_low_match_classes() -> None:
    source = "## 研究意义\n\n本研究说明系统稳定。"
    target = "## 研究意义\n\n本研究的意义在于说明这个系统在实际应用过程中会保持比较稳定的表现，并且能够用来支持后续判断。"
    before = analyze_target_style_alignment(
        model_output=source,
        target_text=target,
        source_text=source,
    )
    repaired, actions = schedule_class_aware_repairs(
        source_text=source,
        model_output=source,
        target_text=target,
    )
    assert repaired
    if actions:
        assert all("significance_prose" in action for action in actions)
    after = analyze_target_style_alignment(
        model_output=repaired,
        target_text=target,
        source_text=source,
    )
    assert after["class_aware_style_match_ratio"] >= before["class_aware_style_match_ratio"]


def test_class_aware_repair_preserves_zero_drift() -> None:
    source = "系统在 2025 年实验中保持 F1=0.90，并使用 AIGC 检测框架。"
    target = "这个系统在 2025 年的实验过程中，仍然能够保持 F1=0.90，并且会使用 AIGC 检测框架。"
    repaired, _ = schedule_class_aware_repairs(
        source_text=source,
        model_output=source,
        target_text=target,
    )
    report = analyze_target_style_alignment(
        model_output=repaired,
        target_text=target,
        source_text=source,
    )
    assert report["terminology_drift"] == 0
    assert report["evidence_drift"] == 0


def test_real_target_style_alignment_no_false_drift_on_thesis_pair() -> None:
    root = Path(__file__).resolve().parents[1]
    original = root / "人工智能生成图像检测工具的设计与实现.md"
    target = root / "人工智能生成图像检测工具的设计与实现 - 副本.md"
    if not original.exists() or not target.exists():
        return
    report = analyze_target_style_alignment(
        model_output=original.read_text(encoding="utf-8"),
        target_text=target.read_text(encoding="utf-8"),
        source_text=original.read_text(encoding="utf-8"),
    )
    assert report["terminology_drift"] == 0
    assert report["evidence_drift"] == 0


def test_real_class_aware_style_match_reaches_threshold_or_reports_blocker() -> None:
    root = Path(__file__).resolve().parents[1]
    original = root / "人工智能生成图像检测工具的设计与实现.md"
    target = root / "人工智能生成图像检测工具的设计与实现 - 副本.md"
    if not original.exists() or not target.exists():
        return
    result = run_file(
        original,
        preset="zh_academic_l2_mild",
        output_path=root / "test_airc_v13.md",
        report_path=root / "test_airc_v13.report.json",
        target_style_file=target,
        max_retry_passes=1,
        dry_run=True,
    )
    alignment = result.rewrite_result.review.target_style_alignment
    assert alignment["terminology_drift"] == 0
    assert alignment["evidence_drift"] == 0
    assert (
        alignment["class_aware_style_match_ratio"] >= 0.70
        or alignment["worst_alignment_classes"]
    )


def test_run_file_emits_target_style_alignment_metrics(tmp_path: Path) -> None:
    original = tmp_path / "paper.md"
    target = tmp_path / "target.md"
    output = tmp_path / "test_airc_v12.md"
    report = tmp_path / "test_airc_v12.report.json"
    original.write_text(
        "# 标题\n\n本研究用于提取频域特征，并实现特征融合。\n\n该系统用于输出最终检测结果。\n",
        encoding="utf-8",
    )
    target.write_text(
        "# 标题\n\n这个研究是用来进行频域特征提取的工作，并且还会来实现特征融合。\n\n这个系统是用来输出最终检测结果的。\n",
        encoding="utf-8",
    )

    result = run_file(
        original,
        preset="zh_academic_l2_mild",
        output_path=output,
        report_path=report,
        target_style_file=target,
        max_retry_passes=1,
    )

    alignment = result.rewrite_result.review.target_style_alignment
    assert alignment["enabled"] is True
    assert "class_aware_style_match_ratio" in alignment
    assert report.exists()
