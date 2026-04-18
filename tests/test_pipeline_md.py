from pathlib import Path

from airc_skill.config import RewriteMode
from airc_skill.pipeline import rewrite_file


def test_markdown_pipeline_preserves_headings_and_code_blocks(tmp_path: Path) -> None:
    source = tmp_path / "paper.md"
    original = """# 一级标题

本研究主要围绕在线学习平台展开讨论。总的来说，这一议题在很多方面都具有重要意义。

## 方法

研究首先描述场景，其次说明对象。

```python
print("keep this block")
```
"""
    source.write_text(original, encoding="utf-8")

    result = rewrite_file(source, mode=RewriteMode.STRONG)
    revised = result.text

    assert result.review.decision in {"pass", "pass_with_minor_risk", "reject"}
    assert revised.strip()
    assert original.count("# ") + original.count("## ") == revised.count("# ") + revised.count("## ")
    assert original.count("```") == revised.count("```")
