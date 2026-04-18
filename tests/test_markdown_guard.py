from airc_skill.markdown_guard import protect, restore


def test_markdown_guard_round_trip() -> None:
    original = """# Title

Here is `inline code`, a [link](https://example.com), and an image ![alt](img.png).

$$
E = mc^2
$$

```python
print("hello")
```
"""

    protected, placeholders = protect(original)

    assert protected != original
    assert placeholders
    assert restore(protected, placeholders) == original
