"""Tests for the README results updater."""

from pathlib import Path

from eval.readme import END_MARKER, START_MARKER, build_section, update_readme


def _write_readme(tmp_path: Path) -> Path:
    path = tmp_path / "README.md"
    path.write_text(
        "# Title\n\nSome text.\n\n"
        f"{START_MARKER}\n## Measured results\n\nold content\n{END_MARKER}\n\nAfter.\n",
        encoding="utf-8",
    )
    return path


def test_update_rewrites_the_block(tmp_path):
    readme = _write_readme(tmp_path)
    assert update_readme(readme, tmp_path) is True
    text = readme.read_text(encoding="utf-8")
    assert "old content" not in text
    assert "Measured results" in text
    assert "Title" in text and "After." in text


def test_update_is_idempotent(tmp_path):
    readme = _write_readme(tmp_path)
    update_readme(readme, tmp_path)
    assert update_readme(readme, tmp_path) is False


def test_update_without_markers_is_a_no_op(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("# No markers here\n", encoding="utf-8")
    assert update_readme(path, tmp_path) is False


def test_section_reports_when_there_is_no_run(tmp_path):
    section = build_section(tmp_path)
    assert "No run recorded yet" in section
