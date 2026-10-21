import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from minirag.chunking import chunk  # noqa: E402


def test_empty_text_returns_no_chunks():
    assert chunk("", size=100, overlap=10) == []


def test_whitespace_only_text_returns_no_chunks():
    assert chunk("  \n\n\t ", size=100, overlap=10) == []


def test_text_shorter_than_size_is_one_chunk():
    result = chunk("hello world", size=100, overlap=10, source="a.md")
    assert result == [{"source": "a.md", "heading": "", "text": "hello world"}]


def test_text_exactly_size_is_one_chunk():
    assert len(chunk("x" * 50, size=50, overlap=10)) == 1


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 11), (0, 0), (-5, 0), (10, -1)])
def test_invalid_size_or_overlap_raises(size, overlap):
    with pytest.raises(ValueError):
        chunk("some text", size=size, overlap=overlap)


def test_overlap_is_actually_shared():
    text = "".join(chr(ord("a") + i % 26) for i in range(100))
    result = chunk(text, size=30, overlap=10)
    assert len(result) > 1
    for prev, nxt in zip(result, result[1:]):
        assert prev["text"][-10:] == nxt["text"][:10]


def test_windows_cover_whole_text_without_gaps():
    text = "".join(chr(ord("a") + i % 26) for i in range(100))
    result = chunk(text, size=30, overlap=10)
    rebuilt = result[0]["text"] + "".join(c["text"][10:] for c in result[1:])
    assert rebuilt == text


def test_zero_overlap_partitions_text():
    text = "abcdefghij" * 5
    result = chunk(text, size=10, overlap=0)
    assert "".join(c["text"] for c in result) == text


def test_no_tiny_duplicate_tail_chunk():
    # 40 chars, size 30, overlap 10: second window ends exactly at the end.
    result = chunk("x" * 40, size=30, overlap=10)
    assert [len(c["text"]) for c in result] == [30, 20]


def test_no_heading_gives_empty_heading():
    result = chunk("just a paragraph\nof text", size=100, overlap=0)
    assert result[0]["heading"] == ""


def test_headings_split_sections_and_label_chunks():
    text = "# Title\nintro\n\n## First\nalpha\n\n## Second\nbeta\n"
    result = chunk(text, size=100, overlap=0, source="d.md")
    assert [(c["heading"], c["text"]) for c in result] == [
        ("Title", "intro"),
        ("First", "alpha"),
        ("Second", "beta"),
    ]
    assert all(c["source"] == "d.md" for c in result)


def test_text_before_first_heading_is_kept():
    result = chunk("preamble\n# H\nbody", size=100, overlap=0)
    assert [(c["heading"], c["text"]) for c in result] == [("", "preamble"), ("H", "body")]


def test_empty_section_is_skipped():
    text = "# A\n\n# B\nreal content\n# C\n   \n"
    result = chunk(text, size=100, overlap=0)
    assert [(c["heading"], c["text"]) for c in result] == [("B", "real content")]


def test_chunks_do_not_cross_heading_boundaries():
    text = "# A\n" + "a" * 25 + "\n# B\n" + "b" * 25
    result = chunk(text, size=20, overlap=5)
    for c in result:
        assert set(c["text"]) <= ({"a"} if c["heading"] == "A" else {"b"})


def test_long_section_keeps_its_heading_on_every_chunk():
    result = chunk("# Long\n" + "word " * 100, size=50, overlap=10)
    assert len(result) > 1
    assert {c["heading"] for c in result} == {"Long"}


def test_hash_inside_code_fence_is_not_a_heading():
    text = "# Real\n```bash\n# install deps\npip install x\n```\nafter"
    result = chunk(text, size=500, overlap=0)
    assert len(result) == 1
    assert result[0]["heading"] == "Real"
    assert "# install deps" in result[0]["text"]


def test_hash_without_space_is_not_a_heading():
    result = chunk("#hashtag is text", size=100, overlap=0)
    assert result[0]["heading"] == ""


def test_trailing_hashes_stripped_from_heading():
    result = chunk("## Title ##\nbody", size=100, overlap=0)
    assert result[0]["heading"] == "Title"
