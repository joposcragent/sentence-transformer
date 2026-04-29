import numpy as np

from st_service.log_preview import PREVIEW_MAX_CHARS, preview_text, preview_vector


def test_preview_text_short_unchanged():
    s = "hello"
    assert preview_text(s) == s


def test_preview_text_truncates_with_ellipsis():
    s = "a" * (PREVIEW_MAX_CHARS + 10)
    out = preview_text(s)
    assert len(out) == PREVIEW_MAX_CHARS + len("...")
    assert out.endswith("...")
    assert out[:PREVIEW_MAX_CHARS] == "a" * PREVIEW_MAX_CHARS


def test_preview_text_exact_max_no_ellipsis():
    s = "b" * PREVIEW_MAX_CHARS
    assert preview_text(s) == s


def test_preview_vector_list_short():
    v = [1.0, 2.0, 3.0]
    assert preview_vector(v) == str(v)


def test_preview_vector_numpy():
    v = np.array([1.0, 2.0], dtype=np.float64)
    assert preview_vector(v) == "[1.0, 2.0]"


def test_preview_vector_truncates_long_string():
    v = list(range(200))
    s = str(v)
    assert len(s) > PREVIEW_MAX_CHARS
    out = preview_vector(v)
    assert len(out) == PREVIEW_MAX_CHARS + len("...")
    assert out.endswith("...")
    assert out[:PREVIEW_MAX_CHARS] == s[:PREVIEW_MAX_CHARS]
