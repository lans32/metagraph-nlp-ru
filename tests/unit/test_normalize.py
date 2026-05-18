from metagraph_nlp.parsers.normalize import normalize_text


def test_normalize_quotes_and_dashes():
    src = "Он сказал: \u00abпривет\u00bb \u2014 и ушёл."
    out = normalize_text(src)
    assert "\u00ab" not in out and "\u00bb" not in out
    assert "\u2014" not in out
    assert '"привет"' in out


def test_normalize_whitespace_and_newlines():
    src = "Первая строка.\r\n\r\n\r\n   Вторая   строка.\t\tТретья."
    out = normalize_text(src)
    assert "\r" not in out
    assert "   " not in out
    assert "\n\n\n" not in out


def test_normalize_empty():
    assert normalize_text("") == ""
