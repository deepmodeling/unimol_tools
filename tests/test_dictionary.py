from unimol_tools.data.dictionary import Dictionary


def test_dictionary_basic():
    d = Dictionary()
    d.add_symbol(d.unk_word, is_special=True)
    idx = d.add_symbol("foo")
    assert "foo" in d
    assert d.index("foo") == idx
    assert d[idx] == "foo"
    unk_idx = d.index(d.unk_word)
    assert d.index("bar") == unk_idx