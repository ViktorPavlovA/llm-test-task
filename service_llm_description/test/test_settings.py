from src.utils.config_read import Settings


def test_llm_search_internet_false_from_env(monkeypatch):
    """
    Treat ``LLM_SEARCH_INTERNET=false`` as tools disabled.

    """
    monkeypatch.setenv("LLM_SEARCH_INTERNET", "false")
    settings = Settings()
    assert settings.llm_search_internet is False


def test_llm_search_internet_true_from_env(monkeypatch):
    """
    Treat ``LLM_SEARCH_INTERNET=true`` as tools enabled.

    """
    monkeypatch.setenv("LLM_SEARCH_INTERNET", "true")
    settings = Settings()
    assert settings.llm_search_internet is True


def test_search_context_chars_from_env(monkeypatch):
    """
    Read ``SEARCH_CONTEXT_CHARS`` and derive snippet clip size.

    """
    monkeypatch.setenv("SEARCH_CONTEXT_CHARS", "1200")
    settings = Settings()
    assert settings.search_context_chars == 1200
    assert settings.search_snippet_chars == 300
