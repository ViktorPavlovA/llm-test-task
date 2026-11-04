from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVICE_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_DIR = _SERVICE_DIR.parent


def _env_files() -> tuple[str, ...]:
    """
    Collect existing ``.env`` paths for pydantic-settings.

    Returns
    -------
    files : tuple of str
        Service-level file first, then the repository root file.

    """
    files: list[str] = []
    for path in (_SERVICE_DIR / ".env", _REPO_DIR / ".env"):
        if path.is_file():
            files.append(str(path))
    return tuple(files)


class Settings(BaseSettings):
    """
    LLM and web-search settings.

    Keys and URLs come from the environment, never from source code.

    """

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    model_api: str = Field(
        default="http://127.0.0.1:8686/v1",
        validation_alias=AliasChoices("MODEL_API", "LLM_BASE_URL"),
        description="OpenAI-compatible base URL",
    )
    model_api_key: str = Field(
        default="local",
        validation_alias=AliasChoices("MODEL_API_KEY", "LLM_API_KEY"),
    )
    llm_model: str = Field(
        default="local-llm",
        validation_alias=AliasChoices("LLM_MODEL", "MODEL_NAME"),
    )
    llm_temperature: float = Field(default=0.25, ge=0.0, le=1.5)
    llm_max_tokens: int = Field(default=2500, ge=256, le=8192)
    llm_timeout_seconds: float = Field(default=180.0, ge=5.0)
    llm_retries: int = Field(default=2, ge=0, le=4)
    llm_search_internet: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LLM_SEARCH_INTERNET",
            "SEARCH_ENABLED",
            "WEB_SEARCH",
        ),
        description="When true, the research agent may call search_web and fetch_page",
    )
    search_max_results: int = Field(default=5, ge=1, le=10)
    search_timeout_seconds: float = Field(default=12.0, ge=3.0, le=60.0)
    search_context_chars: int = Field(
        default=1800,
        ge=400,
        le=8000,
        validation_alias=AliasChoices("SEARCH_CONTEXT_CHARS", "SEARCH_PAGE_CHARS"),
        description="Max characters of fetched page text sent to the research agent",
    )

    @field_validator("model_api")
    @classmethod
    def api_not_empty(cls, value: str) -> str:
        """
        Strip trailing slashes and reject an empty API URL.

        Parameters
        ----------
        value : str
            Raw ``MODEL_API`` value.

        Returns
        -------
        cleaned : str
            Normalized base URL.

        """
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            raise ValueError("MODEL_API is not set")
        return cleaned

    @field_validator("model_api_key")
    @classmethod
    def key_not_empty(cls, value: str) -> str:
        """
        Reject an empty API key.

        Parameters
        ----------
        value : str
            Raw ``MODEL_API_KEY`` value.

        Returns
        -------
        cleaned : str
            Stripped key.

        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("MODEL_API_KEY is not set")
        return cleaned

    @property
    def chat_url(self) -> str:
        """
        Chat completions endpoint derived from ``model_api``.

        Returns
        -------
        url : str
            ``{model_api}/chat/completions``.

        """
        return f"{self.model_api}/chat/completions"

    @property
    def models_url(self) -> str:
        """
        Models listing endpoint derived from ``model_api``.

        Returns
        -------
        url : str
            ``{model_api}/models``.

        """
        return f"{self.model_api}/models"

    @property
    def health_url(self) -> str:
        """
        llama.cpp health URL (parent of ``/v1`` when present).

        Returns
        -------
        url : str
            ``{origin}/health``.

        """
        base = self.model_api
        if base.endswith("/v1"):
            base = base[:-3]
        return f"{base}/health"

    @property
    def search_snippet_chars(self) -> int:
        """
        Per-result snippet length derived from ``search_context_chars``.

        Returns
        -------
        length : int
            Clip size for DuckDuckGo snippets, between 120 and 400.

        """
        return max(120, min(400, self.search_context_chars // 4))


@lru_cache
def get_settings() -> Settings:
    """
    Return a process-wide Settings instance.

    The ``.env`` files are read once.

    Returns
    -------
    settings : Settings
        Cached configuration object.

    """
    return Settings()
