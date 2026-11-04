from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AnswerLLMModel(BaseModel):
    """
    Draft product card assembled from locked specs and LLM copy.

    Parameters
    ----------
    title : str
        SEO title, at most 70 characters.
    description : str
        SEO body, 800 to 1200 characters.
    key_specs : list of str
        Product-specific specifications confirmed by input or search.
    search_queries : list of str
        Three to five search queries.
    commercial_block : str
        Who the instrument is for and which jobs it covers.
    data_missing : list of str
        Fields that still need a supplier value.
    review_notes : list of str
        Editor notes that must not be published as-is.

    """

    title: str = Field(
        ..., max_length=70, description="SEO title, at most 70 characters"
    )
    description: str = Field(
        ...,
        min_length=800,
        max_length=1200,
        description="SEO description, 800 to 1200 characters",
    )
    key_specs: list[str] = Field(
        ...,
        min_length=1,
        description="Confirmed product-specific specifications",
    )
    search_queries: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Search queries",
    )
    commercial_block: str = Field(
        ...,
        min_length=1,
        description="Commercial blurb for the catalog manager",
    )
    data_missing: list[str] = Field(
        default_factory=list,
        description="Fields that have no source data",
    )
    review_notes: list[str] = Field(
        default_factory=list,
        description="Editor notes, not for publication",
    )

    @field_validator("title")
    @classmethod
    def title_length(cls, value: str) -> str:
        """
        Collapse whitespace and reject an empty or overlong title.

        Parameters
        ----------
        value : str
            Raw title from the model or fallback.

        Returns
        -------
        cleaned : str
            Normalized title of at most 70 characters.

        """
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Title must not be empty")
        if len(cleaned) > 70:
            raise ValueError("Title must not exceed 70 characters")
        return cleaned

    @field_validator("description")
    @classmethod
    def description_length(cls, value: str) -> str:
        """
        Require a description length between 800 and 1200 characters.

        Parameters
        ----------
        value : str
            Raw SEO description.

        Returns
        -------
        cleaned : str
            Stripped description inside the allowed range.

        """
        cleaned = value.strip()
        length = len(cleaned)
        if length < 800 or length > 1200:
            raise ValueError(f"Description must be 800-1200 characters (got {length})")
        return cleaned

    @field_validator("key_specs", "search_queries", "data_missing", "review_notes")
    @classmethod
    def lists_clean(cls, value: list[str]) -> list[str]:
        """
        Drop empty items and strip surrounding whitespace.

        Parameters
        ----------
        value : list of str
            Raw list from the model.

        Returns
        -------
        cleaned : list of str
            Non-empty stripped strings.

        """
        return [item.strip() for item in value if item and str(item).strip()]

    @field_validator("commercial_block")
    @classmethod
    def commercial_not_empty(cls, value: str) -> str:
        """
        Reject an empty commercial block.

        Parameters
        ----------
        value : str
            Raw commercial text.

        Returns
        -------
        cleaned : str
            Stripped non-empty text.

        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Commercial block must not be empty")
        return cleaned


class CardResponse(AnswerLLMModel):
    """
    API payload: card fields plus copy-ready markdown.

    Parameters
    ----------
    markdown : str
        The same card rendered as Markdown.
    sources : list of str
        Pages used when the research agent filled missing specs.
    web_used : bool
        True when the research node ran.

    """

    markdown: str = Field(..., description="The same card as Markdown")
    sources: list[str] = Field(
        default_factory=list,
        description="Pages the agent used for missing specs",
    )
    web_used: bool = Field(
        default=False,
        description="Whether internet search ran for this card",
    )
