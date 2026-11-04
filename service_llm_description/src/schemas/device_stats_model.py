from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PROMPT_INJECTION_RE = re.compile(
    r"(?:"
    r"<\|im_start\|>"
    r"|<\|im_end\|>"
    r"|<\|system\|>"
    r"|</?system>"
    r"|<<\s*SYS\s*>>"
    r"|\[/?INST\]"
    r"|\"role\"\s*:\s*\"(?:system|assistant|developer)\""
    r"|(?:^|[\n\r])\s*(?:system|assistant|developer)\s*:"
    r"|system\s+prompt"
    r"|ignore\s+(?:all\s+)?previous\s+instructions"
    r"|forget\s+(?:your\s+)?(?:system\s+)?instructions"
    r"|системн(?:ый|ого|ые)\s+промпт"
    r"|игнорируй\s+(?:все\s+)?предыдущие\s+инструкции"
    r"|забудь\s+(?:свои\s+)?инструкции"
    r")",
    re.IGNORECASE,
)


def _reject_prompt_injection(value: str) -> str:
    """
    Reject text that looks like a system prompt or instruction override.

    Parameters
    ----------
    value : str
        Normalized field text.

    Returns
    -------
    value : str
        The same text when it is safe.

    Raises
    ------
    ValueError
        If the text contains a chat-template marker or jailbreak phrase.

    """
    if _PROMPT_INJECTION_RE.search(value):
        raise ValueError(
            "Field must not contain a system prompt or instruction override"
        )
    return value


def _scan_spec_value(raw: Any) -> None:
    """
    Walk spec values and reject prompt-injection strings.

    Parameters
    ----------
    raw : Any
        Spec value: text, number, list, or nested object.

    """
    if isinstance(raw, str):
        cleaned = " ".join(raw.split())
        if cleaned:
            _reject_prompt_injection(cleaned)
        return
    if isinstance(raw, dict):
        for key, item in raw.items():
            label = " ".join(str(key).split())
            if label:
                _reject_prompt_injection(label)
            _scan_spec_value(item)
        return
    if isinstance(raw, list):
        for item in raw:
            _scan_spec_value(item)


class DeviceStatsModel(BaseModel):
    """
    Instrument record. Field names match the assignment JSON.

    Parameters
    ----------
    name : str
        Instrument name.
    manufacturer : str or None
        Maker, if present.
    specs : dict
        Technical specifications as key-value pairs.
    description : str or None
        Datasheet fragment or free-text description.
    type : str or None
        Instrument type.
    model : str or None
        Model code.
    additional_info : str or None
        Legacy alias of ``description``.

    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1, max_length=200, description="Instrument name")
    manufacturer: str | None = Field(None, max_length=150, description="Manufacturer")
    specs: dict[str, Any] = Field(
        default_factory=dict,
        description="Technical specifications",
    )
    description: str | None = Field(
        None,
        max_length=5000,
        description="Text description or datasheet fragment",
    )
    type: str | None = Field(None, max_length=100, description="Instrument type")
    model: str | None = Field(None, max_length=100, description="Model")
    additional_info: str | None = Field(
        None,
        max_length=5000,
        description="Legacy alias of description",
    )

    @field_validator(
        "name",
        "manufacturer",
        "description",
        "type",
        "model",
        "additional_info",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        """
        Collapse whitespace; empty strings become None.

        Also reject system-prompt and jailbreak fragments.

        Parameters
        ----------
        value : str or None
            Raw text field.

        Returns
        -------
        cleaned : str or None
            Normalized text, or None when empty.

        """
        if value is None:
            return None
        cleaned = " ".join(str(value).split())
        if not cleaned:
            return None
        return _reject_prompt_injection(cleaned)

    @field_validator("specs")
    @classmethod
    def specs_is_dict(cls, value: Any) -> dict[str, Any]:
        """
        Require ``specs`` to be a mapping with non-empty keys.

        Parameters
        ----------
        value : Any
            Incoming specs payload.

        Returns
        -------
        cleaned : dict
            Mapping with stripped keys. None becomes an empty dict.

        """
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("specs must be an object {name: value}")
        cleaned: dict[str, Any] = {}
        for key, raw in value.items():
            label = " ".join(str(key).split())
            if not label:
                continue
            _reject_prompt_injection(label)
            if raw is None:
                continue
            if isinstance(raw, str) and not raw.strip():
                continue
            _scan_spec_value(raw)
            cleaned[label] = raw
        return cleaned

    @model_validator(mode="after")
    def fill_description_alias(self) -> DeviceStatsModel:
        """
        Copy ``additional_info`` into ``description`` when the latter is empty.

        Returns
        -------
        device : DeviceStatsModel
            The same instance after alias fill.

        """
        if not self.name:
            raise ValueError("Name must not be empty")
        if self.description is None and self.additional_info:
            self.description = self.additional_info
        return self
