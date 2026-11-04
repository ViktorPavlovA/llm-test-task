class DescriptionServiceError(Exception):
    """
    Base error for the product-card pipeline.

    """


class LLMUnavailableError(DescriptionServiceError):
    """
    Raised when the local LLM endpoint cannot be reached in time.

    """


class LLMBadResponseError(DescriptionServiceError):
    """
    Raised when the model reply cannot be turned into a valid card.

    """
