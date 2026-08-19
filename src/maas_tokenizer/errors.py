"""Typed service failures."""


class TokenCountError(RuntimeError):
    """Base class for expected token-count failures."""


class UnknownModelError(TokenCountError):
    """The request selected an unregistered model."""


class ProcessorRequiredError(TokenCountError):
    """A multimodal processor is required to determine input tokens."""


class RequestProcessingError(TokenCountError):
    """The request could not be validated, rendered, or encoded."""

