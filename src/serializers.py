from typing import Optional

from src.models import Chunk, PipelineComponent


class BasicSerializer(PipelineComponent):
    """
    The most basic serializer.

    Returns the chunk with text added in the `serialized_text` field.
    """

    def __call__(self, chunk: Chunk) -> Chunk:
        """Run serialization."""
        chunk.serialized_text = chunk.text
        return chunk


class HeadingAwareSerializer(PipelineComponent):
    """
    Returns the text of the chunk and its heading according to a template.

    Template defaults to "{text} – {heading}". If the chunk doesn't have an associated
    heading, just the text will be returned.
    """

    def __init__(self, template: Optional[str]) -> None:
        self.template: str = template or "{text} – {heading}"

    def __call__(self, chunk: Chunk) -> Chunk:
        """Run serialization."""

        chunk.serialized_text = (
            self.template.format(text=chunk.text, heading=chunk.heading)
            if chunk.heading
            else chunk.text
        )

        return chunk
