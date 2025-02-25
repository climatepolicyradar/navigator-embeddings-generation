from typing import Optional
from abc import ABC, abstractmethod
import logging

from pydantic import BaseModel

from cpr_sdk.parser_models import BlockType

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    """A unit part of a document."""

    id: str
    text: str
    chunk_type: BlockType
    # TODO: do we want multiple headings here? this is what docling does.
    heading: Optional["Chunk"] = None
    # Bounding boxes can be arbitrary polygons according to Azure
    # TODO: is this true according to the backend or frontend?
    bounding_boxes: Optional[list[list[tuple[float, float]]]]
    pages: Optional[list[int]]
    tokens: Optional[list[str]] = None
    serialized_text: Optional[str] = None

    def _verify_bbox_and_pages(self) -> None:
        if self.bounding_boxes is not None and self.pages is not None:
            assert len(self.bounding_boxes) == len(self.pages)

    def merge(self, other: "Chunk", text_separator: str = " ") -> "Chunk":
        """
        Merge the chunk with another chunk.

        The ID and chunk type are taken from the first chunk.
        """

        nullable_properties_with_issue = [
            property_name
            for property_name in ["bounding_boxes", "pages"]
            if (getattr(self, property_name) is None)
            != (getattr(other, property_name) is None)
        ]

        optional_properties_with_issue = [
            property_name
            for property_name in ["heading", "tokens", "serialized_text"]
            if (getattr(self, property_name) is None)
            or (getattr(other, property_name) is None)
        ]

        if nullable_properties_with_issue:
            raise ValueError(
                f"Properties {nullable_properties_with_issue} of chunks being merged must be either both None or both not None."
            )

        if optional_properties_with_issue:
            logger.warning(
                f"Properties {optional_properties_with_issue} of chunks being merged have been set for one or more chunks. These properties will be lost in the merge."
            )

        combined_bounding_boxes = (
            self.bounding_boxes + other.bounding_boxes
            if (self.bounding_boxes is not None) and (other.bounding_boxes is not None)
            else None
        )
        combined_pages = (
            self.pages + other.pages
            if (self.pages is not None) and (other.pages is not None)
            else None
        )

        return Chunk(
            # TODO: can we better handle IDs when merging chunks?
            id=self.id,
            text=text_separator.join([self.text, other.text]),
            chunk_type=self.chunk_type,
            bounding_boxes=combined_bounding_boxes,
            pages=combined_pages,
            heading=None,
            tokens=None,
            serialized_text=None,
        )


class PipelineComponent(ABC):
    """
    A component of the pipeline.

    When called, takes a list of chunks as input and returns a list of chunks. This
    should be used as the base class for every pipeline component except for the
    encoder.
    """

    @abstractmethod
    def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        """Base class for any pipeline component."""
        raise NotImplementedError
