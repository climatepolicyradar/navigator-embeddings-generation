"""
Components which processes sequences of chunks in the pipeline.

These could filter, clean or modify the chunks in some way.
"""

from typing import Sequence, Optional
from logging import getLogger
import re

from src.models import Chunk, ChunkType, PipelineComponent

logger = getLogger(__name__)


class IdentityChunkProcessor(PipelineComponent):
    """Returns all the chunks. Useful for testing."""

    def __call__(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Run document cleaning"""
        return list(chunks)


class ChunkTypeFilter(PipelineComponent):
    """Filter out chunks of specified types."""

    def __init__(self, types_to_remove: list[str]) -> None:
        """
        Args:

        :param types_to_remove: the types of chunk to remove
        """
        for _type in types_to_remove:
            try:
                ChunkType(_type)
            except NameError:
                logger.warning(
                    f"Blocks to filter should be of a known block type, removing {_type} "
                    f"from the list. "
                )
                types_to_remove.remove(_type)

        self.types_to_remove = types_to_remove

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run chunk type filtering."""
        return [
            chunk for chunk in chunks if chunk.chunk_type not in self.types_to_remove
        ]


class RemoveShortTableCells(PipelineComponent):
    """
    Remove table cells under a certain number of characters, or are all numeric.

    These aren't useful for encoding or search.
    """

    def __init__(self, min_chars: int = 10, remove_all_numeric: bool = True) -> None:
        self.min_chars = min_chars
        self.remove_all_numeric = remove_all_numeric

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run table cell filtering."""
        new_chunks: list[Chunk] = []

        for chunk in chunks:
            if chunk.chunk_type != ChunkType.TABLE_CELL:
                new_chunks.append(chunk)
                continue

            if len(chunk.text) < self.min_chars:
                continue

            # Matches strings that are entirely numeric, optionally with a +/- sign,
            # commas, spaces and decimal point
            if self.remove_all_numeric and re.match(
                r"^[+-]?[\d,\s]*\.?\d+$", chunk.text.strip()
            ):
                continue

            new_chunks.append(chunk)

        return new_chunks


class RemoveRepeatedAdjacentChunks(PipelineComponent):
    """
    Remove chunks of the same type that are repeated, keeping the first.

    This is useful for headers, footers and headings that may be repeated once per page
    in a document.
    """

    def __init__(
        self,
        chunk_types=[
            ChunkType.SECTION_HEADING,
            ChunkType.TITLE,
            ChunkType.PAGE_HEADER,
            ChunkType.PAGE_FOOTER,
            ChunkType.FOOTNOTE,
        ],
        ignore_case: bool = True,
    ) -> None:
        """
        Args:

        :param chunk_types: list of chunk types to check for repeating
        :param ignore_case: whether filtering ignores case. Defaults to True
        """
        self.chunk_types = chunk_types
        self.ignore_case = ignore_case

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run repeated adjacent chunk filtering."""
        new_chunks: list[Chunk] = []
        current_chunk_of_type: dict[ChunkType, Optional[str]] = {
            chunk_type: None for chunk_type in self.chunk_types
        }

        for chunk in chunks:
            if chunk.chunk_type not in self.chunk_types:
                new_chunks.append(chunk)
                continue

            current_text = current_chunk_of_type[chunk.chunk_type]
            chunk_text = chunk.text.lower() if self.ignore_case else chunk.text

            match current_text:
                case None:
                    # First time seeing this chunk type
                    current_chunk_of_type[chunk.chunk_type] = chunk_text
                    new_chunks.append(chunk)
                case matched_text if matched_text != chunk_text:
                    # Different text than previous chunk of this type
                    current_chunk_of_type[chunk.chunk_type] = chunk_text
                    new_chunks.append(chunk)
                case _:
                    # Same text as previous chunk of this type, skip it
                    continue

        return new_chunks


class AddHeadings(PipelineComponent):
    """
    Add headings to chunks.

    Only works at a single-level. This means that (subheading -> heading) and
    (text -> subheading) relationships will exist, but not (text -> heading) if there
    is a subheading between them.
    """

    def __init__(self) -> None:
        self.heading_types = {ChunkType.TITLE}
        self.subheading_types = {ChunkType.SECTION_HEADING, ChunkType.PAGE_HEADER}

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Add headings to chunks."""

        current_heading = None
        current_subheading = None

        # Make a copy of the chunks
        new_chunks = list(chunks)

        for chunk in new_chunks:
            if chunk.chunk_type in self.heading_types:
                current_heading = chunk
                # A heading is the top level, so we don't want it to have a heading
                # itself
                continue
            elif chunk.chunk_type in self.subheading_types:
                current_subheading = chunk
                chunk.heading = current_heading
                continue

            chunk.heading = current_subheading or current_heading

        return new_chunks
