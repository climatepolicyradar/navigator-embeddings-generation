from abc import ABC, abstractmethod
from typing import Sequence
from logging import getLogger
import re

from src.models import Chunk, ChunkType

logger = getLogger(__name__)


class BaseDocumentCleaner(ABC):
    """Base class for perfoming cleaning on a sequence of chunks"""

    @abstractmethod
    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run document cleaning"""
        raise NotImplementedError()


class IdentityDocumentCleaner(BaseDocumentCleaner):
    """Returns all the chunks. Useful for testing."""

    def __call__(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Run document cleaning"""
        return list(chunks)


class ChunkTypeFilter(BaseDocumentCleaner):
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

        self.chunks_to_drop = types_to_remove

    def __call__(self, chunks: Sequence[Chunk]) -> Sequence[Chunk]:
        """Run chunk type filtering."""
        return [
            chunk for chunk in chunks if chunk.chunk_type not in self.chunks_to_drop
        ]


class RemoveShortTableCells(BaseDocumentCleaner):
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
