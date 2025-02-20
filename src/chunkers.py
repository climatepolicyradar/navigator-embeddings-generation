"""Classes for chunking documents."""

from typing import Sequence
from abc import ABC, abstractmethod

from src.models import Chunk


class BaseChunker(ABC):
    """Base class for performing chunking on a document."""

    @abstractmethod
    def __call__(
        self,
        chunks: Sequence[Chunk],
    ) -> Sequence[Chunk]:
        """Run chunker."""

        raise NotImplementedError


class IdentityChunker(BaseChunker):
    """Returns the text blocks converted to chunks, with no attempts at changing them."""

    def __call__(
        self,
        chunks: Sequence[Chunk],
    ) -> list[Chunk]:
        """Run chunker."""

        return list(chunks)


# class GreedyStructureAwareChunker(BaseChunker):
#     """
#     Chunker which makes use of the structure of the document.

#     All headings and subheadings are separate chunks. All consecutive non-heading chunks
#     are merged into single text blocks with the respective type.
#     """

#     def __call__(self, document: Sequence[Chunk]) -> Sequence[Chunk]:
#         """Run chunking."""
