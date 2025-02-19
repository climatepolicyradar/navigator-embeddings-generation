from abc import ABC, abstractmethod
from typing import Sequence

from src.models import Chunk


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
