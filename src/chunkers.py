"""Classes for chunking documents."""

from typing import Sequence
from abc import ABC, abstractmethod

from cpr_sdk.parser_models import ParserOutput, PDFTextBlock
from src.models import ChunkType, Chunk


class BaseChunker(ABC):
    """Base class for performing chunking on a document."""

    @abstractmethod
    def __call__(
        self,
        document: ParserOutput,
    ) -> Sequence[Chunk]:
        """Run chunker."""

        raise NotImplementedError


class IdentityChunker(BaseChunker):
    """Returns the text blocks converted to chunks, with no attempts at changing them."""

    def __call__(
        self,
        document: ParserOutput,
    ) -> list[Chunk]:
        """Run chunker."""

        if not document.text_blocks:
            return []

        chunks = [
            Chunk(
                id=text_block.text_block_id,
                text=text_block.to_string(),
                chunk_type=ChunkType(text_block.type),
                bounding_boxes=[text_block.coords]
                if isinstance(text_block, PDFTextBlock) and text_block.coords
                else None,
                pages=[text_block.page_number]
                if isinstance(text_block, PDFTextBlock)
                else None,
            )
            for text_block in document.text_blocks
        ]

        return chunks
