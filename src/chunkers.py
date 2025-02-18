"""Classes for chunking documents."""

from typing import Optional, Sequence
from enum import Enum
from abc import ABC

from pydantic import BaseModel
from cpr_sdk.models import BaseDocument


class ChunkType(str, Enum):
    """Possible types of chunk"""

    # TODO: should this replace BlockType in the SDK?

    TEXT = "Text"
    TITLE = "Title"
    LIST = "List"
    TABLE = "Table"
    TABLE_CELL = "TableCell"
    FIGURE = "Figure"
    PAGE_HEADER = "pageHeader"
    PAGE_FOOTER = "pageFooter"
    PAGE_NUMBER = "pageNumber"
    SECTION_HEADING = "sectionHeading"
    DOCUMENT_HEADER = "Document Header"
    FOOTNOTE = "footnote"


class Chunk(BaseModel):
    """A unit part of a document."""

    text: str
    chunk_type: ChunkType
    headings: Optional[Sequence["Chunk"]] = None
    # Bounding boxes can be arbitrary polygons according to Azure
    # TODO: is this true according to the backend or frontend?
    bounding_boxes: Optional[list[list[tuple[float, float]]]]
    pages: Optional[list[int]]

    def _verify_bbox_and_pages(self) -> None:
        if self.bounding_boxes is not None and self.pages is not None:
            assert len(self.bounding_boxes) == len(self.pages)


class BaseChunker(ABC):
    """Base class for performing chunking on a document."""

    def __call__(
        self,
        document: BaseDocument,
    ) -> Sequence[Chunk]:
        """
        [(text, text_block_type), ...]

        HEADING 1.1
        TEXT --> merge/split
        TEXT --> merge/split
        TEXT --> merge/split
        TEXT --> merge/split
        TEXT --> merge/split
        HEADING 1.1 --> remove
        TEXT --> merge/split
        TEXT --> merge/split
        HEADING
        TABLECELL
        TABLECELL
        TABLECELL
        TABLECELL
        TABLECELL
        TABLECELL
        TABLECELL
        TABLECELL
        TABLECELL --> do we ignore these under a certain length?
        LISTITEM
        LISTITEM
        LISTITEM
        LISTITEM --> s: detect these. does this exist in metrics code?

        TABLECAPTION --> 1 table

        - dealing with lists: what is the 'sentence' equivalent?
        """

        raise NotImplementedError


class IdentityChunker(BaseChunker):
    """Returns the text blocks converted to chunks, with no attempts at changing them."""

    def __call__(
        self,
        document: BaseDocument,
    ) -> list[Chunk]:
        """Run chunker"""

        if not document.text_blocks:
            return []

        chunks = [
            Chunk(
                text=text_block.to_string(),
                chunk_type=ChunkType(text_block.type),
                bounding_boxes=[text_block.coords] if text_block.coords else None,
                pages=[text_block.page_number],
            )
            for text_block in document.text_blocks
        ]

        return chunks
