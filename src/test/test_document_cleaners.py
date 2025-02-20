from src.models import Chunk, ChunkType
from src.document_cleaners import RemoveShortTableCells, RemoveRepeatedAdjacentChunks


def test_remove_short_table_cells_drop_numeric():
    """Test filtering of short and numeric table cells."""
    cleaner = RemoveShortTableCells(min_chars=5, remove_all_numeric=True)
    chunks = [
        Chunk(
            text="short",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="this is long enough",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="123.45",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="1,234.56",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="-123.45",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="not a table cell",
            chunk_type=ChunkType.TEXT,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="12345 with text",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 4
    assert result[0].text == "short"
    assert result[1].text == "this is long enough"
    assert result[2].text == "not a table cell"
    assert result[3].text == "12345 with text"


def test_remove_short_table_cells_keep_numeric():
    """Test keeping numeric cells when remove_all_numeric is False."""
    cleaner = RemoveShortTableCells(min_chars=6, remove_all_numeric=False)

    chunks = [
        Chunk(
            text="short",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="123.45",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="1,234.56",
            chunk_type=ChunkType.TABLE_CELL,
            bounding_boxes=None,
            pages=None,
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 2
    assert result[0].text == "123.45"
    assert result[1].text == "1,234.56"


def test_remove_repeated_adjacent_chunks():
    """Test filtering of repeated chunks of the same type."""
    cleaner = RemoveRepeatedAdjacentChunks()
    chunks = [
        Chunk(
            text="Header",
            chunk_type=ChunkType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="Some content",
            chunk_type=ChunkType.TEXT,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="Header",
            chunk_type=ChunkType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="Different Header",
            chunk_type=ChunkType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="footnote",
            chunk_type=ChunkType.FOOTNOTE,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="important title",
            chunk_type=ChunkType.TITLE,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="footnote",
            chunk_type=ChunkType.FOOTNOTE,
            bounding_boxes=None,
            pages=None,
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 5
    assert result[0].text == "Header"
    assert result[1].text == "Some content"
    assert result[2].text == "Different Header"
    assert result[3].text == "footnote"
    assert result[4].text == "important title"


def test_remove_repeated_adjacent_chunks_case_sensitive():
    """Test case-sensitive filtering of repeated chunks."""
    cleaner = RemoveRepeatedAdjacentChunks(ignore_case=False)
    chunks = [
        Chunk(
            text="Header",
            chunk_type=ChunkType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
        ),
        Chunk(
            text="HEADER",
            chunk_type=ChunkType.PAGE_HEADER,
            bounding_boxes=None,
            pages=None,
        ),
    ]

    result = cleaner(chunks)

    assert len(result) == 2
    assert result[0].text == "Header"
    assert result[1].text == "HEADER"
