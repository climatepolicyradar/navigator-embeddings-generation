from src.models import Chunk, ChunkType
from src.document_cleaners import RemoveShortTableCells


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
