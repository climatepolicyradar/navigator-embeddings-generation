import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from cpr_sdk.parser_models import BlockType, ParserOutput, TextBlock

from src.ml import SentenceEncoder

logger = logging.getLogger(__name__)


def replace_text_blocks(block: ParserOutput, new_text_blocks: Sequence[TextBlock]):
    """Updates the text blocks in the ParserOutput object."""
    if block.pdf_data:
        block.pdf_data.text_blocks = new_text_blocks  # type: ignore
    elif block.html_data:
        block.html_data.text_blocks = new_text_blocks  # type: ignore

    return block


def filter_blocks(
    parser_output: ParserOutput, remove_block_types: Sequence[str]
) -> Sequence[TextBlock]:
    """
    Given an ParserOutput filter the contained TextBlocks.

    Return this as a list of TextBlocks.
    """
    filtered_blocks = []
    # TODO: this denotes a bug in the data access library that should be fixed
    for block in parser_output.get_text_blocks(including_invalid_html=True):
        if block.type.title() not in remove_block_types:
            filtered_blocks.append(block)
        else:
            logger.info(
                f"Filtered {block.type} block from {parser_output.document_id}.",
                extra={
                    "props": {
                        "document_id": parser_output.document_id,
                        "block_type": block.type,
                        "remove_block_types": remove_block_types,
                    }
                },
            )
    return filtered_blocks


def filter_on_block_type(
    document: ParserOutput, remove_block_types: List[str]
) -> ParserOutput:
    """Remove the text blocks of the types declared in the remove block types array."""
    for _filter in remove_block_types:
        try:
            BlockType(_filter)
        except NameError:
            logger.warning(
                f"Blocks to filter should be of a known block type, removing {_filter} "
                f"from the list. "
            )
            remove_block_types.remove(_filter)

    return replace_text_blocks(
        block=document,
        new_text_blocks=filter_blocks(
            parser_output=document, remove_block_types=remove_block_types
        ),
    )


def encode_parser_output(
    encoder: SentenceEncoder,
    input_obj: ParserOutput,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Encode a parser output object.

    Produce a numpy array of description embedding and a numpy array of text
    embeddings for a parser output.

    :param encoder: sentence encoder
    :param input_obj: parser output object
    :param batch_size: batch size for encoding text blocks
    :param device: device to use for encoding
    """

    description_embedding = encoder.encode(
        input_obj.document_description, device=device
    )

    text_blocks = input_obj.get_text_blocks()

    if text_blocks:
        text_embeddings = encoder.encode_batch(
            [block.to_string() for block in text_blocks],
            batch_size=batch_size,
            device=device,
        )
    else:
        text_embeddings = None

    return description_embedding, text_embeddings
