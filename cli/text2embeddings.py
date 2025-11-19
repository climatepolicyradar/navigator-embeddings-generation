"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import NewType, Optional
from uuid import uuid4

import click
import numpy as np
from cpr_sdk.parser_models import ParserOutput
from pydantic import BaseModel
from tqdm.auto import tqdm

from src import config
from src.languages import doc_has_supported_language
from src.ml import SBERTEncoder
from src.s3 import s3_object_read_text, save_ndarray_to_s3_as_npy, write_json_to_s3
from src.utils import encode_parser_output, filter_on_block_type

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
            "formatter": "json",
        },
    },
    "loggers": {},
    "root": {
        "handlers": ["console"],
        "level": LOG_LEVEL,
    },
    "formatters": {"json": {"()": "pythonjsonlogger.jsonlogger.JsonFormatter"}},
}

logger = logging.getLogger(__name__)
logging.config.dictConfig(DEFAULT_LOGGING)

# Example: CCLW.executive.1813.2418
DocumentImportId = NewType("DocumentImportId", str)


class EmbeddingResult(BaseModel):
    """Result of processing a document for embeddings generation."""

    document_id: str
    error: Optional[str] = None


def process_document(
    document_id: DocumentImportId,
    encoder: SBERTEncoder,
    input_dir: str,
    output_dir: str,
    s3: bool,
    device: str,
) -> EmbeddingResult:
    """
    Process a single document end-to-end: load, validate, filter, encode, and save.

    This function handles all steps for a single document:
    1. Load document from S3 or local filesystem
    2. Check if language is supported
    3. Filter unwanted text block types
    4. Generate embeddings
    5. Save embeddings and document JSON

    Args:
        document_id: The document import ID to process
        encoder: The sentence encoder to use
        input_dir: Directory containing input JSON files
        output_dir: Directory to save outputs to
        s3: Whether to use S3 for I/O
        device: Device to use for encoding

    Returns:
        EmbeddingResult with document_id and optional error message
    """
    result = EmbeddingResult(document_id=document_id)

    try:
        # Step 1: Load document
        file_path = os.path.join(input_dir, document_id + ".json")
        json_content = (
            s3_object_read_text(file_path) if s3 else Path(file_path).read_text()
        )
        parser_output = ParserOutput.model_validate_json(json_content)

        # Step 2: Check if language is supported
        if not doc_has_supported_language(parser_output):
            result.error = "Filtered out: unsupported language"
            return result

        # Step 3: Filter unwanted text block types
        parser_output: ParserOutput = filter_on_block_type(
            document=parser_output, remove_block_types=config.BLOCKS_TO_FILTER
        )

        # Step 4: Generate embeddings
        description_embedding, text_embeddings = encode_parser_output(
            encoder, parser_output, config.ENCODING_BATCH_SIZE, device=device
        )

        combined_embeddings = (
            np.vstack([description_embedding, text_embeddings])
            if text_embeddings is not None
            else description_embedding.reshape(1, -1)
        )

        # Step 5: Save embeddings
        embeddings_output_path_npy = os.path.join(
            output_dir, parser_output.document_id + ".npy"
        )
        if not s3:
            Path(embeddings_output_path_npy).parent.mkdir(parents=True, exist_ok=True)
        (
            save_ndarray_to_s3_as_npy(combined_embeddings, embeddings_output_path_npy)
            if s3
            else np.save(embeddings_output_path_npy, combined_embeddings)
        )

        # Step 6: Save document JSON
        embeddings_output_path_json = os.path.join(
            output_dir, parser_output.document_id + ".json"
        )
        if not s3:
            Path(embeddings_output_path_json).parent.mkdir(parents=True, exist_ok=True)
        (
            write_json_to_s3(
                parser_output.model_dump_json(indent=2), embeddings_output_path_json
            )
            if s3
            else Path(embeddings_output_path_json).write_text(
                parser_output.model_dump_json(indent=2)
            )
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception(
            f"Processing document {document_id} failed",
            extra={"props": {"document_id": document_id}},
        )
        result.error = error_msg

    return result


def write_results_file(
    results: list[EmbeddingResult], input_dir_path: str, s3: bool
) -> None:
    """
    Write results to a JSON file.

    Args:
        results: List of embedding results.
        input_dir_path: Directory to write results file to that is specific to the run.
        s3: Whether to write to S3.
    """
    results_path = os.path.join(
        input_dir_path, "reports", "embeddings", uuid4().hex + ".json"
    )

    results_json = json.dumps(
        [result.model_dump() for result in results],
        indent=2,
    )

    if s3:
        write_json_to_s3(results_json, results_path)
    else:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        Path(results_path).write_text(results_json)


class CommaSeparatedList(click.ParamType):
    """A Custom ParamType allowing comma separated lists to be pass to the cli."""

    name = "comma_separated_list"

    def convert(self, value, param, ctx):
        """Convert the value passed in to the cli to the desired format."""
        if value is None:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]


@click.command()
@click.argument("input-dir-path")
@click.argument("embeddings-input-dir-path")
@click.argument("embeddings-output-dir-path")
@click.argument("document-import-ids", type=CommaSeparatedList())
@click.option(
    "--s3",
    is_flag=True,
    required=False,
    help="Whether or not we are reading from and writing to S3.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "mps", "cpu"]),
    help="Device to use for embeddings generation",
    required=True,
    default="cpu",
)
def run_as_cli(
    input_dir_path: str,
    embeddings_input_dir_path: str,
    embeddings_output_dir_path: str,
    document_import_ids: list[DocumentImportId],
    s3: bool,
    device: str,
):
    """
    Run CLI to produce embeddings from document parser JSON outputs.

    Each embeddings file is called {id}.json where {id} is the document ID of the
    input. Its first line is the description embedding and all other lines are
    embeddings of each of the text blocks in the document in order. Encoding will
    run CPU unless device is set to 'cuda' or 'mps'.

    Args:
        input_dir_path: Directory containing the input directory for the run.
        embeddings_input_dir_path: Directory containing JSON files to process for embeddings
            generation.
        embeddings_output_dir_path: Directory to save embeddings to.
        document_import_ids: A list of the document import ids to run on.
        s3: Whether we are reading from and writing to S3.
        device: Device to use for embeddings generation.
            Must be either "cuda", "mps", or "cpu".
    """

    return run_embeddings_generation(
        input_dir_path=input_dir_path,
        embeddings_input_dir_path=embeddings_input_dir_path,
        embeddings_output_dir_path=embeddings_output_dir_path,
        document_import_ids=document_import_ids,
        s3=s3,
        device=device,
    )


def run_embeddings_generation(
    input_dir_path: str,
    embeddings_input_dir_path: str,
    embeddings_output_dir_path: str,
    document_import_ids: list[DocumentImportId],
    s3: bool,
    device: str,
):
    """
    Run CLI to produce embeddings from document parser JSON outputs.

    See docstring for run_as_cli for details.
    """

    logger.info(
        "Running embeddings generation...",
        extra={
            "props": {
                "input_dir_path": input_dir_path,
                "embeddings_input_dir_path": embeddings_input_dir_path,
                "embeddings_output_dir_path": embeddings_output_dir_path,
                "document_import_ids": document_import_ids,
                "s3": s3,
                "device": device,
            }
        },
    )

    logger.info(f"Loading sentence-transformer model {config.SBERT_MODEL}")
    encoder = SBERTEncoder(config.SBERT_MODEL)

    logger.info(
        "Processing documents.",
        extra={
            "props": {
                "ENCODING_BATCH_SIZE": config.ENCODING_BATCH_SIZE,
                "total_documents": len(document_import_ids),
                "target_languages": config.TARGET_LANGUAGES,
                "blocks_to_filter": config.BLOCKS_TO_FILTER,
            }
        },
    )

    # Process each document independently
    results: list[EmbeddingResult] = []

    for document_id in tqdm(document_import_ids, unit="docs"):
        result = process_document(
            document_id=document_id,
            encoder=encoder,
            input_dir=embeddings_input_dir_path,
            output_dir=embeddings_output_dir_path,
            s3=s3,
            device=device,
        )
        results.append(result)

    # Log summary statistics
    successful = sum(1 for r in results if r.error is None)
    failed = len(results) - successful

    logger.info(
        "Done processing documents.",
        extra={
            "props": {
                "total": len(results),
                "successful": successful,
                "failed": failed,
            }
        },
    )

    # Write out results
    write_results_file(results, input_dir_path, s3)


if __name__ == "__main__":
    run_as_cli()
