"""CLI to convert JSON documents outputted by the PDF parsing pipeline to embeddings."""

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import NewType, Optional

import click
import numpy as np
from cpr_sdk.parser_models import ParserOutput
from pydantic import BaseModel
from tqdm.auto import tqdm

from src import config
from src.languages import get_docs_of_supported_language
from src.ml import SBERTEncoder
from src.s3 import check_file_exists_in_s3, save_ndarray_to_s3_as_npy, write_json_to_s3
from src.utils import (
    encode_parser_output,
    filter_on_block_type,
    get_Text2EmbeddingsInput_array,
)

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


def process_task(
    task: ParserOutput,
    encoder: SBERTEncoder,
    output_dir: str,
    s3: bool,
    device: str,
) -> None:
    """
    Process a single task to generate embeddings.

    Args:
        task: The parser output task to process
        encoder: The sentence encoder to use
        output_dir: Directory to save outputs to
        s3: Whether to use S3 for I/O
        device: Device to use for encoding
    """
    task_output_path = os.path.join(output_dir, task.document_id + ".json")

    (
        write_json_to_s3(task.model_dump_json(indent=2), task_output_path)
        if s3
        else Path(task_output_path).write_text(task.model_dump_json(indent=2))
    )

    embeddings_output_path = os.path.join(output_dir, task.document_id + ".npy")

    file_exists = (
        check_file_exists_in_s3(embeddings_output_path)
        if s3
        else os.path.exists(embeddings_output_path)
    )
    if file_exists:
        logger.info(
            f"Embeddings output file '{embeddings_output_path}' already exists, "
            "skipping processing."
        )
        return

    description_embedding, text_embeddings = encode_parser_output(
        encoder, task, config.ENCODING_BATCH_SIZE, device=device
    )

    combined_embeddings = (
        np.vstack([description_embedding, text_embeddings])
        if text_embeddings is not None
        else description_embedding.reshape(1, -1)
    )

    (
        save_ndarray_to_s3_as_npy(combined_embeddings, embeddings_output_path)
        if s3
        else np.save(embeddings_output_path, combined_embeddings)
    )


def write_results_file(
    results: list[EmbeddingResult], output_dir: str, s3: bool
) -> None:
    """
    Write results to a JSON file.

    Args:
        results: List of embedding results
        output_dir: Directory to write results file to
        s3: Whether to write to S3
    """
    results_path = os.path.join(output_dir, "embeddings_results.json")
    results_json = json.dumps(
        [result.model_dump() for result in results],
        indent=2,
    )

    if s3:
        write_json_to_s3(results_json, results_path)
    else:
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
@click.argument(
    "input-dir",
)
@click.argument(
    "output-dir",
)
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
    input_dir: str,
    output_dir: str,
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

    Args: input_dir: Directory containing JSON files output_dir: Directory to save
    embeddings to s3: Whether we are reading from and writing to S3.
    document_import_ids: A list of the document import ids to run on. device (str):
    Device to use for embeddings generation. Must be either "cuda", "mps", or "cpu".
    """

    return run_embeddings_generation(
        input_dir=input_dir,
        output_dir=output_dir,
        document_import_ids=document_import_ids,
        s3=s3,
        device=device,
    )


def run_embeddings_generation(
    input_dir: str,
    output_dir: str,
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
                "input_dir": input_dir,
                "output_dir": output_dir,
                "document_import_ids": document_import_ids,
                "s3": s3,
                "device": device,
            }
        },
    )

    logger.info("Constructing Text2EmbeddingsInput objects from parser output jsons.")
    tasks = get_Text2EmbeddingsInput_array(input_dir, s3, document_import_ids)

    logger.info(
        "Filtering tasks to those with supported languages.",
        extra={"props": {"target_languages": config.TARGET_LANGUAGES}},
    )
    tasks = get_docs_of_supported_language(tasks)
    logger.info(
        f"Found {len(tasks)} tasks with supported languages.",
        extra={
            "props": {
                "tasks": [
                    {
                        "lang": task.languages,
                        "translated": task.translated,
                        "document_id": task.document_id,
                    }
                    for task in tasks
                ]
            }
        },
    )

    logger.info(
        "Filtering unwanted text block types.",
        extra={"props": {"BLOCKS_TO_FILTER": config.BLOCKS_TO_FILTER}},
    )
    tasks = filter_on_block_type(
        inputs=tasks, remove_block_types=config.BLOCKS_TO_FILTER
    )

    logger.info(f"Loading sentence-transformer model {config.SBERT_MODEL}")
    encoder = SBERTEncoder(config.SBERT_MODEL)

    logger.info(
        "Encoding text from documents.",
        extra={
            "props": {
                "ENCODING_BATCH_SIZE": config.ENCODING_BATCH_SIZE,
                "tasks_number": len(tasks),
            }
        },
    )

    results: list[EmbeddingResult] = []

    for task in tqdm(tasks, unit="docs"):
        result = EmbeddingResult(document_id=task.document_id)

        try:
            process_task(task, encoder, output_dir, s3, device)
        except Exception as e:
            msg = f"Processing document {task.document_id} failed: {e}"
            logger.exception(msg, extra={"props": {"document_id": task.document_id}})
            result.error = f"{type(e).__name__}: {str(e)}"

        results.append(result)

    logger.info("Done processing documents.")

    # Write out results
    write_results_file(results, output_dir, s3)


if __name__ == "__main__":
    run_as_cli()
