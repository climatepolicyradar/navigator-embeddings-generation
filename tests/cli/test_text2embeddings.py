import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from click.testing import CliRunner
from cpr_sdk.parser_models import ParserOutput

from cli.text2embeddings import (
    EmbeddingResult,
    process_document,
    run_as_cli,
    write_results_file,
)
from src.s3 import s3_object_read_text


def test_run_encoder_local(
    test_html_file_json,
    test_pdf_file_json,
    test_no_content_type_file_json,
):
    """Test that the encoder runs with local input and output paths and outputs the correct files."""

    with tempfile.TemporaryDirectory() as embeddings_input_dir_path:
        with tempfile.TemporaryDirectory() as embeddings_output_dir_path:
            with tempfile.TemporaryDirectory() as input_dir_path:
                document_import_ids = []

                # Create test files
                for file in [
                    test_html_file_json,
                    test_pdf_file_json,
                    test_no_content_type_file_json,
                ]:
                    file_path = (
                        Path(embeddings_input_dir_path) / f"{file['document_id']}.json"
                    )
                    file_path.write_text(json.dumps(file))

                    document_import_ids.append(file["document_id"])

                runner = CliRunner()
                result = runner.invoke(
                    run_as_cli,
                    [
                        input_dir_path,
                        embeddings_input_dir_path,
                        embeddings_output_dir_path,
                        ",".join(document_import_ids),
                    ],
                )
                assert result.exit_code == 0

                assert set(Path(embeddings_output_dir_path).glob("*.json")) == {
                    Path(embeddings_output_dir_path) / "test_html.json",
                    Path(embeddings_output_dir_path) / "test_pdf.json",
                    Path(embeddings_output_dir_path) / "test_no_content_type.json",
                }
                assert set(Path(embeddings_output_dir_path).glob("*.npy")) == {
                    Path(embeddings_output_dir_path) / "test_html.npy",
                    Path(embeddings_output_dir_path) / "test_pdf.npy",
                    Path(embeddings_output_dir_path) / "test_no_content_type.npy",
                }

                for path in Path(embeddings_output_dir_path).glob("*.json"):
                    assert ParserOutput.model_validate(json.loads(path.read_text()))

                for path in Path(embeddings_output_dir_path).glob("*.npy"):
                    assert np.load(str(path)).shape[1] == 768

                # test_html has the `has_valid_text` flag set to false, so the numpy file
                # should only contain a description embedding
                assert np.load(
                    str(Path(embeddings_output_dir_path) / "test_html.npy")
                ).shape == (1, 768)

                # Validate that results file is produced
                results_dir = Path(input_dir_path) / "reports" / "embeddings"
                assert results_dir.exists()

                # Get the results file (should be only one .json file)
                results_files = list(results_dir.glob("*.json"))
                assert len(results_files) == 1

                results_file = results_files[0]

                # Validate results file contents
                file_content = json.loads(results_file.read_text())
                assert len(file_content) == 3

                # Validate that all document IDs are present in results
                result_document_ids = {result["document_id"] for result in file_content}
                assert result_document_ids == set(document_import_ids)

                # Validate that results have the expected structure
                for result in file_content:
                    assert "document_id" in result
                    assert "error" in result


def test_run_embeddings_on_translated(
    test_html_file_json,
    test_pdf_file_json,
    test_no_content_type_file_json,
):
    """Test that the embeddings generation cli can handle translated documents."""

    with tempfile.TemporaryDirectory() as embeddings_input_dir_path:
        with tempfile.TemporaryDirectory() as embeddings_output_dir_path:
            with tempfile.TemporaryDirectory() as input_dir_path:
                # Create HTML File
                html_file_path = (
                    Path(embeddings_input_dir_path)
                    / f"{test_html_file_json['document_id']}.json"
                )
                html_file_path.write_text(json.dumps(test_html_file_json))

                # Create PDF File
                pdf_file_path = (
                    Path(embeddings_input_dir_path)
                    / f"{test_pdf_file_json['document_id']}.json"
                )
                pdf_file_path.write_text(json.dumps(test_pdf_file_json))

                # Create No Content Type File (both non/translated)
                test_no_content_type_file_json["translated"] = True
                no_content_type_translated_file_path = (
                    Path(embeddings_input_dir_path)
                    / f"{test_no_content_type_file_json['document_id']}_translated_en.json"
                )
                no_content_type_translated_file_path.write_text(
                    json.dumps(test_no_content_type_file_json)
                )

                test_no_content_type_file_json["translated"] = False
                test_no_content_type_file_json["languages"] = ["fr"]

                no_content_type_file_path = (
                    Path(embeddings_input_dir_path)
                    / f"{test_no_content_type_file_json['document_id']}.json"
                )
                no_content_type_file_path.write_text(
                    json.dumps(test_no_content_type_file_json)
                )

                document_import_ids = [
                    test_html_file_json["document_id"],
                    test_pdf_file_json["document_id"],
                    test_no_content_type_file_json["document_id"],
                ]

                runner = CliRunner()
                result = runner.invoke(
                    run_as_cli,
                    [
                        input_dir_path,
                        embeddings_input_dir_path,
                        embeddings_output_dir_path,
                        ",".join(document_import_ids),
                    ],
                )
                assert result.exit_code == 0

                assert set(Path(embeddings_output_dir_path).glob("*.json")) == {
                    Path(embeddings_output_dir_path) / "test_html.json",
                    Path(embeddings_output_dir_path) / "test_pdf.json",
                    Path(embeddings_output_dir_path) / "test_no_content_type.json",
                }
                assert set(Path(embeddings_output_dir_path).glob("*.npy")) == {
                    Path(embeddings_output_dir_path) / "test_html.npy",
                    Path(embeddings_output_dir_path) / "test_pdf.npy",
                    Path(embeddings_output_dir_path) / "test_no_content_type.npy",
                }


def test_s3_client(
    s3_bucket_and_region,
    pipeline_s3_objects_main,
    pipeline_s3_client_main,
    input_prefix,
):
    """Prior to running the embeddings generation tests assert that the mock s3 bucket is in the required state."""
    list_response = pipeline_s3_client_main.client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"], Prefix=input_prefix
    )
    assert list_response["KeyCount"] == len(pipeline_s3_objects_main)


def test_run_encoder_s3(
    s3_bucket_and_region,
    pipeline_s3_objects_main,
    pipeline_s3_client_main,
    test_input_dir_s3,
    test_output_dir_s3,
    output_prefix,
):
    """Test that the encoder runs with S3 input and output paths and outputs the correct files."""

    document_import_ids = [
        "CCLWTEST.executive.1000.1000",
        "CCLWTEST.executive.1001.1001",
        "CCLWTEST.executive.1002.1002",
    ]

    input_dir_path: str = (
        f's3://{s3_bucket_and_region["bucket"]}/input/2023-04-13T09.17.37.953810/'
    )

    runner = CliRunner()
    result = runner.invoke(
        run_as_cli,
        [
            input_dir_path,
            test_input_dir_s3,
            test_output_dir_s3,
            ",".join(document_import_ids),
            "--s3",
        ],
    )

    assert result.exit_code == 0

    list_response = pipeline_s3_client_main.client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"], Prefix=output_prefix
    )

    assert list_response["KeyCount"] == len(pipeline_s3_objects_main) * 2

    files = [
        o["Key"] for o in list_response.get("Contents", []) if o["Key"] != output_prefix
    ]

    assert len(files) == len(pipeline_s3_objects_main) * 2

    assert set(files) == {
        f"{output_prefix}/test_html.json",
        f"{output_prefix}/test_html.npy",
        f"{output_prefix}/test_no_content_type.json",
        f"{output_prefix}/test_no_content_type.npy",
        f"{output_prefix}/test_pdf.json",
        f"{output_prefix}/test_pdf.npy",
    }

    s3_files_json = [file for file in files if file.endswith(".json")]
    s3_files_npy = [file for file in files if file.endswith(".npy")]

    assert len(s3_files_json) == len(pipeline_s3_objects_main)
    assert len(s3_files_npy) == len(pipeline_s3_objects_main)

    for file in s3_files_json:
        file_obj = pipeline_s3_client_main.client.get_object(
            Bucket=s3_bucket_and_region["bucket"], Key=file
        )
        file_text = file_obj["Body"].read().decode("utf-8")
        file_json = json.loads(file_text)
        assert ParserOutput.model_validate(file_json)

    for file in s3_files_npy:
        file_obj = pipeline_s3_client_main.client.get_object(
            Bucket=s3_bucket_and_region["bucket"], Key=file
        )
        file_text = file_obj["Body"]
        file_bytes = io.BytesIO(file_text.read())
        assert np.load(file_bytes).shape[1] == 768

    # Validate that results file is produced in S3
    results_prefix = "input/2023-04-13T09.17.37.953810/reports/embeddings/"
    results_list_response = pipeline_s3_client_main.client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"], Prefix=results_prefix
    )

    # Should have one results file
    assert results_list_response["KeyCount"] == 1

    # Get the results file key
    results_key = results_list_response["Contents"][0]["Key"]

    # Read and validate the results file
    results_path = f"s3://{s3_bucket_and_region['bucket']}/{results_key}"
    file_content = json.loads(s3_object_read_text(results_path))

    assert len(file_content) == len(document_import_ids)

    # Validate that all document IDs are present in results
    result_document_ids = {result["document_id"] for result in file_content}
    assert result_document_ids == set(document_import_ids)

    # Validate that results have the expected structure
    for result in file_content:
        assert "document_id" in result
        assert "error" in result


def test_process_document_success(test_pdf_file_json) -> None:
    """Test that process_document successfully processes a document and collects results."""
    document_id = "test_doc_123"

    with tempfile.TemporaryDirectory() as input_dir:
        with tempfile.TemporaryDirectory() as output_dir:
            # Create input JSON file
            input_file = Path(input_dir) / f"{document_id}.json"
            input_file.write_text(json.dumps(test_pdf_file_json))

            # Mock encoder
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = np.array([0.1] * 768)
            mock_encoder.encode_batch.return_value = np.array(
                [[0.2] * 768, [0.3] * 768]
            )

            # Process document
            result = process_document(
                document_id=document_id,
                encoder=mock_encoder,
                input_dir=input_dir,
                output_dir=output_dir,
                s3=False,
                device="cpu",
            )

            # Validate result collection
            assert isinstance(result, EmbeddingResult)
            assert result.document_id == document_id
            assert result.error is None

            # Validate output files were created
            output_json = Path(output_dir) / f"{test_pdf_file_json['document_id']}.json"
            output_npy = Path(output_dir) / f"{test_pdf_file_json['document_id']}.npy"

            assert output_json.exists()
            assert output_npy.exists()

            # Validate JSON output
            output_data = json.loads(output_json.read_text())
            assert ParserOutput.model_validate(output_data)

            # Validate embeddings output
            embeddings = np.load(str(output_npy))
            assert embeddings.shape[1] == 768


def test_process_document_error_handling() -> None:
    """Test that process_document catches errors and returns them in the result."""
    document_id = "test_doc_456"

    with tempfile.TemporaryDirectory() as input_dir:
        with tempfile.TemporaryDirectory() as output_dir:
            # Create input JSON file with invalid content to trigger an error
            input_file = Path(input_dir) / f"{document_id}.json"
            input_file.write_text("invalid json content")

            # Mock encoder
            mock_encoder = MagicMock()

            # Process document - should catch the JSON parsing error
            result = process_document(
                document_id=document_id,
                encoder=mock_encoder,
                input_dir=input_dir,
                output_dir=output_dir,
                s3=False,
                device="cpu",
            )

            # Validate error was caught and returned
            assert isinstance(result, EmbeddingResult)
            assert result.document_id == document_id
            assert result.error is not None
            assert (
                "ValidationError" in result.error
                or "JSONDecodeError" in result.error
                or "ValueError" in result.error
            )


def test_write_results_file_local() -> None:
    """Test that write_results_file writes results to local filesystem correctly."""
    results = [
        EmbeddingResult(document_id="doc1", error=None),
        EmbeddingResult(document_id="doc2", error="SomeError: error message"),
        EmbeddingResult(document_id="doc3", error=None),
    ]

    with tempfile.TemporaryDirectory() as input_dir:
        # Write results to local filesystem
        write_results_file(results=results, input_dir_path=input_dir, s3=False)

        # Find the results file (it has a random UUID filename)
        results_dir = Path(input_dir) / "reports" / "embeddings"
        assert results_dir.exists()

        # Get the results file (should be only one .json file)
        results_files = list(results_dir.glob("*.json"))
        assert len(results_files) == 1

        results_file = results_files[0]

        # Validate file contents
        file_content = json.loads(results_file.read_text())
        assert len(file_content) == 3

        # Validate each result
        assert file_content[0]["document_id"] == "doc1"
        assert file_content[0]["error"] is None

        assert file_content[1]["document_id"] == "doc2"
        assert file_content[1]["error"] == "SomeError: error message"

        assert file_content[2]["document_id"] == "doc3"
        assert file_content[2]["error"] is None


def test_write_results_file_s3(s3_bucket_and_region, pipeline_s3_client_main) -> None:
    """Test that write_results_file writes results to S3 correctly."""
    results = [
        EmbeddingResult(document_id="doc1", error=None),
        EmbeddingResult(document_id="doc2", error="SomeError: error message"),
    ]

    input_dir_path = f's3://{s3_bucket_and_region["bucket"]}/input/test_run/'

    # Write results to S3
    write_results_file(results=results, input_dir_path=input_dir_path, s3=True)

    # List objects in the results directory
    list_response = pipeline_s3_client_main.client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"],
        Prefix="input/test_run/reports/embeddings/",
    )

    # Should have one results file
    assert list_response["KeyCount"] == 1

    # Get the results file key
    results_key = list_response["Contents"][0]["Key"]

    # Read and validate the results file
    results_path = f"s3://{s3_bucket_and_region['bucket']}/{results_key}"
    file_content = json.loads(s3_object_read_text(results_path))

    assert len(file_content) == 2

    # Validate each result
    assert file_content[0]["document_id"] == "doc1"
    assert file_content[0]["error"] is None

    assert file_content[1]["document_id"] == "doc2"
    assert file_content[1]["error"] == "SomeError: error message"
