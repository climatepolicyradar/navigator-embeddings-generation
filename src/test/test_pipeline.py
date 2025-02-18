from cpr_sdk.parser_models import ParserOutput

from src.pipeline import Pipeline
from src.chunkers import IdentityChunker
from src.document_cleaners import IdentityDocumentCleaner
from src.serializers import BasicSerializer


def test_basic_pipeline(
    test_parser_output_source_url_supported_lang_data: list[ParserOutput],
):
    parser_output: ParserOutput = test_parser_output_source_url_supported_lang_data[0]

    basic_pipeline = Pipeline(
        chunker=IdentityChunker(),
        document_cleaners=[IdentityDocumentCleaner()],
        serializer=BasicSerializer(),
    )
    result = basic_pipeline(parser_output)
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)
    assert len(result) == len(parser_output.text_blocks or [])
