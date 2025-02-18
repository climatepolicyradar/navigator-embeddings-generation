from typing import Optional, Sequence

import numpy as np
from cpr_sdk.parser_models import ParserOutput

from src.models import Chunk
from src.chunkers import BaseChunker
from src.document_cleaners import BaseDocumentCleaner
from src.serializers import BaseSerializer
from src.encoders import BaseEncoder


class Pipeline:
    """Pipeline for document processing."""

    def __init__(
        self,
        chunker: BaseChunker,
        document_cleaners: Sequence[BaseDocumentCleaner],
        serializer: BaseSerializer,
        encoder: Optional[BaseEncoder] = None,
    ) -> None:
        self.chunker = chunker
        self.document_cleaners = document_cleaners
        self.serializer = serializer
        self.encoder = encoder
        # self._validate_pipeline()

        self.pipeline_return_type = list[str] if self.encoder is None else np.ndarray

    def _validate_pipeline(self) -> None:
        """
        Check that the pipeline components are valid.

        Raises a ValueError if not.
        """

        component_type_map = {
            "chunker": BaseChunker,
            "document_cleaners": Sequence[BaseDocumentCleaner],
            "serializer": BaseSerializer,
            "encoder": Optional[BaseEncoder],
        }

        incorrect_typed_components = [
            k for k, v in component_type_map.items() if not isinstance(k, v)
        ]

        if incorrect_typed_components:
            incorrect_typed_components_msg = ",".join(
                f"{k} ({component_type_map[k]})" for k in incorrect_typed_components
            )
            raise ValueError(
                f"The following components have incorrect types: {incorrect_typed_components_msg}"
            )

    def __call__(
        self,
        document: ParserOutput,
        encoder_batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> list[str] | np.ndarray:
        """Run the pipeline on a single document."""

        if self.encoder is not None and encoder_batch_size is None:
            raise ValueError(
                "This pipeline contains an encoder but no batch size was set. Please set a batch size."
            )

        chunks: Sequence[Chunk] = self.chunker(document)

        for cleaner in self.document_cleaners:
            chunks = cleaner(chunks)

        serialized_text: list[str] = [self.serializer(chunk) for chunk in chunks]

        if self.encoder is None:
            return serialized_text
        else:
            return self.encoder.encode_batch(
                text_batch=serialized_text,
                batch_size=encoder_batch_size,  # type: ignore
                device=device,
            )
