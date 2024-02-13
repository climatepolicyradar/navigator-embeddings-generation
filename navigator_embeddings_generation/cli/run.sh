#!/bin/bash
set -e

python -m navigator_embeddings_generation.cli.text2embeddings --s3 --device=cpu "${EMBEDDINGS_INPUT_PREFIX}" "${INDEXER_INPUT_PREFIX}"