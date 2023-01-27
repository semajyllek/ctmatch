#!/usr/bin/env bash

set -aeoux
# or wherever your directory of JSONL files ended up
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input data/nct004x_jsonl \
 -index indexes/nct004x -storePositions -storeDocvectors -storeRaw # or wherever you want the index to be stored
