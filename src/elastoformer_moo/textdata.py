"""Small stdlib-only helpers for reading evaluation text (no network, no heavy imports)."""

from __future__ import annotations

import gzip
import json
from typing import BinaryIO, Iterator, Optional, Sequence


def wikitext_join(rows: Sequence[str]) -> str:
    """Join WikiText rows the way GPTQ, SparseGPT and Wanda do: ``"\\n\\n".join(rows)``.

    The standard perplexity protocol tokenizes this one string. Rows keep their own newlines.
    """
    return "\n\n".join(rows)


def iter_jsonl_gz_texts(fileobj: BinaryIO, max_docs: Optional[int] = None, field: str = "text") -> Iterator[str]:
    """Yield ``field`` from each line of a gzipped JSONL stream, stopping after ``max_docs``.

    Reads lazily, so only the first part of a very large shard is ever transferred when the
    file object is a remote stream.
    """
    produced = 0
    with gzip.GzipFile(fileobj=fileobj) as handle:
        for raw in handle:
            if max_docs is not None and produced >= max_docs:
                return
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)[field]
            produced += 1
