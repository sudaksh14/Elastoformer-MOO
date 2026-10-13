"""Pre-tokenized token files and block sampling.

Tokens are stored as a flat ``uint32`` array (vocabularies above 65k do not fit in
``uint16``). Blocks are drawn at random offsets with a seeded generator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


def _chunks(items: Iterable[str], size: int):
    chunk: list[str] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_token_file(
    texts: Iterable[str],
    tokenizer,
    out_path: str | Path,
    append_eos: bool = True,
    max_tokens: int | None = None,
    batch_size: int = 1,
) -> int:
    """Tokenize ``texts`` and write them to ``out_path``. Returns the number of tokens written.

    With ``max_tokens`` the file is cut to exactly that many tokens and reading stops, so a huge
    or streamed corpus is only consumed as far as needed. ``batch_size > 1`` tokenizes that many
    documents per call, which is much faster with a fast (Rust) tokenizer.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    eos = tokenizer.eos_token_id
    with open(out_path, "wb") as handle:
        for chunk in _chunks(texts, max(1, batch_size)):
            if batch_size > 1:
                encoded = tokenizer(chunk, add_special_tokens=False)["input_ids"]
            else:
                encoded = [tokenizer(chunk[0], add_special_tokens=False)["input_ids"]]
            for ids in encoded:
                if append_eos and eos is not None:
                    ids = ids + [eos]
                if max_tokens is not None and total + len(ids) > max_tokens:
                    ids = ids[: max_tokens - total]
                np.asarray(ids, dtype=np.uint32).tofile(handle)
                total += len(ids)
                if max_tokens is not None and total >= max_tokens:
                    return total
    return total


class TokenBlocks:
    """Random fixed-length blocks from a token file."""

    def __init__(self, path: str | Path, seq_len: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"token file not found: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint32, mode="r")
        self.seq_len = seq_len
        if len(self.tokens) <= seq_len:
            raise ValueError(f"token file has {len(self.tokens)} tokens, need more than seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.tokens) // self.seq_len

    def block(self, index: int) -> torch.Tensor:
        start = index * self.seq_len
        return torch.from_numpy(self.tokens[start : start + self.seq_len].astype(np.int64))

    def sequential(self, batch_size: int, max_batches: int | None = None) -> Iterator[torch.Tensor]:
        """Non-overlapping blocks in file order; used for evaluation."""
        n_blocks, produced = len(self), 0
        for first in range(0, n_blocks, batch_size):
            if max_batches is not None and produced >= max_batches:
                return
            idx = range(first, min(first + batch_size, n_blocks))
            yield torch.stack([self.block(i) for i in idx])
            produced += 1

    def epoch_stream(
        self, batch_size: int, seed: int = 0, rank: int = 0, world: int = 1, skip_batches: int = 0
    ) -> Iterator[torch.Tensor]:
        """Shuffled passes over the file: every block exactly once per epoch, no repeats within an epoch.

        The shuffle depends only on ``(seed, epoch)``, and rank ``r`` of ``world`` takes every ``world``-th block
        of it, so ranks never read the same block in an epoch. ``skip_batches`` fast-forwards without reading
        data, so a resumed run continues exactly where the interrupted one stopped.
        """
        n_blocks, epoch, skip = len(self), 0, skip_batches
        while True:
            order = np.random.default_rng([seed, epoch]).permutation(n_blocks)[rank::world]
            per_epoch = len(order) // batch_size
            if per_epoch == 0:
                raise ValueError(f"{n_blocks} blocks is too few for batch_size={batch_size} on {world} ranks")
            first = min(skip, per_epoch)
            skip -= first
            for b in range(first, per_epoch):
                chosen = order[b * batch_size : (b + 1) * batch_size]
                yield torch.stack([self.block(int(i)) for i in chosen])
            epoch += 1

    def random(self, batch_size: int, seed: int = 0) -> Iterator[torch.Tensor]:
        """Infinite stream of blocks at random offsets."""
        rng = np.random.default_rng(seed)
        high = len(self.tokens) - self.seq_len
        while True:
            starts = rng.integers(0, high, size=batch_size)
            yield torch.stack(
                [torch.from_numpy(self.tokens[s : s + self.seq_len].astype(np.int64)) for s in starts]
            )
