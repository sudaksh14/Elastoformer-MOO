import pytest


def test_tail_blocks_are_disjoint_from_the_evaluation_prefix(tmp_path):
    import numpy as np
    from elastoformer_moo.data import TokenBlocks

    path = tmp_path / "v.bin"
    np.arange(10 * 8, dtype=np.uint32).tofile(path)  # 10 blocks of 8 tokens
    blocks = TokenBlocks(path, 8)
    head = [b for batch in blocks.sequential(2, max_batches=2) for b in batch]   # blocks 0..3
    tail = [b for batch in blocks.tail(2, 2) for b in batch]                     # blocks 6..9
    assert len(head) == 4 and len(tail) == 4
    assert all(int(h[0]) != int(t[0]) for h in head for t in tail)
    assert int(tail[0][0]) == 6 * 8 and int(tail[-1][0]) == 9 * 8
    with pytest.raises(ValueError, match="do not fit"):
        list(blocks.tail(4, 3))
