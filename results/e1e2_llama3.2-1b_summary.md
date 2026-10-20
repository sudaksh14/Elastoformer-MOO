# E1/E2: llama3.2-1b

dtype bfloat16, attention flash_attention_2, seq_len 1024, torch 2.7.0+cu126, transformers 4.51.1, flash_attn 2.8.3, GPU NVIDIA A40

No training: these are the starting points that joint nested training has to beat.

Permutation check (E2): max logit difference 2.012e-04 in float32 (atol 0.0001, rtol 0.0001); 131072 calibration tokens.

## wikitext2_test

| config | KV groups | MLP width | active params | ppl, original order | ppl, importance-ordered |
|---|---|---|---|---|---|
| f025 | 2 | 2048 | 506,005,504 | 2.729e+06 | 3673 |
| f050 | 4 | 4096 | 749,275,136 | 1.582e+05 | 689.6 |
| f075 | 6 | 6144 | 992,544,768 | 8780 | 52.57 |
| f100 | 8 | 8192 | 1,235,814,400 | 11 | 10.99 |

## c4_val

| config | KV groups | MLP width | active params | ppl, original order | ppl, importance-ordered |
|---|---|---|---|---|---|
| f025 | 2 | 2048 | 506,005,504 | 1.652e+06 | 2054 |
| f050 | 4 | 4096 | 749,275,136 | 1.889e+05 | 407.5 |
| f075 | 6 | 6144 | 992,544,768 | 6904 | 44.84 |
| f100 | 8 | 8192 | 1,235,814,400 | 14.7 | 14.7 |
