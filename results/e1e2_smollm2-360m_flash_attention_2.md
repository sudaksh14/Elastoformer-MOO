# E1/E2: smollm2-360m

dtype bfloat16, attention flash_attention_2, seq_len 1024, torch 2.7.0+cu126, transformers 4.51.1, flash_attn 2.8.3, GPU NVIDIA A40

No training: these are the starting points that joint nested training has to beat.

Permutation check (E2): max logit difference 5.131e-04 in float32 (atol 0.0001, rtol 0.0001); 131072 calibration tokens.

## wikitext2_test

| config | KV groups | MLP width | active params | ppl, original order | ppl, importance-ordered |
|---|---|---|---|---|---|
| f025 | 1 | 768 | 133,755,840 | 2.145e+07 | 4489 |
| f050 | 3 | 1280 | 212,399,040 | 4.817e+04 | 406.5 |
| f075 | 4 | 2048 | 298,906,560 | 179.4 | 45.57 |
| f100 | 5 | 2560 | 361,821,120 | 12.96 | 12.97 |

## c4_val

| config | KV groups | MLP width | active params | ppl, original order | ppl, importance-ordered |
|---|---|---|---|---|---|
| f025 | 1 | 768 | 133,755,840 | 1.676e+07 | 4382 |
| f050 | 3 | 1280 | 212,399,040 | 7.072e+04 | 289.9 |
| f075 | 4 | 2048 | 298,906,560 | 223.7 | 45.42 |
| f100 | 5 | 2560 | 361,821,120 | 17.96 | 17.96 |
