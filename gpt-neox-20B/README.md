# gpt-neox

```
rye sync
rye run ct2-transformers-converter --model EleutherAI/gpt-neox-20b --quantization int8 --output_dir gpt_neox_ct2
rye run ct2-transformers-converter --model togethercomputer/GPT-NeoXT-Chat-Base-20B --quantization int8 --output_dir gpt_neox_chat_base_ct2
rye run ct2-transformers-converter --model facebook/m2m100_1.2B --quantization int8 --output_dir m2m100_1.2B_ct2

rye run python src/neox_with_m2m2.py
```
