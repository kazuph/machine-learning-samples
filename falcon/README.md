# falcon

```
rye sync
rye run ct2-transformers-converter --model tiiuae/falcon-7b-instruct --quantization int8 --output_dir falcon-7b-instruct_ct2 --trust_remote_code
rye run ct2-transformers-converter --model facebook/m2m100_1.2B --output_dir m2m100_1.2B_ct2

rye run python src/falcon_with_m2m2.py
```
