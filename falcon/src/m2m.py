import ctranslate2
import transformers
import torch

# cudaが使える場合はcudaを使う
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

translator = ctranslate2.Translator("m2m100_1.2B_ct2", device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_1.2B")


def en2ja(msg):
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(msg))
    target_prefix = [tokenizer.lang_code_to_token["ja"]]
    results = translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target))


def ja2en(msg):
    source = tokenizer.convert_ids_to_tokens(tokenizer.encode(msg))
    target_prefix = [tokenizer.lang_code_to_token["en"]]
    results = translator.translate_batch([source], target_prefix=[target_prefix])
    target = results[0].hypotheses[0][1:]
    return tokenizer.decode(tokenizer.convert_tokens_to_ids(target))


if __name__ == "__main__":
    print(en2ja("Hello, my name is John."))
    print(ja2en("こんにちは、私の名前はジョンです。"))
