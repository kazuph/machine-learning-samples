import ctranslate2
import transformers
import torch

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
device = "cpu"

src_lang = "eng_Latn"
tgt_lang = "jpn_Jpan"

translator = ctranslate2.Translator("nllb-200-1.3B-ct2", device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "facebook/nllb-200-1.3B", src_lang=src_lang
)

source = tokenizer.convert_ids_to_tokens(tokenizer.encode("Hello world!"))
target_prefix = [tgt_lang]
results = translator.translate_batch([source], target_prefix=[target_prefix])
target = results[0].hypotheses[0][1:]

print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
