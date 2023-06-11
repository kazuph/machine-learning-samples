import ctranslate2
import transformers
import torch

ppo = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"

# cudaが使える場合はcudaを使う
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = ctranslate2.Generator("rinna_gpt_neox_ppo_ct2", device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(ppo, use_fast=False)


# プロンプトを作成する
def prompt(msg):
    p = [
        {"speaker": "ユーザー", "text": msg},
    ]
    p = [f"{uttr['speaker']}: {uttr['text']}" for uttr in p]
    p = "<NL>".join(p)
    p = p + "<NL>" + "システム: "
    # print(p)
    return p


# 返信を作成する
def reply(msg):
    p = prompt(msg)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(p))

    results = generator.generate_batch(
        [tokens],
        max_length=256,
        sampling_topk=20,
        sampling_temperature=0.9,
    )

    # resultsの中身を確認する
    print(results)

    text = tokenizer.decode(results[0].sequences_ids[0])

    # textは
    # ユーザー: こんにちわ<NL>システム: </s>『東京ディズニーシー』は...
    # と出るが、欲しいのはシステム: 〜 </s>の部分なので、
    # </s>以前を取得する
    # text = text.split("</s>")[0]
    print("システム(ppo-ct2): " + text + "\n")
    return text
