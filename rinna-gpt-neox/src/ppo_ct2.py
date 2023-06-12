import ctranslate2
import transformers
import torch

ppo = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"

# cudaが使える場合はcudaを使う
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# generator = ctranslate2.Generator("rinna_gpt_neox_ppo_ct2_ft16", device=device)
generator = ctranslate2.Generator("rinna_gpt_neox_ppo_ct2_int8", device=device)
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
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(
            p,
            add_special_tokens=False,
        )
    )

    results = generator.generate_batch(
        [tokens],
        max_length=256,
        sampling_topk=10,
        sampling_temperature=0.7,
        include_prompt_in_result=False,
    )

    text = tokenizer.decode(results[0].sequences_ids[0])
    print("システム(ppo-ct2): " + text + "\n")
    return text


# mainとして実行された場合は、ユーザーからの入力を受け付ける
if __name__ == "__main__":
    import readline
    import time

    questions = [
        "こんにちは",
        "日本の首相は？",
        "日本の首都は？",
        "アメリカの首都は？",
        "アメリカの大統領は？",
        "日本に大統領はいますか？",
    ]
    while True:
        # 実行時間を計測する
        start = time.time()
        print("=========================================")
        if len(questions) > 0:
            q = questions.pop(0)
            print(f"ユーザー: {q}\n")
            reply(q)
        else:
            msg = input("ユーザー: ")
            reply(msg)

        print(f"実行時間: {time.time() - start:.2f}秒")
