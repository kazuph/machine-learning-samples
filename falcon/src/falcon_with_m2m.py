import ctranslate2
import transformers
import torch
import m2m

# cudaが使える場合はcudaを使う
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = ctranslate2.Generator("falcon-7b-instruct_ct2", device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")


def reply(msg):
    msg = m2m.ja2en(msg)
    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(
            msg,
            add_special_tokens=False,
        )
    )
    results = generator.generate_batch(
        [tokens], max_length=256, sampling_topk=10, include_prompt_in_result=False
    )

    text = tokenizer.decode(results[0].sequences_ids[0])
    text = m2m.en2ja(text)
    print("システム(falcon-7b-with-m2m): " + text + "\n")
    return text


if __name__ == "__main__":
    import readline
    import time

    questions = [
        # "こんにちは",
        # "日本の首相は？",
        # "日本の首都は？",
        # "アメリカの首都は？",
        # "アメリカの大統領は？",
    ]
    while True:
        print("=========================================")
        if len(questions) > 0:
            q = questions.pop(0)
            print(f"ユーザー: {q}\n")

            start = time.time()
            output = reply(q)
            print(f"実行時間: {time.time() - start:.2f}秒")
        else:
            msg = input("ユーザー: ")
            start = time.time()
            reply(msg)
            print(f"実行時間: {time.time() - start:.2f}秒")
