import ctranslate2
import transformers
import torch
import m2m

# cudaが使える場合はcudaを使う
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = ctranslate2.Generator("gpt_neox_chat_base_ct2", device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def reply(msg):
    msg = m2m.ja2en(msg)
    # 先頭に<human>:を追加
    msg = "<human>:" + msg
    # 末尾に<bot>:を追加
    msg = msg + "\n<bot>:"
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
    # <bot>:〜<human>:までを抽出
    text = text.split("<human>:")[0]
    # print(text)
    # text = m2m.en2ja(text)
    print("<human>(gpt-neox-20B-with-m2m): " + text)
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
            print(f"<human>: {q}\n")

            start = time.time()
            output = reply(q)
            print(f"実行時間: {time.time() - start:.2f}秒")
        else:
            msg = input("<human>: ")
            start = time.time()
            reply(msg)
            print(f"実行時間: {time.time() - start:.2f}秒")
