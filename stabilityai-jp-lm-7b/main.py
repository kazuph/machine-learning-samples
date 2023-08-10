import ctranslate2
import transformers
import torch

model = "stabilityai/japanese-stablelm-instruct-alpha-7b"
ct2_model = "stabilityai-jp-lm-7b_ct2_int8"

# cudaが使える場合はcudaを使う
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = ctranslate2.Generator(
    ct2_model, device=device)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model, use_fast=False)


# プロンプトを作成する
def prompt(user_query, inputs="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + user_query, ": "]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
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
    print("システム: " + text + "\n")
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
            reply(msg)
