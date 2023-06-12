import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ppo = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"

# float16を利用する
tokenizer = AutoTokenizer.from_pretrained(ppo, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(ppo)

if torch.cuda.is_available():
    model = model.to("cuda")


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
    token_ids = tokenizer.encode(p, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=256,
            temperature=0.7,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :])
    output = output.replace("<NL>", "\n")
    print("システム(ppo): " + output + "\n")
    return output


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
            # 質問、回答、実行時間をcsvで保存する
            with open("ppo.csv", "a") as f:
                f.write(f"{q},{time.time() - start:.2f}\n")
            print(f"実行時間: {time.time() - start:.2f}秒")
        else:
            msg = input("ユーザー: ")
            reply(msg)
