import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sft_v2 = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"

tokenizer = AutoTokenizer.from_pretrained(sft_v2, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(sft_v2)

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
    print("システム(sft_v2): " + output + "\n")
    return output
