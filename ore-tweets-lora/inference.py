import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"
peft_name = "lorappo-rinna-3.6b"
output_dir = "lorappo-rinna-3.6b-results"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, 
    device_map={"":0},
    # load_in_8bit=True,
    # device_map="auto",
)

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# LoRAモデルの準備
model = PeftModel.from_pretrained(
    model,
    peft_name,
    # device_map="auto"
)

# 評価モード
model.eval()


# プロンプトテンプレートの準備
def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""### 指示:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 回答:
"""
    else:
        result = f"""### 指示:
{data_point["instruction"]}

### 回答:
"""

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result


# テキスト生成関数の定義
def generate(instruction, input=None, maxTokens=256) -> str:
    # 推論
    prompt = generate_prompt({'instruction': instruction, 'input': input})
    input_ids = tokenizer(prompt,
                          return_tensors="pt",
                          truncation=True,
                          add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=maxTokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()
    # print(tokenizer.decode(outputs))

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "### 回答:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc + len(sentinel):]
            return result.replace("<NL>", "\n")  # <NL>→改行
        else:
            return 'Warning: Expected prompt template to be emitted.  Ignoring output.'
    else:
        return 'Warning: no <eos> detected ignoring output'


# ユーザーからの入力を受け付ける
import readline
while True:
    msg = input("ユーザー: ")
    msg = msg.replace("\n", "")
    print("{0}\n{1}".format(msg, generate(msg).replace("\n", "")))
