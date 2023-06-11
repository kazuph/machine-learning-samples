import readline

# ユーザーからppoが良いかsfv_v2が良いかを選択してもらう
model = None
while True:
    msg = input("ppo or ppo_ct2 or sft_v2: ")
    if msg == "ppo":
        model = msg
        import ppo

        break
    elif msg == "ppo_ct2":
        model = msg
        import ppo_ct2

        break
    elif msg == "sft_v2":
        model = msg
        import sft_v2

        break
    else:
        print("ppoかsft_v2を入力してください")

# ユーザーからの入力を受け付ける
while True:
    msg = input("ユーザー: ")
    if model == "ppo":
        ppo.reply(msg)
    elif model == "ppo_ct2":
        ppo_ct2.reply(msg)
    elif model == "sft_v2":
        sft_v2.reply(msg)
