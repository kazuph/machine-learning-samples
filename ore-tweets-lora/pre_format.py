import json
import re

# JSONファイルを読み込む
with open('tweets.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 出力を保存するリストを作成
output = []

# 各ツイートについて処理を行う
for tweet in data:
    full_text = tweet['tweet']['full_text']

    # 最初の「、」または「。」で区切る
    match = re.search(r'(.*?[、。])(.*)', full_text)

    # 「、」または「。」が見つかった場合
    if match:
        first_sentence = match.group(1)
        remaining_text = match.group(2)

        # 新しいフォーマットに整形
        formatted = {
            "input": first_sentence,
            "completion": remaining_text
        }

        # 結果をリストに追加
        output.append(formatted)

# 結果を新しいJSONファイルに書き込む
with open('formatted_tweets.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4)
