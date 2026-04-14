import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = "Akwirw ier"
ids = tokenizer.encode(text)
print(f"原文本：{text}")
print(f"Token IDs:{ids}")

for token_id in ids:
    subword = tokenizer.decode([token_id])
    print(f"ID {token_id} -> '{subword}'")

res = tokenizer.decode(ids)
print(f"还原后文本：{res}")
print(f"是否还原成功：{text==res}")


