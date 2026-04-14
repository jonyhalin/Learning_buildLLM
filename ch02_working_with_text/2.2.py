import torch
from torch.utils.data import Dataset,DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)

        for i in range(0,len(token_ids)-max_length,stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx],self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

print("--- 测试 A: max_length=2, stride=2 ---")
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=2, stride=2)
data_iter = iter(dataloader)

for i in range(3):
    inputs, targets = next(data_iter)

    # 1. 转换为列表并去掉 Batch 维度 (取索引 0)
    input_ids = inputs[0].tolist()
    target_ids = targets[0].tolist()

    # 2. 解码为文本
    input_text = tokenizer.decode(input_ids)
    target_text = tokenizer.decode(target_ids)

    print(f"样本 {i + 1}:")
    print(f"  Input IDs:  {input_ids} -> 解码内容: '{input_text}'")
    print(f"  Target IDs: {target_ids} -> 解码内容: '{target_text}'")


print("\n--- 测试 B: max_length=8, stride=2 ---")
dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=8, stride=2)
data_iter = iter(dataloader)

for _ in range(3):
    inputs, targets = next(data_iter)
    print(f"Input: {inputs.tolist()}, Target: {targets.tolist()}")