import json
from torch.utils.data import Dataset

class ChatData(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = json.load(open(path, "r"))

        self.X = []
        for item in self.data:
            query = item["query"]
            response = item["response"]
            self.X.append(f"<startofstring> {query} <bot>: {response} <endofstring>")

        self.X_encoded = tokenizer(self.X, max_length=40, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
