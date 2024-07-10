import pandas as pd
import torch
from torch.utils.data import Dataset

class ChatDataset(Dataset):
    def __init__(self, csv_file, query_vocab, response_vocab, tokenizer):
        self.data = pd.read_csv(csv_file)
        self.query_vocab = query_vocab
        self.response_vocab = response_vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query = self.data.iloc[idx, 0]
        response = self.data.iloc[idx, 1]

        query_tokens = self.tokenizer(query.lower())
        response_tokens = self.tokenizer(response.lower())

        query_indices = [self.query_vocab['<sos>']] + [self.query_vocab[token] for token in query_tokens] + [self.query_vocab['<eos>']]
        response_indices = [self.response_vocab['<sos>']] + [self.response_vocab[token] for token in response_tokens] + [self.response_vocab['<eos>']]

        return torch.tensor(query_indices), torch.tensor(response_indices)
