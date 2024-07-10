# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import spacy
# import pandas as pd
# import pickle
# from model import Encoder, Decoder, Seq2Seq
# from dataset import ChatDataset

# spacy_en = spacy.load('en_core_web_sm')

# def tokenize_en(text):
#     return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# def build_vocab(data, tokenizer, min_freq=1):
#     token_freqs = {}
#     for sentence in data:
#         tokens = tokenizer(sentence.lower())
#         for token in tokens:
#             if token in token_freqs:
#                 token_freqs[token] += 1
#             else:
#                 token_freqs[token] = 1
#     vocab = {token: idx+2 for idx, (token, freq) in enumerate(token_freqs.items()) if freq >= min_freq}
#     vocab['<pad>'] = 0
#     vocab['<unk>'] = 1
#     vocab['<sos>'] = len(vocab)
#     vocab['<eos>'] = len(vocab) + 1
#     vocab_itos = {idx: token for token, idx in vocab.items()}
#     return vocab, vocab_itos

# # Load dataset
# data = pd.read_csv('chat_data.csv')
# query_vocab, query_vocab_itos = build_vocab(data['query'], tokenize_en)
# response_vocab, response_vocab_itos = build_vocab(data['response'], tokenize_en)

# train_data = ChatDataset('chat_data.csv', query_vocab, response_vocab, tokenize_en)
# train_iterator = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=lambda x: x)

# # Model hyperparameters
# INPUT_DIM = len(query_vocab)
# OUTPUT_DIM = len(response_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 512
# N_LAYERS = 2
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Seq2Seq(enc, dec, device).to(device)

# optimizer = optim.Adam(model.parameters())
# criterion = torch.nn.CrossEntropyLoss(ignore_index=query_vocab['<pad>'])

# def train(model, iterator, optimizer, criterion, clip):
#     model.train()
#     epoch_loss = 0
    
#     for i, batch in enumerate(iterator):
#         src = [item[0] for item in batch]
#         trg = [item[1] for item in batch]

#         src = torch.nn.utils.rnn.pad_sequence(src, padding_value=query_vocab['<pad>']).to(device)
#         trg = torch.nn.utils.rnn.pad_sequence(trg, padding_value=response_vocab['<pad>']).to(device)

#         optimizer.zero_grad()
#         output = model(src, trg)
        
#         output_dim = output.shape[-1]
#         output = output[1:].view(-1, output_dim)
#         trg = trg[1:].view(-1)
        
#         loss = criterion(output, trg)
#         loss.backward()
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
        
#         epoch_loss += loss.item()
        
#     return epoch_loss / len(iterator)

# N_EPOCHS = 13
# CLIP = 1

# for epoch in range(N_EPOCHS):
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

# # Save the model and vocabularies
# torch.save(model.state_dict(), 'seq2seq_model.pt')
# with open('query_vocab.pkl', 'wb') as f:
#     pickle.dump(query_vocab, f)
# with open('response_vocab.pkl', 'wb') as f:
#     pickle.dump(response_vocab, f)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import spacy
import pandas as pd
import pickle
from model import Encoder, Decoder, Seq2Seq
from attention import Attention
from dataset import ChatDataset  # Assuming this is defined elsewhere

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def build_vocab(data, tokenizer, min_freq=1):
    token_freqs = {}
    for sentence in data:
        tokens = tokenizer(sentence.lower())
        for token in tokens:
            if token in token_freqs:
                token_freqs[token] += 1
            else:
                token_freqs[token] = 1
    vocab = {token: idx+2 for idx, (token, freq) in enumerate(token_freqs.items()) if freq >= min_freq}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    vocab['<sos>'] = len(vocab)
    vocab['<eos>'] = len(vocab) + 1
    vocab_itos = {idx: token for token, idx in vocab.items()}
    return vocab, vocab_itos

# Load dataset
data = pd.read_csv('chat_data.csv')
query_vocab, query_vocab_itos = build_vocab(data['query'], tokenize_en)
response_vocab, response_vocab_itos = build_vocab(data['response'], tokenize_en)

train_data = ChatDataset('chat_data.csv', query_vocab, response_vocab, tokenize_en)
train_iterator = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=lambda x: collate_fn(x, query_vocab, response_vocab))

# Model hyperparameters
INPUT_DIM = len(query_vocab)
OUTPUT_DIM = len(response_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(ignore_index=query_vocab['<pad>'])

def collate_fn(batch, query_vocab, response_vocab):
    src_batch, trg_batch = zip(*batch)
    src_batch = [torch.tensor(x) for x in src_batch]
    trg_batch = [torch.tensor(x) for x in trg_batch]
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=query_vocab['<pad>'])
    trg_batch = torch.nn.utils.rnn.pad_sequence(trg_batch, padding_value=response_vocab['<pad>'])
    return src_batch, trg_batch

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

N_EPOCHS = 6
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

# Save the model and vocabularies
torch.save(model.state_dict(), 'seq2seq_model.pt')
with open('query_vocab.pkl', 'wb') as f:
    pickle.dump(query_vocab, f)
with open('response_vocab.pkl', 'wb') as f:
    pickle.dump(response_vocab, f)
