# import torch
# import spacy
# import pandas as pd
# import pickle
# from model import Encoder, Decoder, Seq2Seq

# # Load the spaCy model
# spacy_en = spacy.load('en_core_web_sm')

# # Tokenizer function
# def tokenize_en(text):
#     return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

# # Load vocabularies
# with open('query_vocab.pkl', 'rb') as f:
#     query_vocab = pickle.load(f)
# with open('response_vocab.pkl', 'rb') as f:
#     response_vocab = pickle.load(f)

# # Rebuild vocab_itos
# query_vocab_itos = {idx: token for token, idx in query_vocab.items()}
# response_vocab_itos = {idx: token for token, idx in response_vocab.items()}

# # Adjust vocab sizes
# query_vocab_size = len(query_vocab)
# response_vocab_size = len(response_vocab)

# # Debugging: Print vocab sizes
# print(f"Query vocab size: {query_vocab_size}")
# print(f"Response vocab size: {response_vocab_size}")

# # Model hyperparameters
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 512
# N_LAYERS = 2
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

# enc = Encoder(query_vocab_size, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
# dec = Decoder(response_vocab_size, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Seq2Seq(enc, dec, device).to(device)

# # Load the trained model
# model.load_state_dict(torch.load('seq2seq_model.pt'))

# def translate_sentence(model, sentence, query_vocab, response_vocab, device, max_len=50):
#     model.eval()
    
#     tokens = tokenize_en(sentence)
#     tokens = [query_vocab['<sos>']] + [query_vocab.get(token, query_vocab['<unk>']) for token in tokens] + [query_vocab['<eos>']]
    
#     # Debugging: Print token indices
#     print(f"Token indices: {tokens}")
    
#     src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    
#     with torch.no_grad():
#         hidden, cell = model.encoder(src_tensor)
        
#     trg_indexes = [response_vocab['<sos>']]
    
#     for _ in range(max_len):
#         trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
#         with torch.no_grad():
#             output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
#         pred_token = output.argmax(1).item()
#         trg_indexes.append(pred_token)
        
#         if pred_token == response_vocab['<eos>']:
#             break
    
#     trg_tokens = [response_vocab_itos[i] for i in trg_indexes if i in response_vocab_itos]
    
#     return trg_tokens[1:-1]
# # while True:
# #     sentence = input("enter sentence")
# #     if sentence == "quit":
# #         break
# #     translation = translate_sentence(model, sentence, query_vocab, response_vocab, device)
# #     print(" ".join(translation))    
# sentence = "What is srilanka?"
# translation = translate_sentence(model, sentence, query_vocab, response_vocab, device)
# print(" ".join(translation))

import torch
import spacy
import pickle
from model import Encoder, Decoder, Seq2Seq
from attention import Attention

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def load_vocab(filepath):
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def load_vocab_itos(vocab):
    return {v: k for k, v in vocab.items()}

# Load vocabularies
query_vocab = load_vocab('query_vocab.pkl')
response_vocab = load_vocab('response_vocab.pkl')
query_vocab_itos = load_vocab_itos(query_vocab)
response_vocab_itos = load_vocab_itos(response_vocab)

# Load model
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

model.load_state_dict(torch.load('seq2seq_model.pt', map_location=device))
model.eval()

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    
    tokens = [token.lower() for token in tokenize_en(sentence)]
    tokens = [src_vocab['<sos>']] + [src_vocab.get(token, src_vocab['<unk>']) for token in tokens] + [src_vocab['<eos>']]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
    
    trg_indices = [trg_vocab['<sos>']]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indices[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indices.append(pred_token)
        
        if pred_token == trg_vocab['<eos>']:
            break
    
    trg_tokens = [response_vocab_itos[i] for i in trg_indices]
    
    return trg_tokens[1:-1]

def main():
    while True:
        query = input("Enter your query: ")
        if query.lower() == 'quit':
            break
        
        response = translate_sentence(query, query_vocab, response_vocab, model, device)
        print('Response:', ' '.join(response))

if __name__ == '__main__':
    main()

